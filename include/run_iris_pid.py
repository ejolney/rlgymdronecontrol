import argparse
import gym
import gymfc
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI
import math
import os
import pickle

"""
This script evaluates a PID controller in the GymFC environment. This can be
used as a baseline for comparing to other control algorihtms in GymFC and also
to confirm the GymFC environment is setup and installed correctly.

The PID and mix settings reflect what was used in the following paper,

Koch, William, Renato Mancuso, Richard West, and Azer Bestavros. 
"Reinforcement Learning for UAV Attitude Control." arXiv 
preprint arXiv:1804.04154 (2018).

For reference, PID

PID Roll = [2, 10, 0.005]
PID PITCH = [10, 10, 0.005]
PID YAW = [4, 50, 0.0]

and mix for values throttle, roll, pitch, yaw,

rear right motor = [ 1.0, -1.0,  0.598, -1.0 ]  
front rear motor = [ 1.0, -0.927, -0.598,  1.0 ]
rear left motor  = [ 1.0,  1.0,  0.598,  1.0 ]
front left motor = [ 1.0,  0.927, -0.598, -1.0 ]

PID terms were found first using the Ziegler–Nichols method and then manually tuned
for increased response. The Iris quadcopter does not have an X frame therefore 
a custom mixer is required. Using the mesh files found in the Gazebo models they were
imported into a CAD program and the motor constraints were measured. Using these
values the mix calculater found here, https://www.iforce2d.net/mixercalc, was
used to derive the values. The implmementation of the PID controller can be found here,
https://github.com/ivmech/ivPID/blob/master/PID.py, windup has been removed so 
another variable was not introduced.
"""


def plot_step_response(desired, actual,
			end=1., title=None,
			step_size=0.001, threshold_percent=0.1):
	"""
	Args:
	    threshold (float): Percent of the start error
	"""

	#actual = actual[:,:end,:]
	end_time = len(desired) * step_size
	t = np.arange(0, end_time, step_size)

	#desired = desired[:end]
	threshold = threshold_percent * desired

	plot_min = -math.radians(350)
	plot_max = math.radians(350)

	subplot_index = 3
	num_subplots = 3

	f, ax = plt.subplots(num_subplots, sharex=True, sharey=False)
	f.set_size_inches(10, 5)
	if title:
		plt.suptitle(title)
	ax[0].set_xlim([0, end_time])
	res_linewidth = 2
	linestyles = ["c", "m", "b", "g"]
	reflinestyle = "k--"
	error_linestyle = "r--"

	# Always
	ax[0].set_ylabel("Roll (rad/s)")
	ax[1].set_ylabel("Pitch (rad/s)")
	ax[2].set_ylabel("Yaw (rad/s)")

	ax[-1].set_xlabel("Time (s)")


	""" ROLL """
	# Highlight the starting x axis
	ax[0].axhline(0, color="#AAAAAA")
	ax[0].plot(t, desired[:,0], reflinestyle)
	ax[0].plot(t, desired[:,0] -  threshold[:,0] , error_linestyle, alpha=0.5)
	ax[0].plot(t, desired[:,0] +  threshold[:,0] , error_linestyle, alpha=0.5)

	r = actual[:,0]
	ax[0].plot(t[:len(r)], r, linewidth=res_linewidth, color="#ff0000")

	ax[0].grid(True)



	""" PITCH """

	ax[1].axhline(0, color="#AAAAAA")
	ax[1].plot(t, desired[:,1], reflinestyle)
	ax[1].plot(t, desired[:,1] -  threshold[:,1] , error_linestyle, alpha=0.5)
	ax[1].plot(t, desired[:,1] +  threshold[:,1] , error_linestyle, alpha=0.5)
	p = actual[:,1]
	ax[1].plot(t[:len(p)],p, linewidth=res_linewidth, color="#00ff00")
	ax[1].grid(True)


	""" YAW """
	ax[2].axhline(0, color="#AAAAAA")
	ax[2].plot(t, desired[:,2], reflinestyle)
	ax[2].plot(t, desired[:,2] -  threshold[:,2] , error_linestyle, alpha=0.5)
	ax[2].plot(t, desired[:,2] +  threshold[:,2] , error_linestyle, alpha=0.5)
	y = actual[:,2]
	ax[2].plot(t[:len(y)],y , linewidth=res_linewidth, color="#0000ff")
	ax[2].grid(True)

	plt.show()

class Policy(object):
	def action(self, state, sim_time=0, desired=np.zeros(3), actual=np.zeros(3) ):
		pass
	def reset(self):
		pass

class PIDPolicy(Policy):
	def __init__(self):
		self.r = [2, 10, 0.005]
		self.p = [10, 10, 0.005]
		self.y = [4, 50, 0.0]
		self.controller = PIDController(pid_roll = self.r, pid_pitch = self.p, pid_yaw =self.y )

	def action(self, state, sim_time=0, desired=np.zeros(3), actual=np.zeros(3) ):
		# Convert to degrees
		desired = list(map(math.degrees, desired))
		actual = list(map(math.degrees, actual))
		motor_values = np.array(self.controller.calculate_motor_values(sim_time, desired, actual))
		# Need to scale from 1000-2000 to -1:1
		return np.array( [ (m - 1000)/500  - 1 for m in motor_values])

	def reset(self):
		self.controller = PIDController(pid_roll = self.r, pid_pitch = self.p, pid_yaw = self.y )

def eval(env, pi, exps=1):
	for i in range(exps):
		actuals = []
		desireds = []
		pi.reset()
		ob = env.reset()
		while True:
			desired = env.omega_target
			actual = env.omega_actual
			# PID only needs to calculate error between desired and actual y_e
			ac = pi.action(ob, env.sim_time, desired, actual)
			ob, reward, done, info = env.step(ac)
			actuals.append(actual)
			desireds.append(desired)
			if done:
				break
#		env.reset()
		plot_step_response(np.array(desireds), np.array(actuals), title='tried')
	env.close()
	return desireds, actuals

class ModelEval:
	def __init__(self, model_name, env_id, seed=17):
		self.model_name = model_name
		self.env_id = env_id
		self.seed = seed
		print('Evaluating PID')

	def evalModel(self, totaleps=1, err_thresh=.1):
		# Timestep Data
		actions = []
		observations = []
		rewards = []
		errors = []
		desireds = []
		actuals = []

		# Episode Data
		episodes = []
		ep_num = []

		# Setup name
		self.model_dir = '/home/acit/gymfc/rlgymdronecontrol/gymlogs/models/' # Save extracted model dir
		self.model_path = self.model_dir+self.model_name+'/'+self.model_name

		# Setup gym
		env = gym.make(self.env_id)
		# Seed Set
		rank = MPI.COMM_WORLD.Get_rank()
		workerseed = self.seed + 1000000 * rank
		env.seed(workerseed)
		pi = PIDPolicy()

		pi.reset()
		env.reset()
		ob = env.reset()  # reset object for pi

		print('----------=================--------------')
		print('rank: ', rank, 'workerseed: ', workerseed)
		print('----------=================--------------')

		env.render()
		input('Press enter to continue')

		for eps in range (totaleps):
			print(eps)	
			#action = pi.action(ob, env.sim_time, desired, actual)
			#ob, r, done, info = env.step(action)

			# Initialize records
			done = False
			ittr = 0
			ep_data={}
			headers = ['stats', 'actions', 'observations', 'rewards', 
				'roll_err', 'pitch_err', 'yaw_err',
				'droll_v', 'dpitch_v', 'dyaw_v',
				'aroll_v', 'apitch_v', 'ayaw_v']
			for header in headers: ep_data[header] = []

			# Run Environment
			while done != True:
				#action = rand_action(env)
				des = env.omega_target  # desired angular velocities
				actual = env.omega_actual  # current angular velocities 
				action = pi.action(ob, env.sim_time, des, actual)  # choose action	
				ob, r, done, info = env.step(action)  # perform action

				# Record Data
				ep_data['actions'].append(action)
				ep_data['observations'].append(ob)
				ep_data['rewards'].append(r)

				# Errors			
				ep_data['roll_err'].append(abs(ob[0]))
				ep_data['pitch_err'].append(abs(ob[1]))
				ep_data['yaw_err'].append(abs(ob[2]))

				# Step functions
				ep_data['droll_v'].append(env.omega_target[0])
				ep_data['dpitch_v'].append(env.omega_target[1])
				ep_data['dyaw_v'].append(env.omega_target[2])
				ep_data['aroll_v'].append(env.omega_actual[0])
				ep_data['apitch_v'].append(env.omega_actual[1])
				ep_data['ayaw_v'].append(env.omega_actual[2])

				ittr += 1
			pi.reset()


			episodes.append(ep_data) 

			env.reset()
			ep_num.append(eps)
		env.close()
		self.eps = episodes

		self.procEval()

	def procEval(self):
		self.proc_eval = ProcEval(self)
		self.proc_eval.analyzeEval()

	def saveEval(self, save_name=None):
		if save_name: eval_file = self.model_dir+'/'+save_name+'.obj'
		else: eval_file = self.model_path+'_eval.obj'

		try:
			with open(eval_file, 'wb') as evalfile:
				pickle.dump(self, evalfile)
		except IOError:
			print("I/O error")

class ProcEval:
########################################## Info ################################################
# Analysis and plotting object for model evaluation
################################################################################################
	def __init__(self, model_eval):
		self.me = model_eval
		self.eval_size = len(self.me.eps)
	

	def clearAll(self):
		self.eps_avg_r = []  # Average ep rwd
		self.eps_fin_r = []  # Final reward value of episode
		self.eps_del_r = []  # diff of init and fin r
		self.eps_max_r = []  # maximum episode reward

		self.eps_avg_rerr = []  # average roll error of all episodes
		self.eps_avg_perr = []  # average pitch error of all episodes
		self.eps_avg_yerr = []  # average yaw error of all episodes

		self.eps_r_rise = []  # risetime of roll velocity
		self.eps_p_rise = []  # risetime of pitch velocity
		self.eps_y_rise = []  # risetime of yaw velocity


	def analyzeEval(self, errt=0.1):
		self.clearAll()

		for ep in self.me.eps:
			# Rewards
			self.eps_avg_r.append(np.mean(ep['rewards']))
			self.eps_fin_r.append(ep['rewards'][-1:])
			self.eps_del_r.append(abs(ep['rewards'][-1:]-ep['rewards'][0]))
			self.eps_max_r.append(max(ep['rewards']))

			# Errors
			self.eps_avg_rerr.append(np.mean(ep['roll_err']))
			self.eps_avg_perr.append(np.mean(ep['pitch_err']))
			self.eps_avg_yerr.append(np.mean(ep['yaw_err']))

			# Rise Times
			self.eps_r_rise.append(self.riseTime(ep['aroll_v'], ep['droll_v'][0], errt))
			self.eps_p_rise.append(self.riseTime(ep['apitch_v'], ep['dpitch_v'][0], errt))
			self.eps_y_rise.append(self.riseTime(ep['ayaw_v'], ep['dyaw_v'][0], errt)) 


	def riseTime(self, actls, targ, errt):
		rise_time = np.where(abs(actls - targ)<errt)[0]
		if len(rise_time)>0: return rise_time[0]
		else: return len(actls)


	def plotResponse(self, ep_ind=0):
		fig, (ax_rsp, ax_err) = plt.subplots(2,1)
		its = list(range(len(self.me.eps[ep_ind]['actions'])))

		# Response Plot
		ax_rsp.plot(its, self.me.eps[ep_ind]['droll_v'], label='Target Roll Vel', color='#ffaaaa')
		ax_rsp.plot(its, self.me.eps[ep_ind]['aroll_v'], label='Roll Vel', color='#ff0000', linestyle = '--')
		ax_rsp.plot(its, self.me.eps[ep_ind]['dpitch_v'], label='Target Pitch Vel', color='#aaffaa')
		ax_rsp.plot(its, self.me.eps[ep_ind]['apitch_v'], label='Pitch Vel', color='#00ff00', linestyle = '--')
		ax_rsp.plot(its, self.me.eps[ep_ind]['dyaw_v'], label='Target Yaw Vel', color='#aaaaff')
		ax_rsp.plot(its, self.me.eps[ep_ind]['ayaw_v'], label='Yaw Vel', color='#0000ff', linestyle = '--')
		ax_rsp.set_title('Response')
		ax_rsp.set_xlabel('Step')
		ax_rsp.set_ylabel('Velocity (rad/s)')
		ax_rsp.legend()

		# Errors Plot
		ax_err.plot(its, self.me.eps[ep_ind]['roll_err'], label='Roll Error')
		ax_err.plot(its, self.me.eps[ep_ind]['pitch_err'], label='Pitch Error')
		ax_err.plot(its, self.me.eps[ep_ind]['yaw_err'], label='Yaw Error')
		ax_err.plot(its, np.zeros(len(self.me.eps[ep_ind]['roll_err'])), color='#000000')
		ax_err.set_title('State Error')
		ax_err.set_xlabel('Step')
		ax_err.set_ylabel('Error')
		ax_err.legend()
		plt.subplots_adjust(hspace = .4)
		plt.show()



	def plotEps(self):
	# Plot the response and error plots for an indicated episode in the group of
	# examined episode tests.

		# plot
		fig, (ax_rwd, ax_fin, ax_dif) = plt.subplots(3,1)
		fig.suptitle('Gym_FC')

		# Ep Avera Plot
		ax_rwd.plot(list(range(len(self.me.eps))), self.eps_avg_r, label='Average Reward')
		ax_rwd.set_title('Average Reward')
		ax_rwd.set_xlabel('Episode')
		ax_rwd.set_ylabel('Average Reward')
		ax_rwd.legend()

		# Ep Final Reward Plot
		ax_fin.plot(list(range(len(self.me.eps))), self.eps_fin_r, label='Final Reward')
		ax_rwd.set_title('Final Reward')
		ax_fin.set_xlabel('Episode')
		ax_fin.set_ylabel('Reward')
		ax_fin.legend()

		# Ep Reward Difference (Beginning - End) Plot
		ax_dif.plot(list(range(len(self.me.eps))), self.eps_del_r, label='Reward Progress')
		ax_dif.set_title('Episode Reward Difference')
		ax_dif.set_xlabel('Episode')
		ax_dif.set_ylabel('Reward Difference')
		ax_dif.legend()

		plt.show()

"""
This is essentially a port from Betaflight

For the iris the motors have the following constraints,
https://www.iforce2d.net/mixercalc/
4 2 409
2 1 264
4 3 264
3 1 441
4 1 500
2 3 500

"""

class PIDController(object):
	FD_ROLL = 0
	FD_PITCH = 1
	FD_YAW = 2
	PTERM_SCALE = 0.032029
	ITERM_SCALE = 0.244381
	DTERM_SCALE = 0.000529
	minthrottle = 1070
	maxthrottle = 2000

	def __init__(self, pid_roll = [40, 40, 30], pid_pitch = [58, 50, 35], pid_yaw = [80, 45, 20], itermLimit=150):

		# init gains and scale
		self.Kp = [pid_roll[0], pid_pitch[0], pid_yaw[0]]
		self.Kp = [self.PTERM_SCALE * p for p in self.Kp]

		self.Ki = [pid_roll[1], pid_pitch[1], pid_yaw[1]]
		self.Ki = [self.ITERM_SCALE * i for i in self.Ki]

		self.Kd = [pid_roll[2], pid_pitch[2], pid_yaw[2]]
		self.Kd = [self.DTERM_SCALE * d for d in self.Kd]


		self.itermLimit = itermLimit 

		self.previousRateError = [0]*3
		self.previousTime = 0 
		self.previous_motor_values = [self.minthrottle]*4
		self.pid_rpy = [PID(*pid_roll), PID(*pid_pitch), PID(*pid_yaw)]

	def calculate_motor_values(self, current_time, sp_rates, gyro_rates):
		rpy_sums = []
		for i in range(3):
			self.pid_rpy[i].SetPoint = sp_rates[i]
			self.pid_rpy[i].update(current_time, gyro_rates[i])
			rpy_sums.append(self.pid_rpy[i].output)
		return self.mix(*rpy_sums)

	def constrainf(self, amt, low, high):
        # From BF src/main/common/maths.h
		if amt < low:
			return low
		elif amt > high:
			return high
		else:
			return amt

	def mix(self, r, p, y):
		PID_MIXER_SCALING = 1000.0
		pidSumLimit = 10000.#500
		pidSumLimitYaw = 100000.#1000.0#400
		motorOutputMixSign = 1
		motorOutputRange = self.maxthrottle - self.minthrottle# throttle max - throttle min 
		motorOutputMin = self.minthrottle

		currentMixer=[ 
			[ 1.0, -1.0,  0.598, -1.0 ],          # REAR_R
			[ 1.0, -0.927, -0.598,  1.0 ],          # RONT_R
			[ 1.0,  1.0,  0.598,  1.0 ],          # REAR_L
			[ 1.0,  0.927, -0.598, -1.0 ],          # RONT_L
		]
		mixer_index_throttle = 0
		mixer_index_roll = 1
		mixer_index_pitch = 2 
		mixer_index_yaw = 3

		scaledAxisPidRoll = self.constrainf(r, -pidSumLimit, pidSumLimit) / PID_MIXER_SCALING
		scaledAxisPidPitch = self.constrainf(p, -pidSumLimit, pidSumLimit) / PID_MIXER_SCALING
		scaledAxisPidYaw = self.constrainf(y, -pidSumLimitYaw, pidSumLimitYaw) / PID_MIXER_SCALING
		scaledAxisPidYaw = -scaledAxisPidYaw

		# Find roll/pitch/yaw desired output
		motor_count = 4
		motorMix = [0]*motor_count
		motorMixMax = 0
		motorMixMin = 0
		# No additional throttle, in air mode
		throttle = 0
		motorRangeMin = 1000
		motorRangeMax = 2000

		for i in range(motor_count):
			mix = (scaledAxisPidRoll  * currentMixer[i][1] +
				scaledAxisPidPitch * currentMixer[i][2] +
				scaledAxisPidYaw   * currentMixer[i][3])

			if mix > motorMixMax:
				motorMixMax = mix
			elif mix < motorMixMin:
				motorMixMin = mix
			motorMix[i] = mix

		motorMixRange = motorMixMax - motorMixMin
		#print("range=", motorMixRange)

		if motorMixRange > 1.0:
			for i in range(motor_count): 
				motorMix[i] /= motorMixRange
			# Get the maximum correction by setting offset to center when airmode enabled
			throttle = 0.5

		else:
			# Only automatically adjust throttle when airmode enabled. Airmode logic is always active on high throttle
			throttleLimitOffset = motorMixRange / 2.0
			throttle = self.constrainf(throttle, 0.0 + throttleLimitOffset, 1.0 - throttleLimitOffset)

		motor = []
		for i in range(motor_count):
			motorOutput = motorOutputMin + (motorOutputRange * (motorOutputMixSign * motorMix[i] + throttle * currentMixer[i][mixer_index_throttle]))
			motorOutput = self.constrainf(motorOutput, motorRangeMin, motorRangeMax);
			motor.append(motorOutput)

		motor = list(map(int, np.round(motor)))
		return motor


	def is_airmode_active(self):
		return True

	def reset(self):
		for pid in self.pid_rpy:
			pid.clear()

# This file is part of IvPID.
# Copyright (C) 2015 Ivmech Mechatronics Ltd. <bilgi@ivmech.com>
#
# IvPID is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# IvPID is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# title           :PID.py
# description     :python pid controller
# author          :Caner Durmusoglu
# date            :20151218
# version         :0.1
# notes           :
# python_version  :2.7
# ==============================================================================

"""Ivmech PID Controller is simple implementation of a Proportional-Integral-Derivative (PID) Controller in the Python Programming Language.
More information about PID Controller: http://en.wikipedia.org/wiki/PID_controller
"""
import time

class PID:
	"""PID Controller
	"""

	def __init__(self, P=0.2, I=0.0, D=0.0):

		self.Kp = P
		self.Ki = I
		self.Kd = D

		self.sample_time = 0.00
		self.current_time = 0
		self.last_time = self.current_time

		self.clear()

	def clear(self):
		"""Clears PID computations and coefficients"""
		self.SetPoint = 0.0

		self.PTerm = 0.0
		self.ITerm = 0.0
		self.DTerm = 0.0
		self.last_error = 0.0

		# Windup Guard
		self.int_error = 0.0
		self.windup_guard = 20.0

		self.output = 0.0

	def update(self, current_time, feedback_value):
		"""Calculates PID value for given reference feedback

		.. math::
		u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}

		.. figure:: images/pid_1.png
		:align:   center

		Test PID with Kp=1.2, Ki=1, Kd=0.001 (test_pid.py)

		"""
		error = self.SetPoint - feedback_value

		delta_time = current_time - self.last_time
		delta_error = error - self.last_error

		if (delta_time >= self.sample_time):
			self.PTerm = self.Kp * error
			self.ITerm += error * delta_time

			if (self.ITerm < -self.windup_guard):
				self.ITerm = -self.windup_guard
			elif (self.ITerm > self.windup_guard):
				self.ITerm = self.windup_guard

			self.DTerm = 0.0
			if delta_time > 0:
				self.DTerm = delta_error / delta_time

			# Remember last time and last error for next calculation
			self.last_time =current_time
			self.last_error = error

			#print("P=", self.PTerm, " I=", self.ITerm, " D=", self.DTerm)
			self.output = self.PTerm + (self.Ki * self.ITerm) + (self.Kd * self.DTerm)

	def setKp(self, proportional_gain):
		"""Determines how aggressively the PID reacts to the current error with setting Proportional Gain"""
		self.Kp = proportional_gain

	def setKi(self, integral_gain):
		"""Determines how aggressively the PID reacts to the current error with setting Integral Gain"""
		self.Ki = integral_gain

	def setKd(self, derivative_gain):
		"""Determines how aggressively the PID reacts to the current error with setting Derivative Gain"""
		self.Kd = derivative_gain

	def setWindup(self, windup):
		"""Integral windup, also known as integrator windup or reset windup,
		refers to the situation in a PID feedback controller where
		a large change in setpoint occurs (say a positive change)
		and the integral terms accumulates a significant error
		during the rise (windup), thus overshooting and continuing
		to increase as this accumulated error is unwound
		(offset by errors in the other direction).
		The specific problem is the excess overshooting.
		"""
		self.windup_guard = windup

	def setSampleTime(self, sample_time):
		"""PID that should be updated at a regular interval.
		Based on a pre-determined sampe time, the PID decides if it should compute or return immediately.
		"""
		self.sample_time = sample_time


def loadEval(model_name, eval_name=None):
	if eval_name: 
		if eval_name.endswith('.obj'):
			eval_filename = eval_name+'.obj'
	else: eval_filename = model_name+'_eval.obj'

	load_path = os.environ['GYMFC_EXP_MODELSDIR']+model_name+'/'+ eval_filename
	
	if os.path.isfile(load_path):
		with open(load_path, 'rb') as evalfile:
			model_eval = pickle.load(evalfile)
	else:
		raise Exception('Eval obj not found at: {}.'.format(load_path))
			
	return model_eval


def setVars():
	# Set up env Variables
	os.environ['GYMFC_CONFIG'] = '/home/acit/gymfc/examples/configs/iris.config'
	os.environ['OPENAI_LOGDIR'] = '/home/acit/gymfc/rlgymdronecontrol/gymlogs/'
	os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv'
	os.environ['GYMFC_EXP_MODELSDIR'] = '/home/acit/gymfc/rlgymdronecontrol/gymlogs/models/'


def main(env_id, seed):
	setVars()
#	env = gym.make(env_id)
#	rank = MPI.COMM_WORLD.Get_rank()
#	workerseed = seed + 1000000 * rank
#	env.seed(workerseed)
#	pi = PIDPolicy()

#	desireds, actuals = eval(env, pi, exps=2)
#	err = abs(np.array(desireds) - np.array(actuals))
#	print('errmean: ', np.mean(err[:,0]), np.mean(err[:,1]), np.mean(err[:,2]))
#	print('rank: ', rank, 'workerseed: ', workerseed)
#	title = "PID Step Response in Environment {}".format(env_id)
#	plot_step_response(np.array(desireds), np.array(actuals), title=title)
	#plot_step_response(np.array(desireds)-np.array(desireds), err, title=title)
	me = ModelEval('pid', 'AttFC_GyroErr1_M4_Ep-v0')
	me.evalModel()
	me.proc_eval.plotResponse()


if __name__ == "__main__":

	parser = argparse.ArgumentParser("Evaluate a PID controller")
	parser.add_argument('--env-id', help="The Gym environement ID", type=str,
		        default="AttFC_GyroErr-MotorVel_M4_Ep-v0")
	parser.add_argument('--seed', help='RNG seed', type=int, default=17)

	args = parser.parse_args()
	current_dir = os.path.dirname(__file__)
	config_path ='/home/acit/gymfc/examples/configs/iris.config'#os.path.join(current_dir,
		  #             "../configs/iris.config")
	print ("Loading config from ", config_path)
	os.environ["GYMFC_CONFIG"] = config_path

	main(args.env_id, args.seed)
