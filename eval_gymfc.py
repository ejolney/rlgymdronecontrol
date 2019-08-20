import os
import numpy as np
import pandas as pd
from baselines.common import tf_util as U
import gymfc
import gym
import matplotlib.pyplot as plt
import train_gymfc as tg


class TrainLog():
###
# Training log object that reads and stores training log with data as available
# ex. 	>log_file = 'path/to/csv/log_file.csv'
#	>tlog = TrainLog(log_file)
# tlog is now an inclusion of all of the available data from that training session
# ex.	>tlog.episodes  # Episodes so far list
#	array([ 1, 2, 3, ...])
###

	def __init__(self, log_file=''):
		if os.path.isfile(log_file) and log_file.endswith('.csv'):
			log = pd.read_csv(log_file)
			self.parseLog(log)
		else:
			raise Exception('Log .csv file not found at: {}. Use TrainLog([path of .csv])'.format(log_file))


	def parseLog(self, log):
		self.ep_avg_len = log['EpLenMean'].values  # average number of timesteps per ep since last iter

		# RL metric
		self.rew_avg = log['EpRewMean'].values  # average ep rwd since last iter

		# Time keeping
		self.ep_this_iter= log['EpThisIter'].values  # episodes since last iter
		self.episodes = log['EpisodesSoFar'].values  # number of episodes completed
		self.timesteps = log['TimestepsSoFar'].values  # number of steps performed
		self.time_secs = log['TimeElapsed'].values  # seconds since learning started

		# Other Logged metrics
		self.ev_tdlam_before = log['ev_tdlam_before'].values
		self.ent_loss = log['loss_ent'].values
		self.kl_loss = log['loss_kl'].values
		self.entpen_pol_loss = log['loss_pol_entpen'].values
		self.surr_pol_loss = log['loss_pol_surr'].values
		self.vf_loss = log['loss_vf_loss'].values


	def plotLog(self):
		# Setup Plot
		fig, ax = plt.subplots()
		ax.plot(self.episodes, self.rew_avg)
		ax.set_title('Training Progress')
		ax.set_xlabel('Step')
		ax.set_ylabel('Avg_reward Per Episode')
		plt.show()	


def plot(eps, ep_ind=0):
# Plot the response and error plots for an indicated episode in the group of
# examined episode tests.

	# Get episode stats
	ep_stats = epStats(eps)
	# plot
	fig, (ax_rwd, ax_rsp, ax_err) = plt.subplots(3,1)
	fig.suptitle('Gym_FC')

	its = list(range(len(eps[ep_ind]['actions'])))

	# Ep Stats Plot
	ax_rwd.plot(list(range(len(eps))), ep_stats['avg_r'], label='Average Reward')
	ax_rwd.plot(list(range(len(eps))), ep_stats['fin_r'], label='Final Reward')
	ax_rwd.plot(list(range(len(eps))), ep_stats['del_r'], label='Reward Progress')
	ax_rwd.set_title('Average Reward')
	ax_rwd.set_xlabel('Episode')
	ax_rwd.set_ylabel('Average Reward')

	# Response Plot
	ax_rsp.plot(its, eps[ep_ind]['droll_v'], label='Target Roll Vel', color='#ffaaaa')
	ax_rsp.plot(its, eps[ep_ind]['aroll_v'], label='Roll Vel', color='#ff0000')
	ax_rsp.plot(its, eps[ep_ind]['dpitch_v'], label='Target Pitch Vel', color='#aaffaa')
	ax_rsp.plot(its, eps[ep_ind]['apitch_v'], label='Pitch Vel', color='#00ff00')
	ax_rsp.plot(its, eps[ep_ind]['dyaw_v'], label='Target Yaw Vel', color='#aaaaff')
	ax_rsp.plot(its, eps[ep_ind]['ayaw_v'], label='Yaw Vel', color='#0000ff')
	ax_rsp.set_title('Response')
	ax_rsp.set_xlabel('Step')
	ax_rsp.set_ylabel('Velocity (rad/s)')
	ax_rsp.legend()

	# Errors Plot
	ax_err.plot(its, eps[ep_ind]['roll_err'], label='Roll Error')
	ax_err.plot(its, eps[ep_ind]['pitch_err'], label='Pitch Error')
	ax_err.plot(its, eps[ep_ind]['yaw_err'], label='Yaw Error')
	ax_err.plot(its, np.zeros(len(eps[ep_ind]['roll_err'])), color='#000000')
	ax_err.set_title('State Error')
	ax_err.set_xlabel('Step')
	ax_err.set_ylabel('Error')
	ax_err.legend()

	plt.show()



def evalModel(model_path, env_id):

	# Data Record
	actions = []
	observations = []
	rewards = []
	errors = []
	desireds = []
	actuals = []

	#eps = 0
	toteps = 2
	episodes = []
	ep_num = []

	# Set up parameters for quick initialization
	train_params = tg.trainParams()
	train_params.num_timesteps = 1
	train_params.timesteps_per_actorbatch = 1000
	train_params.optim_epochs = 1
	train_params.optim_batchsize = 1
	train_params.seed = 17

	policy_params = tg.policyParams()

	# Initialize and load model
	with U.tf.Graph().as_default():		# Allow Re-running of tf
		pi = tg.train(train_params, policy_params, env_id)
	U.load_variables(model_path)

	env = gym.make(env_id)
	env.reset()
	ob = env.reset()  # reset object for pi

	#env.render()
	#input('Press enter to continue')

	for eps in range (toteps):
		print(eps)	
		#action = rand_action(env)
		action = pi.act(stochastic=False, ob=ob)[0]
		ob, r, done, info = env.step(action)

		# Initialize records
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
			action = pi.act(stochastic=False, ob=ob)[0]  # choose action	
			ob, r, done, info = env.step(action)  # perform action
			des = env.omega_target  # desired angular velocities
			actual = env.omega_actual  # current angular velocities 

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

		# Ep Stats
		stats = {'avg_r' : np.mean(ep_data['rewards']),  # Average ep rwd
			'fin_r' : r,  # Final reward value of episode
			'del_r' : abs(r-ep_data['rewards'][0])  #diff of init and fin r
		}
		ep_data['stats'] = stats

		episodes.append(ep_data) 

		env.reset()
		ep_num.append(eps)
	env.close()
	return episodes


def epStats(eps):
	ep_stats = {}
	for stat_header in eps[0]['stats'].keys(): ep_stats[stat_header] = []
	for ep in range(len(eps)):
		for key in eps[ep]['stats'].keys():
			ep_stats[key].append(eps[ep]['stats'][key])
	return ep_stats

def debugInitialize():
	os.environ['GYMFC_CONFIG'] = '/home/acit/gymfc/examples/configs/iris.config'

	# Gym Environment
	env = ('AttFC_GyroErr-MotorVel_M4_Ep-v0', 'AttFC_GyroErr-MotorVel_M4_Con-v0',
		'AttFC_GyroErr1_M4_Ep-v0 - AttFC_GyroErr10_M4_Ep-v0')
	model_path = '/home/acit/gymlogs/models/testModel'
	eps = evalModel(model_path, env[0])
	return eps


if __name__ == '__main__':
	print('begin')
	
