import os
import numpy as np
import pandas as pd
from baselines.common import tf_util as U
import gymfc
import gym
import sqlite3
import matplotlib.pyplot as plt
import train_gymfc as tg


class TrainLog:
###
# Training log object that reads and stores training log with data as available
# ex. 	>log_file = 'path/to/csv/log_file.csv'
#	>tlog = TrainLog(log_file)
# tlog is now an inclusion of all of the available data from that training session
# ex.	>tlog.episodes  # Episodes so far list
#	array([ 1, 2, 3, ...])
###

	def __init__(self, model_dir=''):
		if model_dir:
			log_file = model_dir+'/log.csv'
			metadata_file = model_dir+'/metadata.csv'

			# Read log file
			if os.path.isfile(log_file):
				log = pd.read_csv(log_file)
				self.parseLog(log)
			else:
				raise Exception('Log .csv file not found at: {}.'.format(log_file))

			# Read Metadata file
			if os.path.isfile(metadata_file):
				md = pd.read_csv(metadata_file)
				# Parse metadata file
				#  train params
				self.num_timesteps = md['num_timesteps']
				self.timesteps_per_actorbatch = md['timesteps_per_actorbatch']
				self.clip_param = md['clip_param']
				self.entcoeff = md['entcoeff']
				self.optim_epochs = md['optim_epochs']
				self.optim_stepsize = md['optim_stepsize']
				self.optim_batchsize = md['optim_batchsize']
				self.gamma = md['gamma']
				self.lam = md['lam']
				self.schedule= md['schedule']
				self.seed = md['seed']
				self.model_path = md['model_path']
				self.model_dir = md['model_dir']
				self.model_name =  md['model_name']
				#  policy params
				self.nodes_per_layer = md['nodes_per_layer']
				self.num_layers = md['num_layers']

			else:
				raise Exception('Metadata .csv file not found at: {}.'.format(log_file))


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


class ModelEval:

	def __init__(self, model_name, env_id):
		self.model_name = model_name
		self.env_id = env_id
		print('Evaluating model {}'.format(model_name))


	def evalModel(self, toteps=1):

		# Data Record
		actions = []
		observations = []
		rewards = []
		errors = []
		desireds = []
		actuals = []

		episodes = []
		ep_num = []
		self.eps_avg_r = []
		self.eps_fin_r = []
		self.eps_del_r = []

		# Set up parameters for quick initialization
		tp = tg.trainParams()
		tp.num_timesteps = 1
		tp.timesteps_per_actorbatch = 1000
		tp.optim_epochs = 1
		tp.optim_batchsize = 1
		tp.seed = 17
		
		pp = tg.policyParams()

		# Initialize and load model
		with U.tf.Graph().as_default():		# Allow Re-running of tf
			pi = tg.train(tp, pp, self.env_id)

		# Load Model
		tp.modelName(self.model_name) # Set up name
		self.model_dir = tp.model_dir # Save extracted model dir
		self.model_path = tp.model_path # Save extracted model path
		U.load_variables(tp.model_path)

		# Make Training Log
		self.train_log = TrainLog(tp.model_dir)
		
		# Setup gym
		env = gym.make(self.env_id)
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
			self.eps_avg_r.append(stats['avg_r'])
			self.eps_fin_r.append(stats['fin_r'])
			self.eps_del_r.append(stats['del_r'])
			ep_data['stats'] = stats

			episodes.append(ep_data) 

			env.reset()
			ep_num.append(eps)
		env.close()
		self.eps = episodes


	def plotResponse(self, ep_ind=0):
		fig, (ax_rsp, ax_err) = plt.subplots(2,1)
		its = list(range(len(self.eps[ep_ind]['actions'])))

		# Response Plot
		ax_rsp.plot(its, self.eps[ep_ind]['droll_v'], label='Target Roll Vel', color='#ffaaaa')
		ax_rsp.plot(its, self.eps[ep_ind]['aroll_v'], label='Roll Vel', color='#ff0000', linestyle = '--')
		ax_rsp.plot(its, self.eps[ep_ind]['dpitch_v'], label='Target Pitch Vel', color='#aaffaa')
		ax_rsp.plot(its, self.eps[ep_ind]['apitch_v'], label='Pitch Vel', color='#00ff00', linestyle = '--')
		ax_rsp.plot(its, self.eps[ep_ind]['dyaw_v'], label='Target Yaw Vel', color='#aaaaff')
		ax_rsp.plot(its, self.eps[ep_ind]['ayaw_v'], label='Yaw Vel', color='#0000ff', linestyle = '--')
		ax_rsp.set_title('Response')
		ax_rsp.set_xlabel('Step')
		ax_rsp.set_ylabel('Velocity (rad/s)')
		ax_rsp.legend()

		# Errors Plot
		ax_err.plot(its, self.eps[ep_ind]['roll_err'], label='Roll Error')
		ax_err.plot(its, self.eps[ep_ind]['pitch_err'], label='Pitch Error')
		ax_err.plot(its, self.eps[ep_ind]['yaw_err'], label='Yaw Error')
		ax_err.plot(its, np.zeros(len(self.eps[ep_ind]['roll_err'])), color='#000000')
		ax_err.set_title('State Error')
		ax_err.set_xlabel('Step')
		ax_err.set_ylabel('Error')
		ax_err.legend()
		plt.show()


	def plotEps(self):
	# Plot the response and error plots for an indicated episode in the group of
	# examined episode tests.

		# plot
		fig, (ax_rwd, ax_fin) = plt.subplots(2,1)
		fig.suptitle('Gym_FC')

		# Ep Avera Plot
		ax_rwd.plot(list(range(len(self.eps))), self.eps_avg_r, label='Average Reward')
#		ax_rwd.set_title('Average Reward')
		ax_rwd.set_xlabel('Episode')
		ax_rwd.set_ylabel('Average Reward')
		ax_rwd.legend()

		# Ep Final Reward Plot
		ax_fin.plot(list(range(len(self.eps))), self.eps_fin_r, label='Final Reward')
#		ax_rwd.set_title('Final Reward')
		ax_fin.set_xlabel('Episode')
		ax_fin.set_ylabel('Reward')
		ax_fin.legend()

		# Ep Reward Difference (Beginning - End) Plot
#		ax_dif.plot(list(range(len(eps))), self.epd_del_r, label='Reward Progress')
#		ax_dif.set_title('Episode Reward Difference')
#		ax_dif.set_xlabel('Episode')
#		ax_dif.set_ylabel('Reward Difference')
#		ax_dif.legend()

		plt.show()

	def save_to_database(self, eps, db_path):
	# Save model evaluation data into database for future use
		conn = sqlite3.connect(db_path)  # Connection to database
		c = conn.cursor()  # Create cursor for executing commands
		
		# Create Tables
		#c.execute('''CREATE TABLE'''


def save_to_database(eps, db_path):
# Save model evaluation data into database for future use
	print('saving model eval database to {}.',format(db_path))

	conn = sqlite3.connect(db_path)  # Connection to database
	c = conn.cursor()  # Create cursor for executing commands
	

	# Create Tables
	sql_cmd = ('''CREATE TABLE ep (
	ep_number INTEGER PRIMARY KEY,
mode	model_name ,
	env_id ,
	eps_avg_r ,
	eps_fin_r ,
	eps_del_r ,
trainl	ep_avg_len ,
	rew_avg ,
	ep_this_iter ,
	episodes ,
	timesteps ,
	time_secs ,
	ev_tdlam_before ,
	ent_loss ,
	kl_loss ,
	entpen_pol_loss ,
	surr_pol_loss ,
	vf_loss ,
meta
	'''


def debugInitialize():
	tg.setVars()

	# Gym Environment
	env = ('AttFC_GyroErr-MotorVel_M4_Ep-v0', 'AttFC_GyroErr-MotorVel_M4_Con-v0',
		'AttFC_GyroErr1_M4_Ep-v0 - AttFC_GyroErr10_M4_Ep-v0')
	model_name = 'TESTYModel'
	eps = ModelEval(model_name, env[0])
	eps.evalModel(2)
	return eps


if __name__ == '__main__':
	print('begin')
	
