import os, csv, pickle, json
import numpy as np
import pandas as pd
from baselines.common import tf_util as U
from mpi4py import MPI
import gymfc
import gym
import sqlite3
import matplotlib.pyplot as plt
import train_gymfc as tg


class ModelEval:
########################################## Info ################################################
# Model evaluation object for running trained model on environment and recording/saving outputs
################################################################################################
	def __init__(self, model_name, env_id):
		self.model_name = model_name
		self.env_id = env_id
		print('Evaluating model {}'.format(model_name))


	def evalModel(self, toteps=1, err_thresh=.1):

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

		# Set up parameters for quick initialization
		tp = tg.trainParams()
		tp.num_timesteps = 1
		tp.timesteps_per_actorbatch = 1000
		tp.optim_epochs = 1
		tp.optim_batchsize = 1
		tp.seed = 17
		
		pp = tg.policyParams()

		# Initialize and load model
		if tp.model_path: tp.model_path = None # prevent override of model
		with U.tf.Graph().as_default():		# Allow Re-running of tf
			pi = tg.train(tp, pp, self.env_id)

		# Load Model
		tp.modelName(self.model_name) # Set up name
		self.model_dir = tp.model_dir # Save extracted model dir
		self.model_path = tp.model_path # Save extracted model path
		U.load_variables(tp.model_path)

		# Make Training Log
	#	self.train_log = TrainLog(tp.model_dir)
		
		# Setup gym
		env = gym.make(self.env_id)
		# Seed Set
		rank = MPI.COMM_WORLD.Get_rank()
		workerseed = tp.seed + 1000000 * rank
		env.seed(workerseed)

		env.reset()
		ob = env.reset()  # reset object for pi

		print('----------=================--------------')
		print('rank: ', rank, 'workerseed: ', workerseed)
		print('----------=================--------------')

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
#				action = env.action_space.sample() # Random action for ctrl
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


class TrainLog:
########################################## Info ################################################
# Training log object that reads and stores training log with data as available
# ex. 	>log_file = 'path/to/csv/log_file.csv'
#	>tlog = TrainLog(log_file)
# tlog is now an inclusion of all of the available data from that training session
# ex.	>tlog.episodes  # Episodes so far list
#	array([ 1, 2, 3, ...])
################################################################################################


	def __init__(self, model_dir=''):
		if model_dir:
			log_file = model_dir+'/log.csv'
			metadata_file = model_dir+'/metadata.csv'

			# Read log file
			if os.path.isfile(log_file):
				log = pd.read_csv(log_file, encoding='utf-8', error_bad_lines=False)
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
		ax.set_xlabel('Episode')
		ax.set_ylabel('Avg_reward Per Episode')
		plt.show()	


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


def debugInitialize():
	tg.setVars()

	# Gym Environment
	env = ('AttFC_GyroErr-MotorVel_M4_Ep-v0', 'AttFC_GyroErr-MotorVel_M4_Con-v0',
		'AttFC_GyroErr1_M4_Ep-v0')
	model_name = 'TESTYModel'
	eps = ModelEval(model_name, env[0])
	eps.evalModel(2)
	return eps


if __name__ == '__main__':
	print('begin')
	
