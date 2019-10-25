import os, csv
import numpy as np
from mpi4py import MPI
from baselines.common import tf_util as U
from baselines import logger
from shutil import copyfile
import gymfc
import gym


class trainParams():
	def __init__(self):
		self.num_timesteps = 4000  # maximum number of timesteps to run learning
		self.timesteps_per_actorbatch = 1000  # Horizon(T)
		self.clip_param = 0.1  # epsilon for clipping
		self.entcoeff = 0.0  # entropy coefficient for exploration
		self.optim_epochs = 10 # How many times to run optimization step
		self.optim_stepsize = 1e-4 # Adam stepsize
		self.optim_batchsize = 64 # How many samples to use for optimization
		self.gamma = 0.99 # Discount factor of reward per future timestep
		self.lam = 0.95  # Advantage estimator parameter (lambda)
		self.schedule='linear'  # Change action randomness through 
		self.seed = 0  # Which random seed to use
		self.model_path = None # Where to save model

	def modelName(self, name):
		self.model_name=name
		self.model_dir=os.environ['GYMFC_EXP_MODELSDIR']+name
		self.model_path=self.model_dir+'/'+name

	def modelDir(self):
		model_name = self.modelName()
		return self.model_path[:-len(model_name)]


class policyParams():
	def __init__(self):
		self.nodes_per_layer = 32
		self.num_layers = 2


class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale
    def reward(self, r):
        return r * self.scale


def train(train_params, policy_params, env_id):
	# Refresh training progress log	
	logger._configure_default_logger()

	from baselines.ppo1 import mlp_policy, pposgd_simple
	U.make_session(num_cpu=1).__enter__()
	def policy_fn(name, ob_space, ac_space):
		return mlp_policy.MlpPolicy(
			name=name, ob_space=ob_space, ac_space=ac_space, # set tensor
			hid_size=policy_params.nodes_per_layer, # set nodes
			num_hid_layers=policy_params.num_layers) # set layers


	# Set up environment
	env = gym.make(env_id)
	env = RewScale(env, 0.1)
	# Seed Set
	rank = MPI.COMM_WORLD.Get_rank()
	workerseed = train_params.seed + 1000000 * rank
	env.seed(workerseed)
	print('----------=================--------------')
	print('rank: ', rank, 'workerseed: ', workerseed)
	print('----------=================--------------')

	# Run Training with stochastic gradient descent
	pi = pposgd_simple.learn(env, policy_fn,
                max_timesteps=train_params.num_timesteps,
                timesteps_per_actorbatch=train_params.timesteps_per_actorbatch,
                clip_param=train_params.clip_param, 
		entcoeff=train_params.entcoeff,
                optim_epochs=train_params.optim_epochs,
                optim_stepsize=train_params.optim_stepsize,
                optim_batchsize=train_params.optim_batchsize,
                gamma=train_params.gamma,
                lam=train_params.lam,
                schedule=train_params.schedule,
	)
	env.close

	# Save Trained Model an meta data
	print(train_params.model_path)
	if train_params.model_path:
		# Make Dir
		model_log_dir=os.environ['GYMFC_EXP_MODELSDIR']+train_params.model_name
		os.makedirs(train_params.model_dir, exist_ok=True)

		# Merge Metadata
		meta_data = {**vars(train_params), **vars(policy_params)}

		# Save Metadata as csv
		md_file = train_params.model_dir+'/'+'metadata.csv'
		md_keys = meta_data.keys()
		try:
			with open(md_file, 'w') as mdfile:
				writer = csv.DictWriter(mdfile, fieldnames = md_keys)
				writer.writeheader()
				writer.writerow(meta_data)
		except IOError:
			print("I/O error")

		# Save Model
		U.save_variables(train_params.model_path)

		# Save Training Progress file
		log_path = os.environ['OPENAI_LOGDIR']+'progress.csv'  # train prog log 
		copyfile(log_path, train_params.model_dir+'/'+'log.csv') # copy log csv
		
	else:
		print('model not named')
	return pi


def setVars():
	# Set up env Variables
	os.environ['GYMFC_CONFIG'] = '/home/acit/gymfc/examples/configs/iris.config'
	os.environ['OPENAI_LOGDIR'] = '/home/acit/gymfc/rlgymdronecontrol/gymlogs/'
	os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv'
	os.environ['GYMFC_EXP_MODELSDIR'] = '/home/acit/gymfc/rlgymdronecontrol/gymlogs/models/'


def runExp():
	setVars()

	# Set up training Paramters	
	train_params = trainParams()
	train_params.num_timesteps = 4000
	train_params.timesteps_per_actorbatch = 2000
	train_params.optim_batchsize = 20
	train_params.optim_epochs = 2
	train_params.modelName('noesc01')

	# Set up policy Parameters
	policy_params = policyParams()

	# Gym environment
	env = ('AttFC_GyroErr-MotorVel_M4_Ep-v0', 'AttFC_GyroErr-MotorVel_M4_Con-v0',
		'AttFC_GyroErr1_M4_Ep-v0')
	env_id = env[2]

	# Run Training
	with U.tf.Graph().as_default():
		train(train_params, policy_params, env_id)

# Automatically set environment variables
print ('Setting Environment Variables')
setVars()

#if __name__ == '__main__':


