import os
import numpy as np
from mpi4py import MPI
from baselines.common import tf_util as U
from baselines import logger
import gymfc
import gym


class trainParams():
	def __init__(self):
		self.num_timesteps = 4000  # maximum number of timesteps to run learning
		self.timesteps_per_actorbatch = 1000  # Horizon(T)
		self.clip_param = 0.1  # ??? epsilon for clipping
		self.entcoeff = 0.0  # entropy coefficient for exploration
		self.optim_epochs = 10 # How many times to run optimization step
		self.optim_stepsize = 1e-4 # ??? Adam stepsize, alpha/learning rate?
		self.optim_batchsize = 64 # How many samples to use for optimization
		self.gamma = 0.99 # Discount factor of reward per future timestep
		self.lam = 0.95  # Advantage estimator parameter (lambda)
		self.schedule='linear'  # Change action randomness through 
		self.seed = 0  # Which random seed to use
		self.model_path = None  # Where to save model

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

	# Save Trained Model
	print(train_params.model_path)
	if train_params.model_path:
		U.save_variables(train_params.model_path)
	else:
		print('No save path')
	return pi


def runExp():
	# Set up env Variables
	os.environ['OPENAI_LOGDIR'] = '/home/acit/gymlogs/'
	os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv'

	# Set up training Paramters	
	train_params = trainParams()
	train_params.num_timesteps = 10000000
	train_params.timesteps_per_actorbatch = 2000
	train_params.optim_batchsize = 100
	train_params.model_path = '/home/acit/gymlogs/models/testModel'

	# Set up policy Parameters
	policy_params = policyParams()

	# Gym environment
	env_id = 'AttFC_GyroErr-MotorVel_M4_Ep-v0'

	# Run Training
	with U.tf.Graph().as_default():
		train(train_params, policy_params, env_id)


#if __name__ == '__main__':


