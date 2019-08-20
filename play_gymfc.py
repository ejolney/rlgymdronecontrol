import numpy as np
from baselines.common import tf_util as U
from baselines import logger
import gymfc
import gym


def train(num_timesteps, seed, model_path=None, env_id=None, params=None):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space, params=params):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=int(params[1]), num_hid_layers=int(params[2]))


env = gym.make('AttFC_GyroErr-MotorVel_M4_Ep-v0')
#env.render()

for _ in range(1000):
	env.step(env.action_space.sample())

env.close()

