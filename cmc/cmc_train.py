import os
from baselines.common import tf_util as U
from baselines import logger
import gymfc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gym
import math
import csv
import time
import argparse
#from mpi4py import MPI
#import argparse
#matplotlib.use('Agg')


def train(num_timesteps, seed, model_path=None, env_id=None, params=None):
    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    def policy_fn(name, ob_space, ac_space, params=params):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=int(params[1]), num_hid_layers=int(params[2]))
	
    print(env_id)
    env = gym.make(env_id)

    # parameters below were the best found in a simple random search
    # these are good enough to make humanoid walk, but whether those are
    # an absolute best or not is not certain
    env = RewScale(env, 0.1)
    if params:
        pi = pposgd_simple.learn(env, policy_fn,
                max_timesteps=num_timesteps,
                timesteps_per_actorbatch=int(params[3]),
                clip_param=0.2, entcoeff=0.0,
                optim_epochs=10,
                optim_stepsize=3e-4,
                optim_batchsize=64,
                gamma=0.99,
                lam=0.95,
                schedule='linear',
            )
    else:
        pi = pposgd_simple.learn(env, policy_fn,
                max_timesteps=num_timesteps,
                timesteps_per_actorbatch=200,
                clip_param=0.2, entcoeff=0.0,
                optim_epochs=10,
                optim_stepsize=3e-4,
                optim_batchsize=64,
                gamma=0.99,
                lam=0.95,
                schedule='linear',
        )
    env.close()
    if model_path:
        U.save_variables(model_path)

    return pi

class RewScale(gym.RewardWrapper):
    def __init__(self, env, scale):
        gym.RewardWrapper.__init__(self, env)
        self.scale = scale
    def reward(self, r):
        return r * self.scale


def read_exps(exp_config):
    with open(exp_config) as exp:
        exp_reader = csv.reader(exp, delimiter=',')
        exp_list = list(exp_reader)
    return exp_list


def train_pi(params):
    env = 'CMC-v0'
    seed = 0
    model_path = '' #set model path ex:'/home/user/CMC/cmc_models/cmcTest01'
    num_timesteps = int(params[0])
    plot_name = 'None'

    pi = train(num_timesteps=num_timesteps, seed=seed, model_path=model_path, env_id=env, params=params)
    return pi


def base_main(pi):
    env = 'CMC-v0'
    seed = 0
    model_path = '' #set model path ex:'/home/user/CMC/cmc_models/cmcTest01'
    plot_name = 'None'
    num_eps = 40

    #pi = train(num_timesteps=1, seed=seed, env_id=env)
    #U.load_variables(model_path)

    env = gym.make(env)
    ob = env.reset()

    # Episode Stats
    total_reward = 0
    num_steps = 0
    total_energy = 0

    # Run Stats
    eps_rewards = []
    eps_steps = []
    eps_energy = []
    eps_error = []
    print('start')
    for _ in range(num_eps):
        total_reward = 0
        num_steps = 0
        total_energy = 0
        total_error = 0
        while True:
            action = pi.act(stochastic=False, ob=ob)
            ob, r, done, _ = env.step(action)

            #env.render()
        
            # Update Stats
            total_reward = total_reward + r
            num_steps += 1
            total_energy = total_energy + abs(action[0])
            total_error = total_error + 0.45 - ob[0]
            if done:
                    env.reset()
                    #print(total_reward, num_steps, total_energy)
                    eps_rewards.append(total_reward)
                    eps_steps.append(num_steps)
                    eps_energy.append(total_energy)
                    eps_error.append(total_error)
                    break
    env.close()
    ep_stats=[np.mean(eps_rewards), np.mean(eps_steps), np.mean(eps_energy), np.mean(eps_error), np.std(eps_rewards), np.std(eps_steps), np.std(eps_energy), np.std(eps_error)]
    #print('Averages:')
    #print(np.mean(eps_rewards), np.mean(eps_steps), np.mean(eps_energy))
    #print('Std:')
    #print(np.std(eps_rewards), np.std(eps_steps), np.std(eps_energy))
    print('end')

    return list(ep_stats)


def run_exp():
    #Param = [num_timestep, nodes, layers, actorbatch]
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--nodes', type=int, default=32)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--actorbatch', type=int, default=100)
    args = parser.parse_args()

    params = list([args.steps, args.nodes, args.layers, args.actorbatch])
    
    
    #exp_config = '/home/user/CMC/exp01.csv'
    #exp_list = read_exps(exp_config)
    #params0 = list(exp_list[0])

    start_train = time.time()
    pi=train_pi(params)
    end_train = time.time()
    train_time = end_train-start_train

    ep_stats=base_main(pi)
    params.extend(ep_stats)
    params.append(train_time)
    print(params)

    rcsv = '' #Set output csv path ex: 'home/user/CMC/rewardFunc.csv'
    with open(rcsv, mode='a') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        data_writer.writerow(params)


if __name__ == '__main__':
    run_exp()
