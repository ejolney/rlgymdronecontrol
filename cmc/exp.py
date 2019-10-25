import subprocess
import itertools
import numpy as np


def exp():
    steps = np.linspace(50000, 50001, num=1, dtype=int)
    nodes = np.linspace(32, 33, num=1, dtype=int)
    layers = np.array([2],dtype=int)
    actorbatch = np.ones(4,dtype=int)*325 #np.linspace(325, 326, num=10, dtype=int)
    exp_list = list(itertools.product(*[steps,nodes,layers,actorbatch]))
    
    return exp_list


def run_exp():
    exp_list = exp()
    cmctrain = '/cmc_train.py' #Set to cmc_train path ex: '/home/user/CMC/cmc_train.py'
    for params in exp_list:
        #print(i)
        steps = str(params[0])
        nodes = str(params[1])
        layers = str(params[2])
        actorbatch = str(params[3])
        subprocess.call(['python3', cmctrain, '--steps',steps, '--nodes',nodes, '--layers',layers, '--actorbatch',actorbatch])


if __name__ == '__main__':
    run_exp()
    #exp()
