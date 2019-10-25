import math
import numpy as np
from .gazebo_env import GazeboEnv
import logging
logger = logging.getLogger("gymfc")


class AttitudeFlightControlEnv(GazeboEnv):
    def __init__(self, **kwargs): 
        self.max_sim_time = kwargs["max_sim_time"]
        self.speedFlag = 1
        super(AttitudeFlightControlEnv, self).__init__()

    def compute_reward(self):
        """ Compute the reward """
        thr = .0628 # .0628-99.5% | .1257-99% | .6283-95%
        c = [0,0,0,1]
#        r_r = -np.clip(np.sum(np.abs(self.error))/(self.omega_bounds[1]*3), 0, 1) # err
        r_r = -np.sum(np.square(self.error)) # err
        c_r = c[0]# error Coefficient
        r_e = -1*np.sum(np.abs(self.act)) # energy
        c_e = c[1] # energy Coefficient
        r_s = -1 * self.speedFlag # speed
        c_s = c[2]# speed Coefficient
        if np.all(np.abs(self.error)<thr): 
            self.speedFlag = 0
            r_g = 1 #- np.sum(np.abs(self.act_act-self.act)) # goal - saturation
#            if np.sum(np.abs(self.act_act-self.act))<.1: r_g = 10  # goal - saturation penalty
#            else: r_g = 1
            r_s = 0
#            r_g=0
        else: r_g=0
        c_g = c[3] # goal Coefficient

        return np.array(r_r*c_r+r_e*c_e+r_s*c_s+r_g*c_g) #Reward Formula
#        return np.array(r_g*c_g)

    def sample_target(self):
        """ Sample a random angular velocity """

        return self.np_random.uniform(self.omega_bounds[0], self.omega_bounds[1], size=3)
#        return np.array([ 0,0,5])

class GyroErrorFeedbackEnv(AttitudeFlightControlEnv):
    def __init__(self, **kwargs): 
        self.observation_history = []
        self.action_history = []
        self.memory_size = kwargs["memory_size"]
        super(GyroErrorFeedbackEnv, self).__init__(**kwargs)
        self.omega_target = self.sample_target()

    def step(self, action):
        self.act_act = action #Save current actual action
        action = np.clip(action, self.action_space.low, self.action_space.high) 
        # Remove saturated actions
#        action[action>self.action_space.high]=0
#        action[action<self.action_space.low]=0
        # Step the sim
        self.act = action #Save current clipped action
        self.action_history.append(self.act)
        self.obs = self.step_sim(action)
        self.error = self.omega_target - self.obs.angular_velocity_rpy
        self.observation_history.append(np.concatenate([self.error]))
        state = self.state()
        done = self.sim_time >= self.max_sim_time
        reward = self.compute_reward()
        info = {"sim_time": self.sim_time, "sp": self.omega_target, "current_rpy": self.omega_actual}
        return state, reward, done, info

    def state(self):
        """ Get the current state """
        # The newest will be at the end of the array
        memory = np.array(self.observation_history[-self.memory_size:])
        return np.pad(memory.ravel(), 
                      ( (3 * self.memory_size) - memory.size, 0), 
                      'constant', constant_values=(0)) 

    def reset(self):
        self.observation_history = []
        self.speedFlag = 1
        return super(GyroErrorFeedbackEnv, self).reset()

class GyroErrorESCVelocityFeedbackEnv(AttitudeFlightControlEnv):
    def __init__(self, **kwargs): 
        self.observation_history = []
        self.memory_size = kwargs["memory_size"]
        super(GyroErrorESCVelocityFeedbackEnv, self).__init__(**kwargs)
        self.omega_target = self.sample_target()

    def step(self, action):
        self.act_act = action
        action = np.clip(action, self.action_space.low, self.action_space.high) 
        self.act = action #Save current action
        # Step the sim
        self.obs = self.step_sim(action)
        self.error = self.omega_target - self.obs.angular_velocity_rpy
        self.observation_history.append(np.concatenate([self.error, self.obs.motor_velocity]))
        state = self.state()
        done = self.sim_time >= self.max_sim_time
        reward = self.compute_reward()
        info = {"sim_time": self.sim_time, "sp": self.omega_target, "current_rpy": self.omega_actual}

        return state, reward, done, info

    def state(self):
        """ Get the current state """
        # The newest will be at the end of the array
        memory = np.array(self.observation_history[-self.memory_size:])
        return np.pad(memory.ravel(), 
                      (( (3+self.motor_count) * self.memory_size) - memory.size, 0), 
                      'constant', constant_values=(0)) 

    def reset(self):
        self.observation_history = []
        return super(GyroErrorESCVelocityFeedbackEnv, self).reset()

class GyroErrorESCVelocityFeedbackContinuousEnv(GyroErrorESCVelocityFeedbackEnv):
    def __init__(self, **kwargs): 
        self.command_time_off = kwargs["command_time_off"]
        self.command_time_on = kwargs["command_time_on"]
        self.command_off_time = None
        super(GyroErrorESCVelocityFeedbackContinuousEnv, self).__init__(**kwargs)

    def step(self, action):
        """ Sample a random angular velocity """
        ret = super(GyroErrorESCVelocityFeedbackContinuousEnv, self).step(action) 

        # Update the target angular velocity 
        if not self.command_off_time:
            self.command_off_time = self.np_random.uniform(*self.command_time_on)
        elif self.sim_time >= self.command_off_time: # Issue new command
            # Commands are executed as pulses, always returning to center
            if (self.omega_target == np.zeros(3)).all():
                self.omega_target = self.sample_target() 
                self.command_off_time = self.sim_time  + self.np_random.uniform(*self.command_time_on)
            else:
                self.omega_target = np.zeros(3)
                self.command_off_time = self.sim_time  + self.np_random.uniform(*self.command_time_off) 

        return ret 


