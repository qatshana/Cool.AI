import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import yaml
import pandas as pd
import random
import pickle
import os

cfg_path = 'config/config.yml'

try:
    with open(cfg_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)   
except Exception as e:
    print('error: %s'%(e))
    
no_inputs=cfg['no_inputs']

class AllVar(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):       
        self.max_in_change=cfg['max_in_change']  
        self.viewer = None         
        self.max_PUE = cfg['max_PUE']
        self.min_PUE = cfg['min_PUE']
        self.max_TZ = cfg['max_TZ']
        self.min_TZ = cfg['min_TZ']
        self.min_Tin = cfg['min_Tin']
        self.max_Tin = cfg['max_Tin']
        self.min_ITU_scaled = cfg['min_ITU_scaled']
        self.max_ITU_scaled = cfg['max_ITU_scaled']
        self.label_path = cfg['labels_path']
        self.pick_path = cfg['pickle_path_result']     
        self.num_ins = no_inputs  # defines number of inputs
        self.scale_var = cfg['scale_var']        
        self.MSE_thresh1 = (cfg['thresh1']*self.scale_var)**2   # defines threshold for MSE

        # Get the function of input-output mapping, and max & min:
        [self.PUE, mins, maxes] = self.get_funcs('PUE')
        [self.TZ1, mins,maxes] = self.get_funcs('TZ1')
        [self.TZ2,  mins,maxes] = self.get_funcs('TZ2')

        # Observation space        
        high = np.array([self.max_TZ, self.max_TZ, self.max_PUE])
        low = np.array([self.min_TZ, self.min_TZ, self.min_PUE])
        self.observation_space = spaces.Box(low=-high, high=high)
        # action space
        self.action_range = cfg['action_range'] 
        self.action_space = spaces.Box(-self.action_range, self.action_range, shape=(self.num_ins,), dtype=np.float32) 

        # initialize logging paramteres
        self.episodes = 0
        self.data = ''
        self.seed()

    def get_funcs(self,var):
        """
        This function loads the pickles with the function approximating the Fluent simulation data.
        """
        fname = (var+".p")         
        pickle_path = os.path.join(self.pick_path,fname)
        [coef,powers,intercept,mins,maxes] = pickle.load(open(pickle_path,'rb'))
        # The function variables you need to-recreate this model & the min & max to set this in the environment.
        out = {'coef': coef, 'powers':powers,'intercept':intercept}
        return out, mins, maxes


    def temp_func(self,var):
        """
        This function is the observer in the RL model.
        The coef, powers, and intercept are used to create a function of the outputs given the inputs.
        There is an option to add noise, to approximate thermal noise or other fluctuations in the environment.
        """
        y = var['intercept']
        for p,c in zip(var['powers'],var['coef']):
            # Exp the inputs to the power for that coef
            a = self.ins**p
            y += c* np.prod(a)
        return y

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, u):
        TOUTZ1, TOUTZ2, PUE = self.state 
        self.steps += 1    
        u = np.clip(u, -self.max_in_change, self.max_in_change)
        self.last_u = u # for rendering      
                
        # Increase or decrease the 5 input values
        self.ins = self.ins + u
        self.ins = np.clip(self.ins,self.min_Tin, self.max_Tin)

        #calculate the output
        TOUTZ1 = self.temp_func(var=self.TZ1)
        PUE = self.temp_func(var=self.PUE)
        TOUTZ2 = 0 # not used, only one zone for now

        # get MSE
        MSE1 = (self.TOUT1_target-TOUTZ1)**2

        # get cost function: sum of MSE's
        costs = MSE1
        TOUTZ1 = np.clip(TOUTZ1,self.min_TZ, self.max_TZ)       
        
        if self.TOUT1_target>=self.min_TZ:        
            MSE1_scaled = MSE1/(self.TOUT1_target)**2 # scale by target temp  
        else:
            MSE1_scaled = MSE1 # do no apply scaling
                                 
        done = ((MSE1_scaled <= self.MSE_thresh1) )
        done = bool(done)
                     
        self.state = np.array([TOUTZ1, TOUTZ2, PUE])
        if cfg['log_output'] == True:
            if cfg['print_output_single'] == True:
                print ("  ",TOUTZ1,self.TOUT1_target)
            self.TOUT_data.append(TOUTZ1)
            self.Step_data.append(self.steps)
        if done == True:
            txt = str(self.TOUT_data)+","
            self.data+=txt
            if cfg['print_output_all'] == True:
                print(self.data)
            if cfg['save_output'] == True:
                output_file = cfg['output_file']
                with open(output_file,'a') as file:
                    file.write(self.data) 
                            
        return self._get_obs(), -costs, done, {}

    def reset(self):
        # reset logging parameters
        self.TOUT_data = []
        self.Step_data = []
        self.episodes += 1
        self.steps = 0

        # reset input and output paramaters
        self.ins = random.uniform(self.min_Tin*np.ones([1,no_inputs]),self.max_Tin*np.ones([1,no_inputs]))
        low = np.array([self.min_TZ, self.min_TZ,1])
        high = np.array([self.max_TZ, self.max_TZ,2])
        if cfg['random_output'] == True:
            self.TOUT1_target = self.np_random.uniform(self.min_TZ,self.max_TZ)
        else:
            self.TOUT1_target = (cfg['TOUT1_target'])
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        TOUTZ1, TOUTZ2,PUE = self.state
        return np.array([TOUTZ1, TOUTZ2, PUE])

    def render(self, mode='human'):    
        pass

    def close(self):
        if self.viewer: self.viewer.close()

