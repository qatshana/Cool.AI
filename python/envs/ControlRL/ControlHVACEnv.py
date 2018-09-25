"""This is a custom OpenAI environment for rocket engine tuning"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import gym
from gym.spaces import Box, Tuple
from gym.envs.classic_control import rendering
import random

import pickle
import os
import yaml

# Set config path. TO DO: import cfg from main script (from where RLlib calls this env)
CWD_PATH = os.getcwd()
cfg_path = 'config/config.yml'

#Figure out which directory you've called this env from:
try:
    with open(cfg_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)   
except Exception as e:
    print('error: %s'%(e))
    
no_inputs=cfg['no_inputs']

class AllVar(gym.Env):
    """
    This class contains all of the functions for the custom Rocket Engine tuning environment. 
    """
    #For rendering the rollout:
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50}
    
    def __init__(self,config = None):
        """
        Variables can be set in config.yml
        """
        
        #self.join_path = join_path
        self.label_path = cfg['labels_path']
        self.pick_path = (cfg['result_path'] + cfg['pickle_path'])
        #self.label_dir = os.path.join(CWD_PATH,self.join_path, self.label_path)

        #Variables inherent to the Fluent data: 
        self.num_ins = no_inputs

        self.scale_var = cfg['scale_var']
        # User set values are below. These can be adjusted in config.yml  
        self.MSE_thresh1 = (cfg['thresh1']*self.scale_var)**2   # set at (4% *10)^2=.16
        self.MSE_thresh2 = (cfg['thresh2']*self.scale_var)**2   #same as above
        self.MSE_thresh3 = (cfg['thresh3']*self.scale_var)**2   #same as above
        
        self.rew_goal = cfg['reward'] * self.scale_var  #100*10=1000

        self.noise = cfg['noise']
        self.minmaxbuffer = cfg['minmaxbuffer']   #===> set to 0

        # Get the function of input-output mapping, and max & min:
        [self.O_CH4_flow_uniformity, mins,maxes] = self.get_funcs('O_CH4_flow_uniformity')
        [self.PUE, mins, maxes] = self.get_funcs('PUE')
        [self.TZ1, mins,maxes] = self.get_funcs('TZ1')
        [self.TZ2,  mins,maxes] = self.get_funcs('TZ2')
        
        
        self.mins = mins# * self.scale_var
        self.maxes = maxes#* self.scale_var
        #Action range is a percentage of the total range
        self.action_range = cfg['action_range']*self.scale_var   #===>.5*10=5

        #Action space is the up & down range for the 4 actions 
        self.action_space = Box(-self.action_range, self.action_range, shape=(self.num_ins,), dtype=np.float32)  # set to +/- 5

        # Observation space has 3 paramters         
        self.observation_space = Tuple((Box(self.mins.values[0],self.maxes.values[0],shape=(1,), dtype=np.float32),
                                        Box(self.mins.values[1],self.maxes.values[1],shape=(1,), dtype=np.float32),
                                        Box(self.mins.values[2],self.maxes.values[2],shape=(1,), dtype=np.float32)))        
        # [AQ] review this code
        self._spec = lambda: None
        self._spec.id = "AllVar-v0"
        
        # For rendering:
        self.viewer = None
        self.labels = cfg['labels']
        
        #initialize variables for tracking:
        self.episode = 0
        self.reward = 0
        self.reset()

    def get_funcs(self,var):
        """
        This function loads the pickles with the function approximating the Fluent simulation data.
        """
        fname = (var+".p")         
        pickle_path = os.path.join(self.pick_path,fname)
        [coef,powers,intercept,mins,maxes] = pickle.load(open(pickle_path,'rb'))
        
        # The 3 function variables you need to-recreate this model & the min & max to set this in the environment.
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
            # Exp the 4 inputs to the power for that coef
            #to plug them into the equation, un-scale them:
#            a = np.multiply(self.ins,(1/self.scale_var))**p
            a = self.ins**p
            y += c* np.prod(a)

        #to fit this into the environment, re-scale:
        y = y #* self.scale_var

        return y


    def test_viable(self,outs):
        """
        Because the regression model doensn't adhere to the bounds of the inputs, some of these outputs
        might be outside the range an engineer would encounter. This prevents such values from being set
        as targets, since that would be unrealistic.
        """        
        viable = True
        for i,temp_i in enumerate(outs):
            if (temp_i <= self.mins[i+no_inputs]):
                viable = False
            elif (temp_i >= self.maxes[i+no_inputs]): 
                viable = False
        return viable
    
    def reset(self): 
        """ 
        This is the function to reset for every new episode. The starting position carries over from
        the previous episode. The goal temperature changes on every episode.
        """
        
        self.steps = 0
                
        self.ins = random.uniform(self.mins.values[:no_inputs],self.maxes.values[:no_inputs])
        #get the corresponding outputs: 
        out_TZ1 = self.temp_func(var=self.TZ1)
        out_TZ2 = self.temp_func(var=self.TZ2)
        out_PUE = self.temp_func(var=self.PUE)

        outs = np.array([out_TZ1,out_TZ2,out_PUE])
        self.state=outs
        ins_temp=self.ins # store initial inputs

        #get goals from random inputs:
        viable = False
        while viable == False:
            self.ins = random.uniform((self.mins.values[:no_inputs]+(self.mins.values[:no_inputs]*self.minmaxbuffer)),self.maxes.values[:no_inputs]-(self.maxes.values[:no_inputs]*self.minmaxbuffer))
            out_TZ1 = self.temp_func(var=self.TZ1)
            out_TZ2 = self.temp_func(var=self.TZ2)
            out_PUE = self.temp_func(var=self.PUE)

            outs = np.array([out_TZ1,out_TZ2,out_PUE])
            
            # Check if viable:
            viable = self.test_viable(outs)

        self.goals = outs

        # These are your current inputs:
        self.ins = ins_temp
        
        #Track episodes and total reward.
        self.episode += 1
        self.reward=0
        self.tot_rew = 0

        return (self.state)

    def step(self, action):
        """
        This function determines the outcome of every action.
        First, the env checks whether the action is within the min & max range of the inputs.
        Second, the corresponding output variables are calculated. 
        Third, the MSE is calculated. 
        The agent is done if the MSE is within the range specied in cfg, and rewarded accordingly.
        Otherwise, the agent is penalized by the amount of the MSE. 
        
        """
        self.steps += 1
        in_var = self.ins

        # Increase or decrease the 4 input values
        self.ins = self.ins+ action 

        #If the agent tries to exceed the range of the mins & maxes, this sets them to the max. 
        
        '''
        
        for i,temp_i in enumerate(new_var):
            if (temp_i <= self.mins[i]):
                new_var[i] = self.mins[i]
            elif (temp_i >= self.maxes[i]): 
                new_var[i] = self.maxes[i]
        '''
             

        # Get all the new outputs:
        
        out_TZ1 = self.temp_func(var=self.TZ1)
        out_TZ2 = self.temp_func(var=self.TZ2)
        out_PUE = self.temp_func(var=self.PUE)

        #check that this is a viable output; if not, reject the action
        #is this temp change viable?
    
        MSE1 = (self.goals[0]-out_TZ1)**2
        MSE2 = (self.goals[1]-out_TZ2)**2
        MSE3 = (self.goals[2]-out_PUE)**2

        MSE = MSE1 +  MSE2 + MSE3

        MSE1_scaled = MSE1/(self.goals[0]**2)
        MSE2_scaled = MSE2/(self.goals[1]**2)
        MSE3_scaled = MSE3/(self.goals[2]**2)
        
       
        # Update your state:
        self.state =np.array([out_TZ1,out_TZ2,out_PUE])
       
        #done = ((MSE1_scaled <= self.MSE_thresh1) & (MSE2_scaled <= self.MSE_thresh2) & (MSE3_scaled <= self.MSE_thresh3))
        # limit to 2 constraints for now
        done = ((MSE1_scaled <= self.MSE_thresh1) & (MSE2_scaled <= self.MSE_thresh2) )

        done = bool(done)

        # Get the corresponding reward:
        #reward = 0
        if done:
            #self.reward += self.rew_goal
            self.reward = 100
            self.tot_rew += self.reward
        else:          
            #self.reward -= MSE *cfg['MSE_scale']
            self.reward = -10
            self.tot_rew += self.reward

        if self.tot_rew <= -1000:
                done = True 
                print ("Done")

        self.tot_rew += self.reward
        self.done = done
        return (self.state, self.reward, done, {'MSE_scaled': MSE1_scaled })

        
    def render(self, mode='human'):
        """
        This function renders the agent's actions.
        The top of the screen tracks the # of steps
 
        """
        pass

    def close(self):
        if self.viewer: self.viewer.close()
