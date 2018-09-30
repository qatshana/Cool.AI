import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path
import yaml



#from gym.spaces import Box, Tuple
#from gym.envs.classic_control import rendering ===> causing issues on AWS disable for now [AQ]
import random
import pickle
import os

cfg_path = 'config/config.yml'

#Figure out which directory you've called this env from:
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
        
        self.max_in_change=cfg['max_in_change']  # for 6 setting this to .1 will achieve .5 cost (high)
        self.dt=cfg['rate_change']
        self.viewer = None 
        
        self.max_PUE=cfg['max_PUE']
        self.min_PUE=cfg['min_PUE']
        self.max_TZ=cfg['max_TZ']
        self.min_TZ=cfg['min_TZ']

        self.min_Tin=cfg['min_Tin']
        self.max_Tin=cfg['max_Tin']

        self.min_ITU_scaled=cfg['min_ITU_scaled']
        self.max_ITU_scaled=cfg['max_ITU_scaled']

        self.label_path = cfg['labels_path']
        self.pick_path = cfg['pickle_path_result']     
        #Variables inherent to the Fluent data: 
        self.num_ins = no_inputs

        self.scale_var = cfg['scale_var']
        # User set values are below. These can be adjusted in config.yml  
        self.MSE_thresh1 = (cfg['thresh1']*self.scale_var)**2   # set at (4% *10)^2=.16
        self.MSE_thresh2 = (cfg['thresh2']*self.scale_var)**2   #same as above
        self.MSE_thresh3 = (cfg['thresh3']*self.scale_var)**2   #same as above
        
        self.rew_goal = cfg['reward'] * self.scale_var  #100*10=1000
        # Get the function of input-output mapping, and max & min:
        [self.PUE, mins, maxes] = self.get_funcs('PUE')
        [self.TZ1, mins,maxes] = self.get_funcs('TZ1')
        [self.TZ2,  mins,maxes] = self.get_funcs('TZ2')

        # Observation space has 3 paramters         
        high = np.array([self.max_TZ, self.max_TZ, self.max_PUE])
        low = np.array([self.min_TZ, self.min_TZ, self.min_PUE])
        self.observation_space = spaces.Box(low=-high, high=high)
        
        self.action_range = cfg['action_range']
        #Action space is the up & down range for the 5 actions 
        self.action_space = spaces.Box(-self.action_range, self.action_range, shape=(self.num_ins,), dtype=np.float32)  # set to +/- 5
        self.seed()

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

    def temp_func2(self,var):
        """
        This function is the observer in the RL model.
        The coef, powers, and intercept are used to create a function of the outputs given the inputs.
        There is an option to add noise, to approximate thermal noise or other fluctuations in the environment.
        """
        y = var['intercept']
        for p,c in zip(var['powers'],var['coef']):
            # Exp the inputs to the power for that coef
            #to plug them into the equation, un-scale them:
            a = self.ins**p
            y += c* np.prod(a)

        #to fit this into the environment, re-scale:
        y = y #* self.scale_var
        return y

    def temp_func(self,var):
        """
        This function is the observer in the RL model.
        The coef, powers, and intercept are used to create a function of the outputs given the inputs.
        There is an option to add noise, to approximate thermal noise or other fluctuations in the environment.
        """
        y = var['intercept']
        for p,c in zip(var['powers'],var['coef']):
            # Exp the inputs to the power for that coef
            #to plug them into the equation, un-scale them:
            a = self.ins**p
            y += c* np.prod(a)

        #to fit this into the environment, re-scale:
        y = y #* self.scale_var
        return y

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def step(self, u):
        TOUTZ1, TOUTZ2, PUE = self.state # th := theta
        self.steps += 1    
        dt = self.dt
        TOUT1_target=self.TOUT1_target
        u = np.clip(u, -self.max_in_change, self.max_in_change)
        self.last_u = u # for rendering      

        #calculate the output
        #TOUTZ1 = TOUTZ1 + u[0]*dt+3*dt*u[1]-4*dt*u[2]+5*dt*u[3]
        #PUE= PUE-u[0]*dt+1.2*dt*u[1]-2*dt*u[2]+1*dt*u[3]
                
        # Increase or decrease the 5 input values
        self.ins = self.ins+ u
        self.ins = np.clip(self.ins,self.min_Tin, self.max_Tin)

        TOUTZ1 = self.temp_func(var=self.TZ1)
        PUE = self.temp_func(var=self.PUE)

        # get MSE
        MSE1 = (self.TOUT1_target-TOUTZ1)**2

        # get cost function
        costs = (self.TOUT1_target-TOUTZ1)**2 + .001*(u[0]**2)
        TOUTZ1 = np.clip(TOUTZ1,self.min_TZ, self.max_TZ)       
                
        MSE1_scaled = MSE1/(self.TOUT1_target)**2 # scale by target temp
            
        #done = ((MSE1_scaled <= self.MSE_thresh1) & (MSE2_scaled <= self.MSE_thresh2) & (MSE3_scaled <= self.MSE_thresh3))
       
        done = ((MSE1_scaled <= self.MSE_thresh1) )
        done = bool(done)

        TOUTZ2 = 0        
        

        self.state = np.array([TOUTZ1, TOUTZ2, PUE])
        if cfg['print_output']==True:
            #print(self.ins)
            print ("  ",TOUTZ1,self.TOUT1_target,done,self.steps)
        return self._get_obs(), -costs, done, {}

    def reset(self):

        self.steps = 0
        self.ins = random.uniform(self.min_Tin*np.ones([1,no_inputs]),self.max_Tin*np.ones([1,no_inputs]))
        low = np.array([self.min_TZ, self.min_TZ,1])
        high = np.array([self.max_TZ, self.max_TZ,2])
        if cfg['random_output']==True:
            self.TOUT1_target = self.np_random.uniform(-np.pi/2,np.pi/2)
        else:
            self.TOUT1_target= (cfg['TOUT1_target'])
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        TOUTZ1, TOUTZ2,PUE = self.state
        return np.array([TOUTZ1, TOUTZ2, PUE])

    def render(self, mode='human'):
    

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer: self.viewer.close()

