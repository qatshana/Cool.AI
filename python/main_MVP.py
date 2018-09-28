"""
Created on Spet 24, 2018
Code creates an RL agent that determines required input parameters for desired outputs
for HVAC system control

.
The main functions are:
    1) Polynomial regression to map inputs to outputs from simulated data
    2) Reinforcement learning algorithm to derive inputs from target outputs
    3) Custom Open Gym Env 

@author: Alex Qatshan)
"""

import sys
import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential
from gym import wrappers
from keras.models import Model

import os
import yaml

EPISODES = 10

# Simple Agent for MSP
class MSPAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # create replay memory using deque
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0

        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

    # get action from model using epsilon-greedy policy
    def get_action(self,env, state):    
        alpha=.1 # Convergence rate=> 1 code will converge in one step 
        action=[alpha*(env.goals[0]-state[0,0]),0,0,alpha*(env.goals[1]-state[0,1]),0,0,0,0,0]
        return action
        
if __name__ == "__main__":    
    env = gym.make('AllVar-v0')    
    state_size=3
    action_size=9
    agent = MSPAgent(state_size, action_size)
    scores, episodes = [], []
    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        while not done:
            if agent.render:
                env.render()
            # get action for the current state and go one step in environment
            # action returned is mainly adjustment to input
            action = agent.get_action(env,state)   
            next_state, reward, done, info = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            # if an action make the episode end, then gives penalty of -100
            reward = reward if not done or score == 499 else -100            
            score += reward
            state = next_state
            if done:
                score = score if score == 500 else score + 100
                scores.append(score)
                episodes.append(e)                                
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon)

                # if the mean of scores of last 10 episode is bigger than 490
                # stop training
                if np.mean(scores[-min(10, len(scores)):]) > 490:
                    sys.exit()

