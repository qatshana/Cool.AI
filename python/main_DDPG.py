import numpy as np
import gym

from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Input, Concatenate
from keras.optimizers import Adam

from rl.agents import DDPGAgent
from rl.memory import SequentialMemory
from rl.random import OrnsteinUhlenbeckProcess

import argparse, os, yaml

def build_actor_model(num_action, observation_shape):
    action_input = Input(shape=(1,)+observation_shape)
    x = Flatten()(action_input)
    x = Dense(16, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(16, activation="relu")(x)
    x = Dense(num_action, activation="linear")(x)
    actor = Model(inputs=action_input, outputs=x)
    return actor

def build_critic_model(num_action, observation_shape):
    action_input = Input(shape=(num_action,),name='action_input')
    observation_input = Input(shape=(1,)+observation_shape,name='observation_input')
    flattened_observation = Flatten()(observation_input)
    x = Concatenate()([action_input, flattened_observation])
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(32)(x)
    x = Activation('relu')(x)
    x = Dense(1)(x)
    x = Activation('linear')(x)
    critic = Model(inputs=[action_input, observation_input], outputs=x)    
    return (critic, action_input)

def build_agent(num_action, observation_shape):
    actor=build_actor_model(num_action, observation_shape)
    critic, critic_action_input = build_critic_model(num_action, observation_shape)
    #print(critic.summary())
    memory = SequentialMemory(limit=100000, window_length=1)
    random_process = OrnsteinUhlenbeckProcess(size=num_action, theta=.15, mu=0., sigma=.3)
    agent = DDPGAgent(nb_actions=num_action, actor=actor, critic=critic, critic_action_input=critic_action_input,
	                  memory=memory, nb_steps_warmup_critic=100, nb_steps_warmup_actor=100,
	                  random_process=random_process, gamma=.99, target_model_update=1e-3)
    return agent

def train(env,ENV_NAME,training_steps):
	nb_actions = env.action_space.shape[0]
	agent = build_agent(env.action_space.shape[0], env.observation_space.shape)
	agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
	agent.fit(env, nb_steps=training_steps, visualize=False, verbose=1, nb_max_episode_steps=200)
	agent.save_weights('results/weights/ddpg_{}_weights.h5f'.format(ENV_NAME), overwrite=False)
	agent.test(env, nb_episodes=1, visualize=False, nb_max_episode_steps=500)

def test(env,ENV_NAME,num_episodes):
	nb_actions = env.action_space.shape[0]
	agent = build_agent(env.action_space.shape[0], env.observation_space.shape)
	agent.compile(Adam(lr=.001, clipnorm=1.), metrics=['mae'])
	agent.load_weights('results/weights/ddpg_{}_weights.h5f'.format(ENV_NAME))
	agent.test(env, nb_episodes=num_episodes, visualize=False, nb_max_episode_steps=200)

if __name__ == "__main__":
    ENV_NAME = 'AllVar-v0'
    env = gym.make(ENV_NAME)
    parser = argparse.ArgumentParser()
    parser.add_argument('-rRL','--run_RL', dest = 'run_RL', type = bool,
                        default = False, help = 'True to run RL model')
    parser.add_argument('-tRL','--test_RL', dest = 'test_RL', type = bool,
                        default = True, help = 'True to plot RL results')
    parser.add_argument('-cfg','--config', dest = 'config_dir', type = str,
                        default = 'config/config.yml', help = 'where the config file is located')
    args = parser.parse_args()
    with open(args.config_dir, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    if args.run_RL == True:
        training_steps=cfg['training_steps']
        train(env,ENV_NAME,training_steps)
    if args.test_RL == True:
        num_episodes=cfg['test_episodes']
        test(env,ENV_NAME,num_episodes)
        



	
	