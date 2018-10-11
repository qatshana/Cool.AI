
# COOL.AI
## Cooling System Control Using Deep Reinforcement Learning


## Abstract



Buildings consume 30% of total energy worldwide. 40% of this enery is consumed by cooling 
and heating systems. 

There is a trade-off between energy consumption and comfort inside buildings (Temperature). 

Existing systems use approximate models with static parameters. These systems exhibit a suboptimal performance (i.e optimum energy does not often imply optimum emperature and vice versa). These systems also fail to account for changes in environment (occupancy and load)


Our approach is to use deep reinforcement learning to control cooling system. This approach does not assume any specific model for the system. Cooling control policy is learned and derived from data. An Agent, via trial-and-error, can make optimal actions even for very complex environments. This system can adapt to changes in environment (inside and outside the facility's/building. Please find below a link to the full presentation for this project: (https://docs.google.com/presentation/d/1yqvlOii1ajwODI5cSZdmXYTSz7KpY64holnhCF_5GHo/edit?usp=sharing)

In this project, I present an end-to-end process to design, train and deploy deep reinforcement learning system 

![alt text](https://github.com/qatshana/Cool.AI/blob/master/images/End-to-End-System.png)

## Installation

1) create env file and install requirement.txt as below. The file contains the python package dependencies for this project. Installation can be performed via

pip install -r requirement.txt

2) copy the following custom OpenAI Gym env to your newly created gym env (myNewEnv in this example):

cp python/envs/__init__.py ~/anaconda3/envs/myNewEnv/lib/python3.6/site-packages/gym/envs

cp -R python/envs/ControlRL/ ~/anaconda3/envs/myNewEnv/lib/python3.6/site-packages/gym/envs


## Input Data
I used EnergyPlus to generate the data for a single zone data center in San Francisco with one cooling system and 3 setpoint for control. Below are the charts for distribution of outside temp and data center server load

![alt text](https://github.com/qatshana/Cool.AI/blob/master/images/OutsideTemp.png)
![alt text](https://github.com/qatshana/Cool.AI/blob/master/images/ITU_Load.png)

## Training

Training can be performed on new data set by executing the following command

python python/main_DDPG -rRL True

Currently, number of samples for training data is set at 30,000. This parameter is training_steps and is set in config/config.yml 


## Inference
DDPG system weights are saved in results/weights. User can run two types of tests: Linear and DDPG

1) Linear test:

python python/main_MVP.py

2) DDP test:

python python/main_DDPG.py

## Performance & Results

![alt text](https://github.com/qatshana/Cool.AI/blob/master/images/ddpg_overshoot.png)
![alt text](https://github.com/qatshana/Cool.AI/blob/master/images/linear_fluctuate_1.png)


