
# HVAC Control Using Deep Reinforcement Learning


## Abstract

Buildings consume 30% of total energy worldwide. 40% of this enery is consumed by cooling 
and heating systems.Facility managers use Energy Management Systems (EMS) to control energy cost 

There is a trade-off between energy consumption and comfort inside buildings (Temperature). Existing systems use approximate models with static parameters

These systems exhibit a suboptimal performance (i.e optimum energy does not often imply optimum emperature and vice versa) These systems also fail to account for changes in environment (occupancy and load)

Our approach is to use deep reinforcement learning to control cooling system. This approach does not assume any specific model for the system. Cooling control policy is learned and derived from data. An Agent, via trial-and-error, can make optimal actions even for very complex environments. This system can adapt to changes in environment (inside and outside the facility's/building

![alt text](https://github.com/qatshana/Cool.AI/blob/master/images/DRL-Cooling-Model.png)

##Installation
The file, requirement.txt, contains the python package dependencies for this project. Installation can be performed via

pip install -r requirement.txt

## Input Data
I used EnergyPlus to simulate a 1 Zone Data Center in San Fransico with one cooling systems (see picture below) and 3 different inputs 
Output data from simulation is saved in the data folder


## Inference

Pre-trained weights are saved in reesults/weights folder. You can run test using two different approachs as stated in the presentation. 1) A linear approach (Minimum Viable Product), or 2) DDPG approach:

1) using linear model (MVP)

python python/main_MVP.py 

2) using DDPG

python python/main_DDPG.py 




## Performance 
The graphs below present the performance of the system using the two methods (linear vs. DDPG). Results can be created for any number of samples by setting test_episodes in config/config.yml file 




## Packaging
Install as a single package

pip install setup.py
