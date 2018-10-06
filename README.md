
# COOL.AI
## Cooling System Control Using Deep Reinforcement Learning


## Abstract

Buildings consume 30% of total energy worldwide. 40% of this enery is consumed by cooling 
and heating systems.Facility managers use Energy Management Systems (EMS) to control energy cost 

There is a trade-off between energy consumption and comfort inside buildings (Temperature). Existing systems use approximate models with static parameters

These systems xhibit a suboptimal performance (i.e optimum energy does not often imply optimum emperature and vice versa) These systems also fail to account for changes in environment (occupancy and load)


Our approach is to use deep reinforcement learning to control cooling system. This approach does not assume any specific model for the system. Cooling control policy is learned and derived from data. An Agent, via trial-and-error, can make optimal actions even for very complex environments. This system can adapt to changes in environment (inside and outside the facility's/building

![alt text](https://github.com/qatshana/HVAC_Control_DRL//blob/master/images/DRL-Cooling-Model.png)

##Installation
The file, requirement.txt, contains the python package dependencies for this project. Installation can be performed via

pip install -r requirement.txt

## Input Data
I used EnergyPlus to simulate a 2 Zone Data Center with two cooling systems (see picture below) and 6 different inputs (3 for each cooling system)
Output data from simulation is saved in the data folder

![alt text](https://github.com/qatshana/Cool.AI/blob/master/images/OutsideTemp.png)
![alt text](https://github.com/qatshana/Cool.AI/blob/master/images/ITU_Load.png)

## Training:

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

## Packaging
Install as a single package

pip install setup.py
