
# HVAC Control Using Deep Reinforcement Learning


## Abstract

Buildings consume 30% of total energy worldwide. 40% of this enery is consumed by cooling 
and heating systems.Facility managers use Energy Management Systems (EMS) to control energy cost 

There is a trade-off between energy consumption and comfort inside buildings (Temperature). Existing systems use approximate models with static parameters

These systems xhibit a suboptimal performance (i.e optimum energy does not often imply optimum emperature and vice versa) These systems also fail to account for changes in environment (occupancy and load)


Our approach is to use deep reinforcement learning to control cooling system. This approach does not assume any specific model for the system. Cooling control policy is learned and derived from data. An Agent, via trial-and-error, can make optimal actions even for very complex environments. This system can adapt to changes in environment (inside and outside the facility's/building

![alt text](https://github.com/qatshana/CNN-MNIST/blob/master/HVAC_Control_DRL/images/DRL-Cooling-Model.png)

##Installation
The file, requirement.txt, contains the python package dependencies for this project. Installation can be performed via

pip install -r requirement.txt

##Input Data
I used EnergyPlus to simulate a 2 Zone Data Center with two cooling systems (see picture below) and 6 different inputs (3 for each cooling system)
Output data from simulation is saved in the data folder


##Inference
Pre-trained weights can be downloaded from DropBox. Test set images can also be found in the datasets folder.

##Post-processing
The traced roof lines would be shown on the original input image

##Packaging
Install as a single package

pip install setup.py