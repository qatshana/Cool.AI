#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Spet 24, 2018
Code creates an RL agent that determines required input parameters for desired outputs
for HVAC system control

.
The main functions are:
    1) Polynomial regression to map inputs to outputs from simulated data
    2) Reinforcement learning algorithm to derive inputs from target outputs
    3) Custom Open Gym Env 

@author: Alex Qatshan
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import the Control_RL functions

import python.func.data_processing as HVACData


# Import a few other funcs for the main script
import argparse, os, yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-rreg','--run_reg', dest = 'run_regression', type = bool,
                        default = False, help = 'True to run the regression')
    # IMPORTANT NOTE: If you turn on the below and save the regression, make sure
    # you do not overwrite it before rolling out your policy. This should only be
    # toggled on or overwritten when training the model. The identical regression
    # coefficients have to be used when rolling out as training.
    parser.add_argument('-sreg','--save_reg', dest = 'save_regression', type = bool,
                        default = True, help = 'True to save the regression')
    parser.add_argument('-rRL','--run_RL', dest = 'run_RL', type = bool,
                        default = False, help = 'True to run RL model')
    parser.add_argument('-pRL','--plot_RL', dest = 'plot_RL', type = bool,
                        default = False, help = 'True to plot RL results')
    parser.add_argument('-plot','--plot_data', dest = 'plot_data', type = int,
                        default = False, help = 'True to plot simulation data')
    parser.add_argument('-cfg','--config', dest = 'config_dir', type = str,
                        default = 'config/config.yml', help = 'where the config file is located')
  
    args = parser.parse_args()

    with open(args.config_dir, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    if (args.plot_data==True) | (args.run_regression == True):

        HVACData.data_process(cfg,args.plot_data,args.run_regression,args.save_regression)

    if args.run_RL == True:
        pass

    if args.plot_RL == True:
        pass