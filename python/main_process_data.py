#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 11:49:31 2018

This codebase creates an RL agent that determines input parameters that reach
the desired ouputs for tuning a rocket engine component.

The main functions are:
    1) Linear regression to map inputs --> outputs based on some cached flow
    simulation data from Fluent.
    2) Reinforcement Learning to derive inputs from target outputs.
    3) Plotting some metrics of the training progress. This can be run
    concurrently while training the agent.


@author: ninalopatina
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import the Rocket_RL functions
import func.data_processing as HVACData

# Import a few other funcs for the main script
import argparse, os, yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-rreg','--run_reg', dest = 'run_regression', type = bool,
                        default = True, help = 'True to run the regression')
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
                        default = False, help = 'True to plot Fluent simulation data in 3d')
    parser.add_argument('-cfg','--config', dest = 'config_dir', type = str,
                        default = 'config/config.yml', help = 'where the config file is located')
    parser.add_argument('-rayinit','--rayinit', dest = 'rayinit', type = bool,
                        default = False, help = 'should only be init the first time you run if running in an IDE')
    args = parser.parse_args()

    # Set config path & load the config variables.
    CWD_PATH = os.getcwd()
    config_path = os.path.join(CWD_PATH,args.config_dir)
    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Add the current working path to the config var.
    cfg['CWD_PATH'] = CWD_PATH

    # If importing new data, set plot_data to true to take a glance at it to verify
    # if it looks reasonable.
    if (args.plot_data==True) | (args.run_regression == True):

        HVACData.data_process(cfg,args.plot_data,args.run_regression,args.save_regression)

    
    if args.plot_RL == True:
        pass
