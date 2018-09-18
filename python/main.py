#!/usr/bin/env python3.6
"""
Created on Sept 17, 2018

Code to control HVAC systems using Deep Reinforcement Learnigns

The main functions are:
    1) Simulation model creation based on data using linear regression (polynomia) to map inputs --> outputs (Data generated from EnergyPlus tool)
    2) Deep Reinforcement Learning to derive inputs from target outputs.
    3) Rendering tool for the states. This can be run concurrently while training the agent.

@author: aqatshan
"""

import argparse, os, yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-rreg','--run_reg', dest = 'run_regression', type = bool,
                        default = True, help = 'True to run the regression')
    parser.add_argument('-sreg','--save_reg', dest = 'save_regression', type = bool,
                        default = False, help = 'True to save the regression')
    parser.add_argument('-rRL','--run_RL', dest = 'run_RL', type = bool,
                        default = True, help = 'True to run RL model')
    parser.add_argument('-pRL','--plot_RL', dest = 'plot_RL', type = bool,
                        default = True, help = 'True to plot RL results')
    parser.add_argument('-plot','--plot_data', dest = 'plot_data', type = int,
                        default = False, help = 'True to plot Fluent simulation data in 3d')
    parser.add_argument('-cfg','--config', dest = 'config_dir', type = str,
                        default = '', help = 'configration file location')
    args = parser.parse_args()

    # Set config path & load the config variables.
    CWD_PATH = os.getcwd()
    cfg['CWD_PATH'] = CWD_PATH
    config_path = os.path.join(CWD_PATH,args.config_dir)
    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    if (args.plot_data==True):
        pass
    if (args.run_regression == True):
        pass
    if args.run_RL == True:
        pass
    if args.plot_RL == True:
        pass
