"""
Created on Spet 24, 2018
Code imports simulated data, performs data cleaning/imputation and uses polynomial regression 
to create interpreter model (to be used in custom Open Gym env) 

@author: Alex Qatshan)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Import data processing functions
import func.data_processing as HVACData

# Import a few other funcs for the main script
import argparse, os, yaml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-rreg','--run_reg', dest = 'run_regression', type = bool,
                        default = True, help = 'True to run the regression')
    parser.add_argument('-sreg','--save_reg', dest = 'save_regression', type = bool,
                        default = True, help = 'True to save the regression')
    parser.add_argument('-cfg','--config', dest = 'config_dir', type = str,
                        default = 'config/config.yml', help = 'where the config file is located')

    parser.add_argument('-plot','--plot_data', dest = 'plot_data', type = int,
                        default = False, help = 'True to plotsimulation data')
    
    args = parser.parse_args()

    # Set config path & load the config variables.
    CWD_PATH = os.getcwd()
    config_path = os.path.join(CWD_PATH,args.config_dir)
    with open(config_path, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    # Add the current working path to the config var.
    cfg['CWD_PATH'] = CWD_PATH
    if (args.plot_data == True) | (args.run_regression == True):
        HVACData.data_process(cfg,args.plot_data,args.run_regression,args.save_regression)
