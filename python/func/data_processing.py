#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 11:57:40 2018

This set of functions imports the flow simulation data, plots the data, and creates a regression model.
The output is the regression model that the RL environment uses to approximate the simulation outputs. 
The output can also include figures if you would like to see the data visualized. 


@author: ninalopatina
"""
# Import all the packages for functions:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import os

from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

def data_process(cfg, plot_data,run_regression,save_regression):
    """
   This function runs the data processing pipeline: 
   1) loading the file into a dataframe
   2) cleaning the data
   3) (optional) plot some data
   4) Running a regression to make a model of the simulation environment
    """
   pass