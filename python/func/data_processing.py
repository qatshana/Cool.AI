"""
Created on Spet 24, 2018
Code imports simulated data, performs data cleaning/imputation and uses polynomial regression 
to create interpreter model (to be used in custom Open Gym env) 

@author: Alex Qatshan

"""

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
   3) Running a regression to make a model of the simulation environment
   4) Also, there two functions to plot data in 2D and 3D
    """
    data_path = os.path.join(cfg['data_file_path'])
    df = load_data(data_path)
    df = clean_data(cfg,df)
    if run_regression == True:
        reg_runner(cfg,df,save_regression)
 
        

def load_data(data_path):
    """
    This function loads csv data from the data path defined in config
    """
    df = pd.read_csv(data_path)
    return df

def clean_data(cfg,df):
    """
    This function cleans data and impute some missing load data ITU
    """
 
    df.drop(['Date/Time'],axis=1,inplace=True)    
    return df

def norm_data(cfg,df):
    """
    Normalize the data.
    """
    if cfg['norm_mode'] == 'full':
        x = df.values #returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled,columns = cfg['all_var'])  
    elif cfg['norm_mode'] == 'max':
        df = df/df.max()

    df = df * cfg['scale_var']
    return df

def graph(formula, m,c, x_range):
    """
    This function graphs the line over linear data
    """
    x = np.array(x_range)
    y = formula(x,m,c)  # <- note now we're calling the function 'formula' with x
    plt.plot(x, y, c = 'red')

def my_formula(x,m,c):
    """
    This function is the formula for the graph function
    """
    return m*x +c

def plot_data():
    """
    This function plots sub-sample of the data for Zone1
    """
    ts = 5000
    interval = 100
    te = ts+interval
    plt.subplot(221)
    plt.plot(df['TDXZ1'][ts:te])
    plt.subplot(222)
    plt.plot(df['TZ1'][ts:te])
    plt.subplot(223)
    plt.plot(df['TDECZ1'][ts:te])
    plt.subplot(224)
    plt.plot(df['TIECZ1'][ts:te])


def save_plot(cfg,fig,title):
    """ 
    This function saves the generated figures in the file indicated by the title
    """
    save_dir = os.path.join(cfg['CWD_PATH'], cfg['repo_path'],cfg['result_path'],cfg['data_result_path'])
    fig.savefig(save_dir + title+".png")

def plot_3dvar(cfg,df):
    """
    This function plots 3 input variables in 3d, with output variable as the 
    color of the points. For 3 output variables, this will create 3 plots. 
    The user manually adjusts which 3/4 input variables to plot.
    """
    for var_o in cfg['out_var']:
        threedee = plt.figure().gca(projection='3d')
        # Manually set which 3 out of 4 variables you want to visualize in a 3d plot, in cfg:
        x = cfg['in_var'][cfg['xvar']]
        y = cfg['in_var'][cfg['yvar']]
        z = cfg['in_var'][cfg['zvar']]
        p = threedee.scatter(xs = df[x], ys = df[y], zs= df[z] ,c=df[var_o])
        threedee.set_xlabel(x), threedee.set_ylabel(y), threedee.set_zlabel(z)
        threedee.set_title(var_o) #title is the output variables
        plt.colorbar(p)
        save_plot(cfg,plt,title = var_o)

def reg_runner(cfg,df,save_regression):
    """
    This function runs a polynomial regression on 5 input variables to create
    a model of the outputs. The resulting polynomial function is saved in a 
    pickle format, which is accessed by the RL environment as the interpreter.

    """

    df_inputs = df[['TDXZ1','TDECZ1','TIECZ1','TOUT','ITUZ1']]

    for var in cfg['out_var']:
        #Split into train and test set
        X_train, X_test, y_train, y_test = train_test_split(df_inputs, 
                        df[var],test_size=cfg['test_set_fraction'])
         
        # Make a pipeline model with polynomial transformation and linear regression
        #run it for increasing degree of polynomial (complexity of the model)
        # Normalize is set to False because data was already normalized.
        for degree in range(cfg['degree_min'],cfg['degree_max']+1):
            if cfg['reg_model'] == 'linreg':
            
                model = make_pipeline((PolynomialFeatures(degree, interaction_only=False)), 
                                      LinearRegression(normalize = True))
                k = 'linearregression'

            elif cfg['reg_model'] == 'ridge':
                model = make_pipeline((PolynomialFeatures(degree, interaction_only=False)), 
                              Ridge(alpha = cfg['alpha']))
                k = 'ridge'
            
            model.fit(X_train,y_train)
            test_score = model.score(X_test,y_test)
            print(cfg['reg_model'],' results for degree ', str(degree), '& var ',var,': ', test_score)
            
        coef = model.named_steps[k].coef_
        powers = model.named_steps['polynomialfeatures'].powers_
        intercept = model.named_steps[k].intercept_
       
        if save_regression == True:
            fname = (var+".p")
            pickle_path = os.path.join('results/pickles/',fname)
            pickle.dump([coef,powers,intercept, df.min(),df.max()],open(pickle_path,'wb'))