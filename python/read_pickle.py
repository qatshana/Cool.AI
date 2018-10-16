"""
Created on Spet 24, 2018
Code reads pickle file for polynomical coef and printout results for sanity check


@author: Alex Qatshan
"""

import pickle
import os 
join_path = 'results/pickles'
fname = 'TZ1.p'
pickleFile = os.path.join(join_path,fname)
[coef,powers,intercept,mins,maxes] = pickle.load(open(pickleFile,'rb'))
out = {'coef': coef, 'powers':powers,'intercept':intercept}
print (out)

