# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 21:05:59 2020

@author: RP_PC
"""

import numpy as np
import pandas as pd
import csv
import sklearn
from sklearn import metrics
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from matplotlib.scale import get_scale_names
from matplotlib.axes import Axes, Subplot
import logging
import logging
import numpy as np
import pandas as pd
import csv
import sklearn
from sklearn import metrics
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import linregress
import seaborn as sns






formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

# first file logger
logger = setup_logger('first_logger', 'first_logfile.log')
logger.info('This is just info message')


#allData = pd.read_csv(r'C:\Users\nuran\Desktop\Senior_Project\All_data.csv', skiprows=0, delimiter=',')#print(y)
with open('All_data.csv', 'r') as f:
    allData = pd.read_csv('All_data.csv', skiprows=0, delimiter=',')

    


xtemp00 = ["Population_00","White_00","Black_00","Asian_00","Hispanic_00","Republican_00","Democratic_00","Independence_00","Conservative_00","Liberal_00","Green_00","Working_Families_00","AvgHousePrice_00"]

xtemp10 = ["Population_10","White_10","Black_10","Asian_10","Hispanic_10","Democratic_10","Republican_10","Independence_10","Conservative_10","Working_Families_10","Green_10","AvgHousePrice_10"]

xt = [xtemp00, xtemp10]

y00 = allData['Response_Rate_00']
y10 = allData['Response_Rate_10'] 


for nn in  xtemp00:
    logger = setup_logger(nn, nn +'.log')
    x = allData[nn]
    y = y00
    x = np.array(x).reshape((-1, 1))
    #print(x)
    logger.info("Representation of Models")
    logger.info(nn)
    #logger.info(x)
    
    y = np.array(y)
    #print(y)
    logger.info(y)
    
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    #print("coefficient of determination:", r_sq)
    #print("intercept:", model.intercept_)
    #print("slope:", model.coef_)
    logger.info("coefficient of determination:", r_sq)
    logger.info("intercept:", model.intercept_)
    logger.info("slope:", model.coef_)
    
    new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
    #print("intercept:", new_model.intercept_)
    #print("slope:", new_model.coef_)
    logger.info("intercept:", new_model.intercept_)
    logger.info("slope:", new_model.coef_)
    
    y_pred = model.predict(x)
    #print("predicted response:", y_pred, sep='\n')
    logger.info("predicted response:", y_pred)
    
    slope = new_model.coef_
    intercept = new_model.intercept_
    line = slope*x+intercept
    
    model = smf.ols('y ~ x', data=allData).fit()
    #print(model.summary())
    ms= model.summary()
    #print(model.pvalues)
    logger.info(ms)
    logger.info(model.pvalues)


for nn in  xtemp10:
    logger = setup_logger(nn, nn +'.log')
    x = allData[nn]
    y = y10
    x = np.array(x).reshape((-1, 1))
    #print(x)
    logger.info("Representation of Models")
    logger.info(nn)
    #logger.info(x)
    
    y = np.array(y)
    #print(y)
    logger.info(y)
    
    model = LinearRegression().fit(x, y)
    r_sq = model.score(x, y)
    #print("coefficient of determination:", r_sq)
    #print("intercept:", model.intercept_)
    #print("slope:", model.coef_)
    logger.info("coefficient of determination:", r_sq)
    logger.info("intercept:", model.intercept_)
    logger.info("slope:", model.coef_)
    
    new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
    #print("intercept:", new_model.intercept_)
    #print("slope:", new_model.coef_)
    logger.info("intercept:", new_model.intercept_)
    logger.info("slope:", new_model.coef_)
    
    y_pred = model.predict(x)
    #print("predicted response:", y_pred, sep='\n')
    logger.info("predicted response:", y_pred)
    
    slope = new_model.coef_
    intercept = new_model.intercept_
    line = slope*x+intercept
    
    model = smf.ols('y ~ x', data=allData).fit()
    #print(model.summary())
    ms= model.summary()
    #print(model.pvalues)
    logger.info(ms)
    logger.info(model.pvalues)



allData = pd.read_csv('All_data.csv', skiprows=0, delimiter=',')

xtemp00 = ["Population_00","White_00","Black_00","Asian_00","Hispanic_00","Republican_00","Democratic_00","Independence_00","Conservative_00","Liberal_00","Green_00","Working_Families_00","AvgHousePrice_00"]

xtemp10 = ["Population_10","White_10","Black_10","Asian_10","Hispanic_10","Democratic_10","Republican_10","Independence_10","Conservative_10","Working_Families_10","Green_10","AvgHousePrice_10"]

xt = [xtemp00, xtemp10]

y00 = allData['Response_Rate_00']
y10 = allData['Response_Rate_10'] 


for nn in  xtemp00:
    x = allData[nn]
    y = y00
    #print (x)
    plt.figure(figsize=(10,10))
    sns.regplot(x, y)
    plt.plot([x],[y], 'o', label = x, color='red', markersize='3')#
    
    plt.title('Linear Regression Analysis: Response Rate & ' + nn )
      #  plt.xlabel(x, color='#1C2833')
       # plt.ylabel('Response Rate')
    plt.savefig( nn +'.pdf')
 #   plt.show()

    
for nn in  xtemp10:
    x = allData[nn]
    y = y10
    #print (x)
    plt.figure(figsize=(10,10))
    sns.regplot(x, y)
    plt.plot([x],[y], 'o', label = x, color='red', markersize='3')#
    
    plt.title('Linear Regression Analysis: Response Rate & ' + nn )
      #  plt.xlabel(x, color='#1C2833')
       # plt.ylabel('Response Rate')
    plt.savefig( nn +'.pdf')
 #   plt.show()

