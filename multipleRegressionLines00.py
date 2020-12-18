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
from statsmodels.api import add_constant

allData = pd.read_csv("All_data.csv")

x = allData[['Population_00', 'White_00', 'Black_00', 'Asian_00', 'Hispanic_00', 'Republican_00',
'Democratic_00', 'Independence_00', 'Conservative_00', 'Liberal_00', 'Green_00', 'Working_Families_00', 'AvgHousePrice_00']]

y = allData['Response_Rate_00'] #2000

model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print("coefficient of determination:", r_sq)

new_model = LinearRegression().fit(x, y)
print("intercept:", new_model.intercept_)
print("slope:", new_model.coef_)

slope = new_model.coef_
intercept = new_model.intercept_
line = slope*x+intercept
model = smf.ols('y ~ x', data=allData).fit()
print(model.summary())
print(model.pvalues)
