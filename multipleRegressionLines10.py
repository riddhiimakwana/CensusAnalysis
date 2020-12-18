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

allData = pd.read_csv('All_data.csv', skiprows=0, delimiter=',')

y = allData['Response_Rate_10'] #y

white10 = allData['White_10']
black10 = allData['Black_10']
population10 = allData['Population_10']
asian10 = allData['Asian_10']
hispanic10 = allData['Hispanic_10']
republican10 = allData['Republican_10']
democratic10 = allData['Democratic_10']
independence10 = allData['Independence_10']
conservative10 = allData['Conservative_10']
green10 = allData['Green_10']
workingFamilies10 = allData['Working_Families_10']
avgHousePrice10 = allData['AvgHousePrice_10']
plt.figure(figsize=(10,10))

sns.regplot(white10, y)
sns.regplot(black10, y)
sns.regplot(population10, y)
sns.regplot(asian10, y)
sns.regplot(hispanic10, y)
sns.regplot(republican10, y)
sns.regplot(democratic10, y)
sns.regplot(independence10, y)
sns.regplot(conservative10, y)
sns.regplot(green10, y)
sns.regplot(workingFamilies10, y)
sns.regplot(avgHousePrice10, y)


plt.plot([white10],[y], 'o', label = 'white', color='red', markersize='3')
plt.plot([black10],[y], 'o', label = 'black', color='black', markersize='3')
plt.plot([population10],[y], 'o', label = 'population', color='orange', markersize='3')
plt.plot([asian10],[y], 'o', label = 'asian', color='yellow', markersize='3')
plt.plot([hispanic10],[y], 'o', label = 'hispanic', color='green', markersize='3')
plt.plot([republican10],[y], 'o', label = 'republican', color='blue', markersize='3')
plt.plot([democratic10],[y], 'o', label = 'democratic', color='violet', markersize='3')
plt.plot([independence10],[y], 'o', label = 'independence', color='brown', markersize='3')
plt.plot([conservative10],[y], 'o', label = 'conservative', color='teal', markersize='3')
plt.plot([green10],[y], 'o', label = 'Black', color='green', markersize='3')
plt.plot([workingFamilies10],[y], 'o', label = 'working families', color='aqua', markersize='3')
plt.plot([avgHousePrice10],[y], 'o', label = 'average house price', color='purple', markersize='3')

#print line of best fit

plt.title('Linear Regression Analysis: Response Rate & Factors 2010')
plt.xlabel('Factors 2010', color='#1C2833')
plt.ylabel('Response Rate')
#plt.show()
plt.savefig( "MLR2010.pdf")
#
# clean code
