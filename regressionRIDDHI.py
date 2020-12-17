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

#allData = pd.read_csv(r'‪C:\Users\Riddhi\Downloads\All_data.csv', skiprows=0, delimiter=',')#print(y)
with open('All_data2.csv', 'r') as f:
    allData = pd.read_csv(r'C:\Users\Riddhi\Desktop\regression code\All_data2.csv', skiprows=0, delimiter=',')

#allData = csv.reader(open('All_data.csv'), delimiter=',', lineterminator='\n')
#27
#def convert_to_int(col):
#    return[col[0], int(col[1]), int(col[2]), int(col[3]), int(col[4]), int(col[5]), int(col[6]), int(col[7]), int(col[8]), int(col[9]), int(col[10]), int(col[11]), int(col[12]), int(col[13]), int(col[14]), int(col[15]), int(col[16]), int(col[17]), int(col[18]), int(col[19]), int(col[20]), int(col[21]), int(col[22]), int(col[23]), int(col[24]), int(col[25]), int(col[26]), int(col[27])]

x = allData['Population_00']
#print(x)
y = allData['Response_Rate_00']
#print(x)

x = np.array(x).reshape((-1, 1))
print(x)

y = np.array(y)
print(y)

model = LinearRegression().fit(x, y)
r_sq = model.score(x, y)
print("coefficient of determination:", r_sq)

print("intercept:", model.intercept_)
print("slope:", model.coef_)

new_model = LinearRegression().fit(x, y.reshape((-1, 1)))
print("new intercept:", new_model.intercept_)
print("new slope:", new_model.coef_)

y_pred = model.predict(x)
print("predicted response:", y_pred, sep='\n')

slope = new_model.coef_
intercept = new_model.intercept_
line = slope*x+intercept

x = np.array(x)
y = np.array(y)

model = smf.ols('y ~ x', data=allData).fit()
print(model.summary())
print(model.pvalues)

plt.plot(x, y, 'o', markersize=3)
plt.plot(x, slope*x + intercept)
plt.title('Linear Regression Analysis: Response Rate & Population 2000')
plt.xlabel('Population', color='#1C2833')
plt.ylabel('Response Rates')
plt.show()
