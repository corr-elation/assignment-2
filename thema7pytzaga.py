
from sklearn.linear_model import LinearRegression
from math import sqrt 
import statistics 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

ffdata = pd.read_csv("C:\\Users\\Downloads\\forestfires.csv", header=0, sep=",", engine='python')
print(ffdata)

ffdata = ffdata.sample(frac=1).reset_index(drop=True)

kf = KFold(n_splits=10)
print("\nLinear regression model: Area = b1*Temp + b2*Wind + b3*Rain + b0\n")

allRMSE = np.empty(shape=[0, 1])
testNumber = 0

for train_index, test_index in kf.split(ffdata):
    testNumber += 1
    trainingData = ffdata.iloc[train_index, :]
    testData = ffdata.iloc[test_index, :]
    lm = LinearRegression(normalize=False, fit_intercept=True)
    estimatedModel = lm.fit(trainingData.loc[:, ['temp', 'wind', 'rain']], trainingData.loc[:, ['area']])
    print(">>>Iteration ", testNumber, sep='')
    print("\tEstimated coefficients:")
    print("\t\tb1=", estimatedModel.coef_[0][0], sep='')
    print("\t\tb2=", estimatedModel.coef_[0][1], sep='')
    print("\t\tb3=", estimatedModel.coef_[0][2], sep='')
    print("\t\tb0=", estimatedModel.intercept_, sep='')
    predictedExpenditure = estimatedModel.predict(testData.loc[:, ['temp', 'wind', 'rain']])
    RMSE = sqrt(mean_squared_error(testData.loc[:, ['area']], predictedExpenditure))
    print("\t\tModel RMSE=", RMSE, sep='')
    allRMSE = np.append(allRMSE, RMSE)

print("\n=======================================================")
print(" Final result: Mean RMSE of tests:", statistics.mean(allRMSE), sep='' )
print("=======================================================")

ffdata = ffdata.loc[ffdata.loc[:,'area']< 3.2]

allRMSE = np.empty(shape=[0, 1])
testNumber = 0

for train_index, test_index in kf.split(ffdata):
    testNumber += 1
    trainingData = ffdata.iloc[train_index, :]
    testData = ffdata.iloc[test_index, :]
    lm = LinearRegression(normalize=False, fit_intercept=True)
    estimatedModel = lm.fit(trainingData.loc[:, ['temp', 'wind', 'rain']], trainingData.loc[:, ['area']])
    print(">>>Iteration ", testNumber, sep='')
    print("\tEstimated coefficients:")
    print("\t\tb1=", estimatedModel.coef_[0][0], sep='')
    print("\t\tb2=", estimatedModel.coef_[0][1], sep='')
    print("\t\tb3=", estimatedModel.coef_[0][2], sep='')
    print("\t\tb0=", estimatedModel.intercept_, sep='')
    predictedExpenditure = estimatedModel.predict(testData.loc[:, ['temp', 'wind', 'rain']])
    RMSE = sqrt(mean_squared_error(testData.loc[:, ['area']], predictedExpenditure))
    print("\t\tModel RMSE=", RMSE, sep='')
    allRMSE = np.append(allRMSE, RMSE)

print("\n=======================================================")
print(" Final result: Mean RMSE of tests for the subset:", statistics.mean(allRMSE), sep='' )
print("=======================================================")
