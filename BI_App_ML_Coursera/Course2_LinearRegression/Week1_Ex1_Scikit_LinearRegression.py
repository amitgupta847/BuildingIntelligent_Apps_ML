import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.system("mode con cols=120 lines=70")
import sys

##Here we load the crime rate versus house price data and try to predict the value with
#and without the high levarge point.

def run():
    sales = pd.read_csv('LR_Data\Philadelphia_Crime_Rate_noNA.csv')
    print(sales.columns)
    loadModel(sales, True)
    loadModel(sales, False)
    plt.show()


def loadModel(sales,isWithlevPoint=True):
      
    #reshaping, bec SCIKIT expect single feature to reshape in this order.
 
    if(isWithlevPoint):
        x = sales['CrimeRate'].values.reshape(-1,1)
        y = sales['HousePrice']
    else:
        sales_noCC=sales[sales['MilesPhila'] != 0.0] 
        x = sales_noCC['CrimeRate'].values.reshape(-1,1)
        y = sales_noCC['HousePrice']
       
    crime_model = GetModelFit(x,y)
    print("Model predicted slope value = {0}, and cofficient value for crime rate= {1}"
          .format(crime_model.intercept_,crime_model.coef_ ))

    predictions = crime_model.predict(x)

    #sns.distplot((Y-prediction),bins=50)
 
    plt.plot(sales['CrimeRate'],sales['HousePrice'],'.', x,predictions,'-')
    plt.xlabel('Crime Rate')
    plt.ylabel('House Prices')
    plt.title('Decrease in house price with the increase in crime rate')

def GetModelFit(x,y):
    from sklearn.linear_model import LinearRegression
    crime_model = LinearRegression(normalize=True)
    crime_model.fit(x,y)
    return crime_model



run()

