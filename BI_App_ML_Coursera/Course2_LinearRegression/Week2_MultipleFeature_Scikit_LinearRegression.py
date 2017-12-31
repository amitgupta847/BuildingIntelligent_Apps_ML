import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.system("mode con cols=120 lines=70")
import sys

##Here we load the crime rate versus house price data and try to predict the value with
#and without the high levarge point.

def run():
   
    sales = pd.read_csv('kc_house_data.csv')
    
    print("Loaded the sales data of {0} shape".format(sales.shape))
    print("Columns are {0}".format(sales.columns))
    
    dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 
              'long':float, 'sqft_lot15':float,'sqft_living':float, 'floors':str, 'condition':int,
              'lat':float, 'date':str, 'sqft_basement':int,'yr_built':int, 'id':str, 'sqft_lot':int,
              'view':int}

    train_data = pd.read_csv('kc_house_train_data.csv',dtype=dtype_dict)
    test_data = pd.read_csv('kc_house_test_data.csv',dtype=dtype_dict)

    print("Splitted sales data into train data of {0} shape, and test data of {1} shape"
                                            .format(train_data.shape, test_data.shape))
  
    features =['sqft_living', 'bedrooms', 'bathrooms']
    target=['price']

    print("We gonna fit a model based on following features {0}".format(features))
    
    model=loadModel(train_data,features ,target )
    
    TestModel(model, test_data,features)

    rss= CalculateRSS(model, test_data[features], test_data[target])
    print("\nComputed RSS on test data for this model is: {0}".format(rss))


def loadModel(train_data, features, target='price'):
  
    x = train_data[features]
    y = train_data[target]
       
    model_multiplefeature = GetModelFit(x,y)

    PrintCofficients(model_multiplefeature,features)

    predictions = model_multiplefeature.predict(x)

    #sns.distplot((Y-prediction),bins=50)
 
    plt.plot(train_data['sqft_living'],train_data['price'],'.', train_data['sqft_living'],predictions,'-')

    plt.show()
    return model_multiplefeature

def CalculateRSS(model, data, originalObs):
    
    prediction=model.predict(data)
    error=originalObs-prediction
    rss = np.vdot(error,error)
    return rss



def TestModel(model, test_data, features, target='price'):
  
    x = test_data[features]
    y = test_data[target]

    predictions = model.predict(x)

    #sns.distplot((Y-prediction),bins=50)
 
    plt.plot(test_data['sqft_living'][0:100],test_data['price'][0:100],'.', 
             test_data['sqft_living'][0:100],predictions[0:100],'.')

    plt.show()

  
def GetModelFit(x,y):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression(normalize=True)
    model.fit(x,y)
    return model


def PrintCofficients(model,features):

    cofficients =model.coef_[0]
    rounded_coeff=np.array([np.round(cf,2) for cf in cofficients ])
    print("Model predicted slope value = {0}, and cofficient values= {1}"
                      .format(model.intercept_,rounded_coeff))

    tup=[(feat,coff) for (feat,coff) in zip(features, rounded_coeff)]
    for feat,coff in tup:
        print("Cofficients for {0} is {1}".format(feat,coff))




run()

