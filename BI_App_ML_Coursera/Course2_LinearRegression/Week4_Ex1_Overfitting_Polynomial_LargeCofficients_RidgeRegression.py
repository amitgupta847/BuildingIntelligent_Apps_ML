import pandas as pd
import numpy as np
import math
import random
import matplotlib.pyplot as plt
import os

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

#os.system("mode con cols=120 lines=70")
os.system("mode con cols=120")

#Here we will generate a Sine curve with some noise and will try to fit model with varying polynomial degrees.
# We will see how coefficient size grows too large with mulitple feature of varying degree and 
# finally we will see how Ridge Regression will help us reducing the coefficient or reducing the overfitting of the model
    

def run():
   
    (x,y)=build_SineCurved_Data()
    plot_data(x,y)

    FitMultipleModels(pd.DataFrame({'X1':x,'Y':y}))
      
    print("\n\n Lets see the difference in cofficients using Ridge Regression")
    ApplyRidgeRegression(x,y)

    plt.show()

def ApplyRidgeRegression(x,y):
     F1= plt.Figure()
     plot_data(x,y)
     data=pd.DataFrame({'X1':x,'Y':y})

     for l2_penalty in [1e-25, 1e-10, 1e-6, 1e-3, 1e2]:
        buildPolynomialDataModel(data, deg=16,plot_label='Ridge: ' +str(l2_penalty),ridge=True, l2_penalty=l2_penalty)
     

def build_SineCurved_Data(showGraph=False):
    random.seed(98103)
    n = 30
    numbers =[random.random() for i in range(n)]
    x = np.array(numbers)
    x.sort()

    sinfunc = lambda x: math.sin(4*x)
    y = np.array([sinfunc(xi) for xi in x])
   
    if(showGraph):
        plot_data(x,y)

    #Add random Gaussian noise to y
    random.seed(1)
    e = np.array([random.gauss(0,1.0/3.0) for i in range(n)])
    y = y + e
   
    return (x,y)

 
def FitMultipleModels(data):
    model1 = buildPolynomialDataModel(data,2, plot_label='2 Deg Model Fit')
    model2 = buildPolynomialDataModel(data,4, plot_label='4 Deg Model Fit')
    model3 = buildPolynomialDataModel(data,16, plot_label='16 Deg Model Fit')

def plot_data(x,y): 
    f1=plt.figure()
    plt.plot(x,y,'k.')
    plt.xlabel('x')
    plt.ylabel('y')

    #plt.show()
def buildPolynomialDataModel(data, deg, plot_label='', showCoffandPlot=True, ridge=False, l2_penalty=0):
    
    polynomial_data = polynomial_features(data.drop('Y', axis=1), deg)
   
    my_features = polynomial_data.columns # get the name of the features
    
    model = GetModel(polynomial_data[list(my_features)],data['Y'],ridge, l2_penalty)
    
    if showCoffandPlot:
        PrintCofficients(model, my_features)
        plt.plot(polynomial_data['X1'], model.predict(polynomial_data), label=plot_label)
        plt.legend(loc='upper left')
        plt.axis([0,1,-1.5,2])
    
    return model

def GetModel(x,y,ridge=False,l2_penalty=0):

    if ridge==False:
        model = LinearRegression()
    else:
        model = Ridge(alpha=l2_penalty, normalize=True)
        

    model.fit(x, y)
    return model



def PrintCofficients(model, feature):
    if hasattr(model, 'alpha') ==False:
        print("\n\Model with {0} weights: {1}".format(feature, model.coef_))
    else:
        print("\n\Model with L2Penality: {0}, and with {1} weights: {2}".format(model.alpha,feature, model.coef_))

    print("Intercept weights:", model.intercept_)

# amit: have not used this method yet
def print_coefficients(model):    
    # Get the degree of the polynomial
    deg = len(model.coef_)
    # Get learned parameters as a list
    w = list(model.coef_)
    # Numpy has a nifty function to print out polynomials in a pretty way
    # (We'll use it, but it needs the parameters in the reverse order)
    print ('Learned polynomial for degree' + str(deg) + ':')
    w.reverse()
    print (np.poly1d(w))

def polynomial_features(data, degree):
    data_copy=data.copy()
    for i in range(1,degree):
        data_copy['X'+str(i+1)]=data_copy['X'+str(i)]*data_copy['X1']
    return data_copy

def CalculateRSS(model, data, originalObs):
    
    prediction = model.predict(data)
    assert(prediction.shape == originalObs.shape)
    error = originalObs - prediction
    rss = np.vdot(error,error)
   
    return rss/(2*len(prediction))




def PlotValidationRSS(x,validation_rss_values):
    f4 = plt.figure()
  
    plt.plot(x,validation_rss_values,'-',label='validation')

    plt.title("RSS (Squared error) for Validation data for 1 to 15 Deg Models")
    plt.legend()
    plt.xlabel('Complexity (Degree)')
    plt.ylabel('RSS / (1/2M)')



run()

