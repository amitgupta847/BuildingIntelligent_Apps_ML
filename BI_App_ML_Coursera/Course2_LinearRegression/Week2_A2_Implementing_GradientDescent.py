import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from math import sqrt 
os.system("mode con cols=120 lines=70")
import sys

my_output = 'price'
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 
              'long':float, 'sqft_lot15':float,'sqft_living':float, 'floors':str, 'condition':int,
              'lat':float, 'date':str, 'sqft_basement':int,'yr_built':int, 'id':str, 'sqft_lot':int,
              'view':int}

def run():
   
    sales = pd.read_csv('LR_Data\kc_house_data.csv')
   
    print("Loaded the sales data of {0} shape".format(sales.shape))
    print("\n Columns are {0}".format(sales.columns))
    
    train_data = pd.read_csv('LR_Data\kc_house_train_data.csv',dtype=dtype_dict)
    test_data = pd.read_csv('LR_Data\kc_house_test_data.csv',dtype=dtype_dict)

    print("Splitted sales data into train data of {0} shape, and test data of {1} shape"
                                            .format(train_data.shape, test_data.shape))

    print("\n\n Simple Feature Model")
    SimpleFeatureModel(train_data,test_data)

    print("\n\n Multi Feature Model")
    MultiFeatureModel(train_data,test_data)
    plt.show()

def SimpleFeatureModel(train_data,test_data):
    
    simple_features = ['sqft_living']
    
    (simple_feature_matrix, output) = get_numpy_data(train_data, ['sqft_living'], 'price')
    
    print("We gonna fit a model based on following features {0}".format(simple_features))

    initial_weights = np.array([-47000., 1.])
    step_size = 7e-12
    tolerance = 2.5e7

    costs,simple_weights = regression_gradient_descent(simple_feature_matrix, output,initial_weights, step_size,tolerance,18000)
    
    print(simple_weights)
    #calculate RSS on test data
    CalculateRSSOnTestData(test_data, simple_features,simple_weights)
    plt.title("Simple Feature Cost Model")
    plt.plot(costs)
    plt.xlabel('Iterations')
    plt.ylabel('RSS - Cost')
    
    # plt.show()


def MultiFeatureModel(train_data, test_data):
    
    model_features = ['sqft_living', 'sqft_living15'] 
    # sqft_living15 is the average squarefeet for the nearest 15 neighbors.

    (feature_matrix, output) = get_numpy_data(train_data, model_features, my_output)
  
    initial_weights = np.array([-100000., 1., 1.])
    step_size = 4e-12
    tolerance = 1e9
    costs,multifet_weights = regression_gradient_descent(feature_matrix, output,initial_weights, 
                                                         step_size,tolerance,10000)
    
    print(multifet_weights)
    
    #calculate RSS on test data
    CalculateRSSOnTestData(test_data, model_features, multifet_weights)
 
    f2= plt.figure()
    plt.title("Multiple Feature Cost Model")
    plt.xlabel('Iterations')
    plt.ylabel('RSS - Cost')
    plt.plot(costs)
    


def CalculateRSSOnTestData(test_data, features, weights):
 
    (feature_matrix, output) = get_numpy_data(test_data, features,my_output)
    
    rssOnTEstData = CalculateCost(predict_output(feature_matrix,weights),output)

    print("RSS on test data is: %e " % rssOnTEstData)


def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance,iterations):
    converged = False 
    weights = np.array(initial_weights) # make sure it's a numpy array
    costs = []
        
    while not converged:
    #for itr in range(iterations):
        predictions = predict_output(feature_matrix, weights)
        cost = CalculateCost(predictions, output)
        costs.append(cost)

        errors = predictions - output
        gradient_sum_squares = 0 
                
        for i in range(len(weights)): # loop over each weight
            drivative = feature_derivative(errors, feature_matrix[:, i])
            gradient_sum_squares += drivative * drivative
            weights[i] -= (step_size * drivative)
      
        gradient_magnitude = sqrt(gradient_sum_squares)
        if gradient_magnitude < tolerance:
            converged = True

    return(costs,weights)


def CalculateCost(prediction, originalObs):
    error = originalObs - prediction
    rss = np.vdot(error,error)
    return (rss / (2 * (len(prediction))))  # squared error function


def get_numpy_data(dataFrame, feature, output):
    dataFrame['constant'] = 1
    return(dataFrame[['constant'] + feature].as_matrix(), 
           dataFrame[output].as_matrix())


def predict_output(feature_matrix, weights):
    return np.dot(feature_matrix, weights) # thats the predictions


def feature_derivative(errors, feature):
    return 2 * np.dot(errors, feature)


run()

