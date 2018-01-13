import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import math

os.system("mode con cols=120 lines=70")
import sys

##Here we load the crime rate versus house price data and try to predict the value with
#and without the high levarge point.

def run():
    #sales = pd.read_csv('LR_Data\Philadelphia_Crime_Rate_noNA.csv')
    #print(sales.columns)
    
    print('Example of Basic/Scaler Scaler Sigmoid Function')
    print("Sigmoid of 0 is: {0},\nSigmoid of -100 is: {1}, \nSigmoid of 100 is: {2} \nSigmoid of 1000 is: {3}".
          format(basic_sigmoid(0),basic_sigmoid(-100),basic_sigmoid(100),basic_sigmoid(1000)))


    print('\nExample of Vector Sigmoid Function')
    print("Sigmoid of [0,-100,100,1000] is {0}". format(sigmoid([0,-100,100,1000])))

    print('\nExample of Sigmoid Derivative')
    print("Sigmoid Derivative of input [0,-100,100,1000] is {0}". format(sigmoid_derivative([0,-100,100,1000])))


def basic_sigmoid(z): #only scaler input will work
    s = 1/(1+math.exp(-z))
    return s

def sigmoid(z):  #input is vector or scaler, both will work
    z=np.array(z) # to make sure input is a vector
    s = 1/(1+np.exp(-z))
    return s

#input scaler or vector (numpy array)
def sigmoid_derivative(x):
    s= sigmoid(x)
    ds=s*(1-s)   #note: something multiply with 1-something will always less then the something
    return ds

def image2Vector(image):
     
    """
    Argument:
    image -- a numpy array of shape (length, height, depth)
    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """
    ### START CODE HERE ### (≈ 1 line of code)
    v = image.reshape((image.shape[0]*image.shape[1]* image.shape[2]),1)
    ### END CODE HERE ###
    
    return v

def getImage():
    image1 = np.array([[[ 0.67826139,  0.29380381],
        [ 0.90714982,  0.52835647],
        [ 0.4215251 ,  0.45017551]],

       [[ 0.92814219,  0.96677647],
        [ 0.85304703,  0.52351845],
        [ 0.19981397,  0.27417313]],

       [[ 0.60659855,  0.00533165],
        [ 0.10820313,  0.49978937],
        [ 0.34144279,  0.94630077]]])

    #print ("image2vector(image) = " + str(image2vector(image)))
    return image1

def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).
    
    Argument:
    x -- A numpy matrix of shape (n, m)
    
    Returns:
    x -- The normalized (by row) numpy matrix. You are allowed to modify x.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    # Compute x_norm as the norm 2 of x. Use np.linalg.norm(..., ord = 2, axis = ..., keepdims = True)
    x_norm = np.linalg.norm(x, axis=1, keepdims=True)
    
    # Divide x by its norm.
    x = x / x_norm
    ### END CODE HERE ###

    return x

def softmax(x):
    """Calculates the softmax for each row of the input x.

    Your code should work for a row vector and also for matrices of shape (n, m).

    Argument:
    x -- A numpy matrix of shape (n,m)

    Returns:
    s -- A numpy matrix equal to the softmax of x, of shape (n,m)
    """
    
    ### START CODE HERE ### (≈ 3 lines of code)
    # Apply exp() element-wise to x. Use np.exp(...).
    x_exp = np.exp(x)

    # Create a vector x_sum that sums each row of x_exp. Use np.sum(..., axis = 1, keepdims = True).
    x_sum = np.sum(x_exp, axis =1, keepdims=True)
    
    # Compute softmax(x) by dividing x_exp by x_sum. It should automatically use numpy broadcasting.
    s = x_exp/x_sum

    ### END CODE HERE ###
    
    return s


def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.sum(np.abs(yhat - y))
    ### END CODE HERE ###
    
    return loss


def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    error = yhat-y
    loss = np.dot(error,error)
    ### END CODE HERE ###
    
    return loss

run()