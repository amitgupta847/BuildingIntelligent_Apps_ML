import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
from Week2_A1_Python_Sigmoid import sigmoid
import os
os.system("mode con cols=120 lines=150")
import sys

#Here we  implement the logistic regression in term of Neural network mode.

def run():
        
    print('Loading Image Dataset')
    
    train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

    print('\nTraining Dataset Loaded: Train_X data set shape is: {0}, Train_Y Dataset Shape is: {1}'.
          format(train_set_x_orig.shape, train_set_y.shape))
    print('\nTest Dataset Loaded: Test_X data set shape {0}, Test_Y Dataset Shape{1}'.
          format(test_set_x_orig.shape, test_set_y.shape))

    print('Show a picture example from the dataset')
    ShowExampleImage(train_set_x_orig,train_set_y,classes,31)
    
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
 
   # A trick when you want to flatten a matrix X of shape (a,b,c,d) to a matrix X_flatten of shape (b ∗∗ c ∗∗ d, a) is
   # to use: X_flatten = X.reshape(X.shape[0], -1).T # X.T is the transpose of X
   # reshape image data into a single vector of 64*64*3 = 12888 rows and 209 columns

    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
    test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
    print('\Image data reshaped: Train X data set shape {0}, Test X Dataset Shape{1}'.
          format(train_set_x_flatten.shape, test_set_x_flatten.shape))

    #To represent color images, the red, green and blue channels (RGB) must be specified for each pixel, and so the
    #pixel value is actually a vector of three numbers ranging from 0 to 255.

    # One common preprocessing step in machine learning is to center and standardize your dataset, meaning that you
    # substract the mean of the whole numpy array from each example, and then divide each example by the standard
    # deviation of the whole numpy array.  But for picture datasets, it is simpler and more convenient and works almost

    print('\nDataset normalized by dividing by 255') 
    train_set_x = train_set_x_flatten/255.
    test_set_x = test_set_x_flatten/255.

    print('\nTraining model on 2000 iterations and 0.005 Learning rate..\n') 
    d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
    
    print('\nLets make the pridiction on the mode for image index 1')
    index = 1
    f1=plt.figure()
    plt.title('Image example for prediction')
    plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
    #print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[d["Y_prediction_test"][0,index]].decode("utf-8") +  "\" picture.")
    AnalyzeResults(d)

    FurtherAnalysis(train_set_x, train_set_y, test_set_x, test_set_y)

    plt.show()


def AnalyzeResults(model_results):
    costs = np.squeeze(model_results['costs'])
    f1=plt.figure()
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(model_results["learning_rate"]))
 
def FurtherAnalysis(train_set_x, train_set_y, test_set_x, test_set_y):
    
    learning_rates = [0.01, 0.001, 0.0001]
    models = {}
    
    F4=plt.figure()

    for i in learning_rates:
        print ("\nLearning rate is: " + str(i))
        models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 15000,
                               learning_rate = i, print_cost = False)
        print ('\n' + "-------------------------------------------------------" + '\n')

    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))
    
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.legend(loc='upper center')
    plt.title('Plot for multiple models with different learning rates')
    #frame = legend.get_frame()
    #frame.set_facecolor('0.90')

def model(X_train, Y_train, X_test, Y_test, num_iterations = 2000, learning_rate = 0.5, print_cost = False):
    """
    Builds the logistic regression model by calling the function you've implemented previously
    
    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d -- dictionary containing information about the model.
    """
    
    ### START CODE HERE ###
    
    # initialize parameters with zeros (≈ 1 line of code)
    w, b = Initialize_Params(X_train.shape[0])

    # Gradient descent (≈ 1 line of code)
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    
    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]
    
    # Predict test/train set examples (≈ 2 lines of code)
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    ### END CODE HERE ###

    # Print train Errors
    print("\ntrain accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print('\nPrint Train Data Confusion Matrix')
    PrintConfusionMatrix(Y_prediction_train,Y_train)
    
    # Print test Errors
    print("\ntest accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    print('\nPrint Test Data Confusion Matrix')
    PrintConfusionMatrix(Y_prediction_test, Y_test)

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
    return d

def PrintConfusionMatrix(Y_Pred, Y_Orig):
    
    TP,TN= CalculateTP_TN(Y_Pred,Y_Orig)
   
    print("Out of total {0} cats, model predicted {1} cats successfully.  (True Positive).".
          format(len(Y_Orig[Y_Orig==1]),TP))
    falseNegative= len(Y_Pred[Y_Pred!=1]) - TN
    print("False Negative: {0}".format(falseNegative))
    
    print("Out of total {0} non cats, model predicted {1} non cats successfully. (True Negative).".
          format(len(Y_Orig[Y_Orig!=1]),TN ))
    falsePositive= len(Y_Pred[Y_Pred==1]) - TP
    print("False Positive: {0}".format(falsePositive))

    classification_error=  (falseNegative+falsePositive)/ (TP+TN+ falseNegative +falsePositive)
    print('\nClassificatoin Error is: {0}'.format(classification_error))

    print('\nFinally, Accuracy is (1-Classificatoin Error): {0}'.format(1-classification_error))


def CalculateTP_TN(prediction,original):
    truepositive=0
    truenegative=0
    
    for i in range(original.shape[1]):
        if(original[0][i]==1):
            if(original[0][i] == prediction[0][i]):
                truepositive+=1
        else:
            if(original[0][i] == prediction[0][i]):
                truenegative+=1

    return truepositive, truenegative



def Initialize_Params(dim):
    w=np.zeros(shape=(dim,1))
    b=0

    assert(w.shape==(dim,1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w,b

def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    
    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """
    
    m = X.shape[1]
    
    # FORWARD PROPAGATION (FROM X TO COST)
    ### START CODE HERE ### (≈ 2 lines of code)
    A = sigmoid(np.dot(w.T, X) + b)             # compute activation
    cost = (- 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * (np.log(1 - A)))                                 # compute cost
    ### END CODE HERE ###
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    ### START CODE HERE ### (≈ 2 lines of code)
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)
    ### END CODE HERE ###

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    
    return grads, cost


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    
    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """
    
    costs = []
    
    for i in range(num_iterations):
        
        # Cost and gradient calculation (≈ 1-4 lines of code)
        ### START CODE HERE ### 
        grads, cost = propagate(w, b, X, Y)
        ### END CODE HERE ###
        
        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]
        
        # update rule (≈ 2 lines of code)
        ### START CODE HERE ###
        w = w - learning_rate * dw  # need to broadcast
        b = b - learning_rate * db
        ### END CODE HERE ###
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        
        # Print the cost every 100 training examples
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    ### START CODE HERE ### (≈ 1 line of code)
    A = sigmoid(np.dot(w.T, X) + b)
    ### END CODE HERE ###
    
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        ### START CODE HERE ### (≈ 4 lines of code)
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0
        ### END CODE HERE ###
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction


def ShowExampleImage(data_X,data_Y,classes, index):
    f2=plt.figure()
    plt.title('Example picture from the dataset:')
    plt.imshow(data_X[index])
    print("y = " + str(data_Y[:, index]) + ", it's a '" + classes[np.squeeze(data_Y[:, index])].decode("utf-8") + "' picture.")


run()