import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
#os.system("mode con cols=120 lines=70")
os.system("mode con cols=120")

# Here, We are going to see the effect of small and large l2 penalty on housing data set.
# we will see how high variance is reduced by using appropriate L2 penalty value.
# We will use K-Fold technique to generate best validation data set.
def run():
   
    sales = pd.read_csv('LR_Data\kc_house_data.csv',dtype=dtype_dict)
    
    print("Loaded the sales data of {0} shape".format(sales.shape))
    print("\n Columns are {0}".format(sales.columns))
    
    # sort by sqft_living and for identical square footage break the tie by their prices.
    sales = sales.sort_values(by=['sqft_living', 'price']) 
        
    plt.plot(sales['sqft_living'],sales['price'],'.')
    
    plt.title("15 Deg Polynomial Fit on Same Data set")
    plt.legend()
    plt.xlabel('Area in Square Fit ')
    plt.ylabel('Price')

    buildPolynomialDataModel(sales,'sqft_living', 'price',15,'deg' + str(15))


    #Note: When we have so many features and so few data points, the solution can become highly numerically unstable,
    #which can sometimes lead to strange unpredictable results.  Thus, rather than using no regularization, we will
    #introduce a tiny amount of regularization (l2_penalty=1e-5) to make the solution numerically stable.  (In
    #lecture,we discussed the fact that regularization can also help with numerical stability, and here we are seeing a
    #practical example.)
    print('\n\n Lets visualized the curve after applying ridge regression with 1e-5 penality.')
    buildPolynomialDataModel(sales, 'sqft_living', 'price',deg = 15,plot_label = 'Ridge: ' + str(1e-5),ridge = True, l2_penalty = 1e-5)
       
   
    #Recall from Week 3 that the polynomial fit of degree 15 changed wildly whenever the data changed.  In particular,
    #when we split the sales data into four subsets and fit the model of degree 15, the result came out to be very
    #different for each subset.  The model had a *high variance*.  We will see in a moment that ridge regression
    #reduces such variance.  But first, we must reproduce the experiment we did in Week 3.
 
    #This time we will apply little l2_penality to keep model numerically stable
    print('\n\n Lets visualized the curves for 4 different subset of same data')
    Visulaize15DegRegressionOn4SubsetSetOfSameData(1e-9)

    #Generally, whenever we see weights change so much in response to change in data, we believe the variance of our
    #estimate to be large.  Ridge regression aims to address this issue by penalizing "large" weights.  (Weights of
    #model15 looked quite small, but they are not that small because 'sqft_living' input is in the order of thousands.)
    print('\n\n Lets apply large l2_penality = 1.23e2 to reduce the magnitude of cofficients')
    Visulaize15DegRegressionOn4SubsetSetOfSameData(1.23e2)


    print('\n\n Lets perform 10-fold validation to get best lambda value')
    bestLambdaValue = SelectingBestLambdaValue()

    print('\n\n Now train the model on best lambda value: {0}'.format(bestLambdaValue))
   
    f5=plt.figure()
    plt.plot(sales['sqft_living'],sales['price'],'.')
    plt.title("15 Deg Polynomial Fit with Best L2 Penality value found using 10-fold cross validation")
    plt.xlabel('Area in Square Fit ')
    plt.ylabel('Price')

    model = buildPolynomialDataModel(sales, 'sqft_living', 'price',deg = 15,plot_label = 'Ridge: ' + str(bestLambdaValue),ridge = True, l2_penalty = bestLambdaValue)

    plt.show()



def Visulaize15DegRegressionOn4SubsetSetOfSameData(l2_penalty=0):
    
    f2 = plt.figure()

    set1 = pd.read_csv('LR_Data\wk3_kc_house_set_1_data.csv',dtype=dtype_dict)
    set2 = pd.read_csv('LR_Data\wk3_kc_house_set_2_data.csv',dtype=dtype_dict)
    set3 = pd.read_csv('LR_Data\wk3_kc_house_set_3_data.csv',dtype=dtype_dict)
    set4 = pd.read_csv('LR_Data\wk3_kc_house_set_4_data.csv',dtype=dtype_dict)

    plt.plot(set1['sqft_living'],set1['price'],'.')
    plt.plot(set2['sqft_living'],set2['price'],'.')
    plt.plot(set3['sqft_living'],set3['price'],'.')
    plt.plot(set4['sqft_living'],set4['price'],'.')

    buildPolynomialDataModel(set1,'sqft_living', 'price',15,'Set1',ridge=True, l2_penalty=l2_penalty)
    buildPolynomialDataModel(set2,'sqft_living', 'price',15, 'Set2',ridge=True, l2_penalty=l2_penalty)
    buildPolynomialDataModel(set3,'sqft_living', 'price',15, 'Set3',ridge=True, l2_penalty=l2_penalty)
    buildPolynomialDataModel(set4,'sqft_living', 'price',15, 'Set4',ridge=True, l2_penalty=l2_penalty)
      
    plt.title("15 Deg Polynomial of 4 Subsets of Same Data set with l2 Penality = {0}".format(l2_penalty))
    plt.legend()
    plt.xlabel('Area in Square Fit ')
    plt.ylabel('Price')

    #plt.show()
def buildPolynomialDataModel(data,x_feat, y_feat, deg, plot_label='', showCoffandPlot=True, ridge=False, l2_penalty=0):
    
    polynomial_data = polynomial_features(data[x_feat], deg)
   
    my_features = polynomial_data.columns # get the name of the features
    
    model = GetModel(polynomial_data,data[y_feat],ridge, l2_penalty)
    
    if showCoffandPlot:
        PrintCofficients(model, my_features)
        plt.plot(polynomial_data['power_1'], model.predict(polynomial_data), label=plot_label)
        plt.legend(loc='upper left')

    
    return model

def GetModel(x,y,ridge=False,l2_penalty=0):

    if ridge == False:
        model = LinearRegression()
    else:
        model = Ridge(alpha=l2_penalty, normalize=True)

    model.fit(x, y)
    return model


def PrintCofficients(model, feature):
    if hasattr(model, 'alpha') == False:
        print("\n\Model with {0} weights: {1}".format(feature, model.coef_))
    else:
        print("\n\Model with L2Penality: {0}, and with {1} weights: {2}".format(model.alpha,feature, model.coef_))

    print("Intercept weights:", model.intercept_)

def polynomial_features(data, degree):
    # assume that degree >= 1
    poly_frame = pd.DataFrame()
    poly_frame['power_1'] = data

    if degree > 1:
        for power in range(2, degree + 1): 
            poly_frame['power_' + str(power)] = data ** power

    return poly_frame

def CalculateRSS(model, data, originalObs):
    
    prediction = model.predict(data)
    assert(prediction.shape == originalObs.shape)
    error = originalObs - prediction
    rss = np.vdot(error,error)
   
    return rss / (2 * len(prediction))


def SelectingBestLambdaValue():
    
    train_valid_shuffled = pd.read_csv('LR_Data\wk3_kc_house_train_valid_shuffled.csv', dtype=dtype_dict)
    test = pd.read_csv('LR_Data\wk3_kc_house_test_data.csv', dtype=dtype_dict)
    k = 10

    poly15_data = polynomial_features(train_valid_shuffled['sqft_living'], 15)

    features_list = poly15_data.columns# get the name of the features

    poly15_data['price'] = train_valid_shuffled['price'] # add price to the data since it's the target

    cache_avg_validation_errors = []
    cache_l2_penalties = []

    for l2_penalty in np.logspace(1, 7, num=13):
        #[Amit]: need to scale l2penality at factor * 0.0001
        l2_pen= l2_penalty * 0.0001
        avg_val_error = k_fold_cross_validation(k, l2_pen, poly15_data,features_list,'price')
        cache_l2_penalties.append(l2_pen)
        cache_avg_validation_errors.append(avg_val_error)
        print("error @ %s: %f" % (l2_pen, avg_val_error))
   
    min_rss_found = min(cache_avg_validation_errors)
    best_lambda_value =cache_l2_penalties[cache_avg_validation_errors.index(min_rss_found)]

    print("\n Best lambda value {0} with RSS error is {1} ".format(best_lambda_value,min_rss_found ))

    f3 = plt.figure()
    # Using plt.xscale('log')# will make your plot more intuitive.
    plt.xscale('log')
    plt.plot(cache_l2_penalties,cache_avg_validation_errors, 'r.')

    plt.title("RSS for different Lambda values")
    plt.xlabel('Lambdas')
    plt.ylabel('RSS / (1/2M)')

    return best_lambda_value

def k_fold_cross_validation(k, l2_penal, data, features_list,output_label):
    
    n = len(data)
    rss = 0
    
    for i in range(k):
        start = int((n * i) / k)
        end = int((n * (i + 1)) / k - 1)
        
        #print (i, (start, end))
        #print "i: %s, start: %s, end: %s" % (i, start, end)
        
        validation_data = data[start:end + 1]
        training_data = data[0:start]
        training_data = training_data.append(data[end + 1:n])
    
        #print ("split check: len(val): %s, len(training_data): %s => %s, len(data):
        #%s" % (len(validation_data), len(training_data), len(validation_data) + lken(training_data), n))
        #print (" Modeling @ %s" % (i))
        
        model = GetModel(training_data[features_list], training_data[output_label],ridge=True,l2_penalty=l2_penal)
        
        #predictions = model.predict(validation_data[features_list])
        #error = validation_data[output_label] - predictions
        rss += CalculateRSS(model, validation_data[features_list],validation_data[output_label])  #np.vdot(error,error)
        
    return rss / k

def PlotValidationRSS(x,validation_rss_values):
    f4 = plt.figure()
  
    plt.plot(x,validation_rss_values,'-',label='validation')

    plt.title("RSS (Squared error) for Validation data for 1 to 15 Deg Models")
    plt.legend()
    plt.xlabel('Complexity (Degree)')
    plt.ylabel('RSS / (1/2M)')



dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 
              'long':float, 'sqft_lot15':float,'sqft_living':float, 'floors':str, 'condition':int,
              'lat':float, 'date':str, 'sqft_basement':int,'yr_built':int, 'id':str, 'sqft_lot':int,
              'view':int}

run()

