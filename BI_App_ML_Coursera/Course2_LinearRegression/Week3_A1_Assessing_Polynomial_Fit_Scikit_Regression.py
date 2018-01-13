import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

#os.system("mode con cols=120 lines=70")
os.system("mode con cols=120")

##Here we are going to see the effect of poly degree features and model for each degree
#from 1 to 15
#Finally, we will select a model with a degree, which comes with least RSS. 
    
dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 
              'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 
              'long':float, 'sqft_lot15':float,'sqft_living':float, 'floors':str, 'condition':int,
              'lat':float, 'date':str, 'sqft_basement':int,'yr_built':int, 'id':str, 'sqft_lot':int,
              'view':int}

def run():
   
    sales = pd.read_csv('LR_Data\kc_house_data.csv',dtype=dtype_dict)
    
    print("Loaded the sales data of {0} shape".format(sales.shape))
    print("\n Columns are {0}".format(sales.columns))
    
    VisulaizePolynomialRegression(sales)
    Visulaize15DegPolynomialRegressionOnDifferentSubsetSetOfSameData()
    best_poly_deg=SelectingBestPolynomialDeg()
    
    plt.show()


def VisulaizePolynomialRegression(sales):
 
    sales = sales.sort_values(by=['sqft_living', 'price'])
    
    plt.plot(sales['sqft_living'],sales['price'],'.')

    buildPolynomialDataModel(sales,'sqft_living', 'price',1,'deg' + str(1)) 

    buildPolynomialDataModel(sales,'sqft_living', 'price',2,'deg' + str(2))

    buildPolynomialDataModel(sales,'sqft_living', 'price',3,'deg' + str(3))

    buildPolynomialDataModel(sales,'sqft_living', 'price',15,'deg' + str(15))

    plt.title("1,2,3,15 Deg Polynomial Fit on Same Data set")
    plt.legend()
    plt.xlabel('Area in Square Fit ')
    plt.ylabel('Price')

    #plt.show()
def Visulaize15DegPolynomialRegressionOnDifferentSubsetSetOfSameData():
    
    f2 = plt.figure()
    
    set1 = pd.read_csv('LR_Data\wk3_kc_house_set_1_data.csv',dtype=dtype_dict)
    buildPolynomialDataModel(set1,'sqft_living', 'price',15,'Set1')

    set2 = pd.read_csv('LR_Data\wk3_kc_house_set_2_data.csv',dtype=dtype_dict)
    buildPolynomialDataModel(set2,'sqft_living', 'price',15, 'Set2')

    set3 = pd.read_csv('LR_Data\wk3_kc_house_set_3_data.csv',dtype=dtype_dict)
    buildPolynomialDataModel(set3,'sqft_living', 'price',15, 'Set3')

    set4 = pd.read_csv('LR_Data\wk3_kc_house_set_4_data.csv',dtype=dtype_dict)
    buildPolynomialDataModel(set4,'sqft_living', 'price',15, 'Set4')
      
    plt.title("15 Deg Polynomial of 4 Subsets of Same Data set")
    plt.legend()
    plt.xlabel('Area in Square Fit ')
    plt.ylabel('Price')

    #plt.show()
def buildPolynomialDataModel(data, x_feat, y_feat, deg, plot_label='', showCoffandPlot=True):
    
    polynomial_data = polynomial_sframe(data[x_feat], deg)
   
    my_features = polynomial_data.columns # get the name of the features
    
    model = GetModel(polynomial_data,data[y_feat])
    
    if showCoffandPlot:
        PrintCofficients(model, my_features)
        plt.plot(polynomial_data['power_1'], model.predict(polynomial_data), label=plot_label)
    
    return model

def GetModel(x,y):
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x, y)
    return model

def PrintCofficients(model, feature):
    print("\n\nsqft with {0} weights: {1}".format(feature, model.coef_))
    print("Intercept weights:", model.intercept_)

def polynomial_sframe(data, degree):
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
   
    return rss/(2*len(prediction))


def SelectingBestPolynomialDeg():
    train_data = pd.read_csv('LR_Data\wk3_kc_house_train_data.csv',dtype=dtype_dict)
    print("Loaded the train data of {0} shape".format(train_data.shape))
    
    test_data = pd.read_csv('LR_Data\wk3_kc_house_test_data.csv',dtype=dtype_dict)
    print("Loaded the test data of {0} shape".format(test_data.shape))
    
    validate_data = pd.read_csv('LR_Data\wk3_kc_house_valid_data.csv',dtype=dtype_dict)
    print("Loaded the validation data of {0} shape".format(validate_data.shape))

    all_degmodels =[]
    
    # create model for each deg, from 1 to 15
    for deg in range(1, 15+1):
        model= buildPolynomialDataModel(train_data,'sqft_living', 'price',deg,showCoffandPlot=False)
        all_degmodels.append((deg,model))

    #calculate RSS (Squared error) on training,test and validation data for each model
    train_rss_values=CalculateRSSOnPolynomialDataForAllModels(
        all_degmodels, train_data, 'sqft_living', train_data['price'])    
    test_rss_values=CalculateRSSOnPolynomialDataForAllModels(
        all_degmodels, test_data, 'sqft_living', test_data['price'])
    validation_rss_values=CalculateRSSOnPolynomialDataForAllModels(
        all_degmodels, validate_data,'sqft_living', validate_data['price'])

    #validation_rss_values=list(map(lambda x:x/1E05,  validation_rss_values))

    best_poly_deg= validation_rss_values.index(min(validation_rss_values))+1
    print ("\nDegree of the model with least RSS on validation data is: ", best_poly_deg ,"with value of ", min(validation_rss_values))

    f3 = plt.figure()
    x= np.array([i for i in range(1,15+1)])
    plt.plot(x,train_rss_values,'-',label='Test')
    plt.plot(x,test_rss_values,'-',label='Train')
    #plt.plot(x,validation_rss_values,'-',label='validation')

    plt.title("RSS (Squared error) for training and test data for 1 to 15 Deg Models")
    plt.legend()
    plt.xlabel('Complexity (Degree)')
    plt.ylabel('RSS / (1/2M)')

    PlotValidationRSS(x,validation_rss_values)

    return best_poly_deg

def PlotValidationRSS(x,validation_rss_values):
    f4 = plt.figure()
  
    plt.plot(x,validation_rss_values,'-',label='validation')

    plt.title("RSS (Squared error) for Validation data for 1 to 15 Deg Models")
    plt.legend()
    plt.xlabel('Complexity (Degree)')
    plt.ylabel('RSS / (1/2M)')


def CalculateRSSOnPolynomialDataForAllModels(models,data,x_feat,original_y):
   
    rss_values =[]
    
    for (deg,model) in models:
        # print (deg)
        poly_data = polynomial_sframe(data[x_feat],deg)
        
        rss=CalculateRSS(model, poly_data, original_y)
       
        rss_values.append(rss)

    
    return rss_values

run()

