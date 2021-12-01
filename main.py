import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def main():
    #Importing dataset
    nyc_bike_data = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')
    locations = ['Brooklyn Bridge', 'Manhattan Bridge', 'Williamsburg Bridge', 'Queensboro Bridge']
    means = []
    for loc in locations:
        nyc_bike_data[loc] = nyc_bike_data[loc].str.replace(',', '')
        nyc_bike_data[loc] = nyc_bike_data[loc].astype(int)
        means.append(nyc_bike_data[loc].mean())
    print(locations)
    print(means)

    #Feature and target matrices
    X = nyc_bike_data[['High Temp (°F)', 'Low Temp (°F)', 'Precipitation']]
    y = nyc_bike_data[['Total']]
    y = y['Total'].str.replace(',', '')
    X['Precipitation'] = X['Precipitation'].str.replace('T', '0', regex=True)
    X['Precipitation'] = X['Precipitation'].str.replace('(', '', regex=True)
    X['Precipitation'] = X['Precipitation'].str.replace(')', '', regex=True)
    X['Precipitation'] = X['Precipitation'].str.replace('S', '', regex=True)
    X['Precipitation'] = X['Precipitation'].str.replace(' ', '', regex=True)
    # print(y)

    #Training and testing split, with 25% of the data reserved as the test set
    X = (X.to_numpy()).astype(float)
    # print(X)
    y = (y.to_numpy()).astype(int)
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)

    #Normalizing training and testing data
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    #Define the range of lambda to test
    lmbda = np.logspace(-1, 2, num=51)
    # print(lmbda)

    MODEL = []
    MSE = []
    for l in lmbda:
        #Train the regression model using a regularization parameter of l
        model = train_model(X_train,y_train,l)

        #Evaluate the MSE on the test set
        mse = error(X_test,y_test,model)

        #Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)

    #Plot the MSE as a function of lmbda
    plt.plot(lmbda, MSE)
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Regularization Parameter Lambda")
    plt.title("MSE vs. Lambda")
    plt.savefig("mse_lambda.png")
    plt.close()

    #Find best value of lmbda in terms of MSE
    ind = np.argmin(MSE)
    [lmda_best,MSE_best,model_best] = [lmbda[ind],MSE[ind],MODEL[ind]]

    print('Best lambda tested is ' + str(lmda_best) + ', which yields an MSE of ' + str(MSE_best))

    return model_best

def main_2():
#Importing dataset
    nyc_bike_data = pd.read_csv('NYC_Bicycle_Counts_2016_Corrected.csv')

    #Feature and target matrices
    y = nyc_bike_data[['Precipitation']]
    X = nyc_bike_data[['Total']]
    X = X['Total'].str.replace(',', '')
    y['Precipitation'] = y['Precipitation'].str.replace('T', '0', regex=True)
    y['Precipitation'] = y['Precipitation'].str.replace('(', '', regex=True)
    y['Precipitation'] = y['Precipitation'].str.replace(')', '', regex=True)
    y['Precipitation'] = y['Precipitation'].str.replace('S', '', regex=True)
    y['Precipitation'] = y['Precipitation'].str.replace(' ', '', regex=True)
    # print(y)

    #Training and testing split, with 25% of the data reserved as the test set
    X = (X.to_numpy()).astype(int)
    X = X.reshape(-1, 1)
    # print(X)
    y = (y.to_numpy()).astype(float)
    [X_train, X_test, y_train, y_test] = train_test_split(X, y, test_size=0.25, random_state=101)

    #Normalizing training and testing data
    [X_train, trn_mean, trn_std] = normalize_train(X_train)
    X_test = normalize_test(X_test, trn_mean, trn_std)

    #Define the range of lambda to test
    lmbda = np.logspace(-1, 2, num=51)
    # print(lmbda)

    MODEL = []
    MSE = []
    for l in lmbda:
        #Train the regression model using a regularization parameter of l
        model = train_model(X_train,y_train,l)

        #Evaluate the MSE on the test set
        mse = error(X_test,y_test,model)

        #Store the model and mse in lists for further processing
        MODEL.append(model)
        MSE.append(mse)

    #Plot the MSE as a function of lmbda
    print(lmbda)
    print(MSE)
    plt.plot(lmbda, MSE)
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Regularization Parameter Lambda")
    plt.title("MSE vs. Lambda")
    plt.savefig("mse_lambda_2.png")

    #Find best value of lmbda in terms of MSE
    ind = np.argmin(MSE)
    [lmda_best,MSE_best,model_best] = [lmbda[ind],MSE[ind],MODEL[ind]]

    print('Best lambda tested is ' + str(lmda_best) + ', which yields an MSE of ' + str(MSE_best))

    return model_best


#Function that normalizes features in training set to zero mean and unit variance.
#Input: training data X_train
#Output: the normalized version of the feature matrix: X, the mean of each column in
#training set: trn_mean, the std dev of each column in training set: trn_std.
def normalize_train(X_train):
    mean = np.mean(X_train, axis = 0)
    std = np.std(X_train, axis = 0)
    X = (X_train - mean) / std
    return X, mean, std


#Function that normalizes testing set according to mean and std of training set
#Input: testing data: X_test, mean of each column in training set: trn_mean, standard deviation of each
#column in training set: trn_std
#Output: X, the normalized version of the feature matrix, X_test.
def normalize_test(X_test, trn_mean, trn_std):
    X = (X_test - trn_mean) / trn_std
    return X



#Function that trains a ridge regression model on the input dataset with lambda=l.
#Input: Feature matrix X, target variable vector y, regularization parameter l.
#Output: model, a numpy object containing the trained model.
def train_model(X,y,l):
    model = linear_model.Ridge(alpha = l, fit_intercept = True)
    model.fit(X, y)
    return model


#Function that calculates the mean squared error of the model on the input dataset.
#Input: Feature matrix X, target variable vector y, numpy model object
#Output: mse, the mean squared error
def error(X,y,model):
    yn = model.predict(X)
    mse = np.square(np.subtract(yn, y)).mean()
    return mse

if __name__ == '__main__':
    model_best = main()
    #We use the following functions to obtain the model parameters instead of model_best.get_params()
    print(model_best.coef_)
    print(model_best.intercept_)

    model_best_2 = main_2()
    print(model_best_2.coef_)
    print(model_best_2.intercept_)


# if __name__ == '__main__':
#     datapath = 'NYC_Bicycle_Counts_2016_Corrected.csv'
#     degrees = [1, 2, 3, 5, 6, 7 ,8, 9, 10]

#     df = pd.read_csv(datapath)
#     print(df.keys())
#     x = df[[df.keys()[2], df.keys()[3], df.keys()[4]]]
#     print(x)
#     x = x.to_numpy()
#     # print(x)
#     # paramFits = main(datapath, degrees)
#     # print(paramFits)