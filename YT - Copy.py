import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv("prices.csv")

#TCS Script data between 02 Mar 2021 and 04 Mar 2022
predict_col = 'Close Price'
predict_out = 7
test_size = 0.3

def prepare_data(df,predict_col,predict_out,test_size):
    label = df[predict_col].shift(-predict_out) #creating new column called label with the last 5 rows are nan
    X = np.array(df[[predict_col]]) #creating the feature array
    X = preprocessing.scale(X) #processing the feature array
    X_predict = X[-predict_out:] #creating the column i want to use later in the predicting method
    X = X[:-predict_out] # X that will contain the training and testing
    label.dropna(inplace=True) #dropping na values
    y = np.array(label)  # assigning Y
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size, random_state=0) #cross validation

    response = [X_train,X_test , Y_train, Y_test , X_predict]
    return response

X_train, X_test, Y_train, Y_test , X_predict =prepare_data(df,predict_col,predict_out,test_size); #calling the method were the cross validation and data preperation is in
learner = LinearRegression() #initializing linear regression model

learner.fit(X_train,Y_train) #training the linear regression model

score=learner.score(X_test,Y_test)#testing the linear regression model
predict= learner.predict(X_predict) #set that will contain the predicted data
response={}#creting json object
response['predicted_value']=predict

print(response)