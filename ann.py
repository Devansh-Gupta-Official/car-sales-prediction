from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# can also set encoding as latin
dataset = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')   # we have to use the encoding parameter as our data has a lot of special characters like email has @ and . etc.


# can also use X = dataset.drop(['Customer Name','Customer e-mail','Country','Car purchase amount'],axis=1)  here axis =1 signifies that we want to drop only the columns entirely not the rows
X = dataset.iloc[:, 3:-1].values
print(X.shape)


# can also use y=dataset['Car Purchasing Amount']
y = dataset.iloc[:, -1].values
print(y.shape)   # y has 500 rows and 1 column
# y=y.reshape(-1,1)


# PRINTING FIRST FIVE OR LAST FIVE ROWS OF THE DATASET
# print(dataset.head(5))
# print(dataset.tail(5))

# VISUALISING THE DATASET
# print(sns.pairplot(dataset))  #plotting all columns against one another


# # MIN MAX SCALING
# from sklearn.preprocessing import MinMaxScaler
# scaler= MinMaxScaler()
# X_scaled=scaler.fit_transform(X)
# y=y.reshape(-1,1)    # reshaping y as scaling does not work on a single column dataset   
# y_scaled=scaler.fit_transform(y)


# DIVIDING IN TRAINING AND TEST SET
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


# BUILDING THE MODEL
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=25,input_dim = 5, activation='relu'))   # input_dim is input dimension (age, gender, etc...)
ann.add(tf.keras.layers.Dense(units=25, activation='relu'))    # input_dim is default 25 as we are building the model in sequential order.
ann.add(tf.keras.layers.Dense(units=1, activation='linear'))


# COMPILING AND RUNNING THE MODEL
ann.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
ann.fit(X_train, y_train, batch_size=25, epochs=50, verbose=1, validation_split = 0.2)         
# verbose specifies how much info need to be shown during training
# validation split divides the training data again to avoid overfitting


# PRINTING PREDICTED AND ACTUAL VALUES TOGETHER
answer = ann.predict(X_test)
answer.reshape(-1, 1)
print(answer[:, 0])
np.set_printoptions(precision=2)   #to set no. of digits after decimal to 2
print(np.concatenate((answer.reshape(len(answer),1), y_test.reshape(len(y_test),1)),1))


# PREDICTING A SINGLE VALUE
# test = np.array([[1,50,50000,10000,600000]])
test = np.array([[0,44,63000,11500,370000]])
print(ann.predict(test))


# EVALUATING MODEL PERFORMANCE
from sklearn.metrics import r2_score
print(r2_score(y_test,answer))