from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# can also set encoding as latin
dataset = pd.read_csv('Car_Purchasing_Data.csv', encoding='ISO-8859-1')


# can also use X = dataset.drop(['Customer Name','Customer e-mail','Country','Car purchase amount'],axis=1)  here axis =1 signifies that we want to drop only the columns entirely not the rows
X = dataset.iloc[:, 3:-1].values
print(X.shape)


# can also use y=dataset['Car Purchasing Amount']
y = dataset.iloc[:, -1].values
print(y.shape)
# y=y.reshape(-1,1)


# PRINTING FIRST FIVE OR LAST FIVE ROWS OF THE DATASET
# print(dataset.head(5))
# print(dataset.tail(5))


# VISULAISING THE DATASET
# print(sns.pairplot(dataset))


# FEATURE SCALING
sc = StandardScaler()
X_train = sc.fit_transform(X)
y = sc.fit_transform(y)

# DIVIDING IN TRAINING AND TEST SET
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=25, activation='relu'))
ann.add(tf.keras.layers.Dense(units=25, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='linear'))

ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

ann.fit(X_train, y_train, batch_size=32, epochs=100)

answer = ann.predict(X_test)
answer.reshape(-1, 1)
print(answer[:, 0])
