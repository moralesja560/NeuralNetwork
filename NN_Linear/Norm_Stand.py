import os
from unittest import result
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def listFrom1toN(n):
    list_from_1_to_n = []
    for x in range(1,n+1):
        list_from_1_to_n.append(x)
    return list_from_1_to_n

################### CHAPTER 64 - NORMALIZATION AND STANDARDIZATION #############################



#step 1: load data from CSV
insurance_data = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")


# What if we wanted to get these in a similar scale?
#Normalization is a technique often applied as part of data preparation for machine learning.
#The goal of that, is to change the values of the numeric columns in the dataset to a common scales
# without distorting differences in the ranges of values.
# Neural networks converge faster when data is normalized.

#create a column transformer.
ct = make_column_transformer(
			(MinMaxScaler(), ["age", "bmi","children"]), #turn all values in these columns between zero and one.
			(OneHotEncoder(handle_unknown="ignore"), ["sex","smoker","region"])
			)

X = insurance_data.drop("charges", axis=1)
y = insurance_data["charges"]


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#Fit the column transformer to our training data
ct.fit(X_train)

#Transform training and test data with normalization (MinMaxScaler) and OneHotEncoder
X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

#What does out data look like now

#print(X_train.loc[0])
#print(X_train_normal[0])

#############################################################

tf.random.set_seed(42)
insurance_model = tf.keras.Sequential()

insurance_model.add(tf.keras.layers.Dense(10))
insurance_model.add(tf.keras.layers.Dense(10))
insurance_model.add(tf.keras.layers.Dense(1))

# Step 2 .- Compile the model
#we can add as much layers as we want
insurance_model.compile(loss = tf.keras.losses.mae,
              optimizer = tf.keras.optimizers.SGD(),
              metrics = ["mae"]
              )
#fit the model
history1 = insurance_model.fit(X_train_normal,y_train, epochs=1000,verbose=0)



#test data set
print(insurance_model.evaluate(X_test_normal,y_test))

############################################################

# Set random seed
tf.random.set_seed(42)

# Add an extra layer and increase number of units
insurance_model_2 = tf.keras.Sequential([
  tf.keras.layers.Dense(200), # 100 units
  tf.keras.layers.Dense(100), # 10 units
  tf.keras.layers.Dense(1) # 1 unit (important for output layer)
])

# Compile the model
insurance_model_2.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(), # Adam works but SGD doesn't 
                          metrics=['mae'])

# Fit the model and save the history (we can plot this)
history2 = insurance_model_2.fit(X_train_normal, y_train, epochs=200, verbose=0)


#test data set
print(insurance_model_2.evaluate(X_test_normal,y_test))

############################################################

tf.random.set_seed(42)
insurance_model_3 = tf.keras.Sequential()

insurance_model_3.add(tf.keras.layers.Dense(100, activation = "relu"))
insurance_model_3.add(tf.keras.layers.Dense(10, activation = "relu"))
insurance_model_3.add(tf.keras.layers.Dense(1))

# Step 2 .- Compile the model
#we can add as much layers as we want
insurance_model_3.compile(loss = tf.keras.losses.mae,
              optimizer = tf.keras.optimizers.Adam(),
              metrics = ["mae"]
              )
#fit the model
history3 = insurance_model_3.fit(X_train_normal,y_train, epochs=1000,verbose=0)



#test data set
print(insurance_model_3.evaluate(X_test_normal,y_test))

############################################################

tf.random.set_seed(42)
insurance_model4 = tf.keras.Sequential()

insurance_model4.add(tf.keras.layers.Dense(150, activation ="relu"))
insurance_model4.add(tf.keras.layers.Dense(150, activation ="relu"))
insurance_model4.add(tf.keras.layers.Dense(150, activation ="relu"))
insurance_model4.add(tf.keras.layers.Dense(125, activation ="relu"))
insurance_model4.add(tf.keras.layers.Dense(1))

# Step 2 .- Compile the model
#we can add as much layers as we want
insurance_model4.compile(loss = tf.keras.losses.mae,
              optimizer = tf.keras.optimizers.Adam(),
              metrics = ["mae"]
              )
#fit the model
history4 = insurance_model4.fit(X_train_normal,y_train, epochs=1000,verbose=0)


print(insurance_model4.evaluate(X_test_normal,y_test))




#step 6: visualize the data 
plt.figure(figsize =(10,7))
#plot training data in blue
plt.scatter(history3.epoch,history3.history["loss"], c="blue", label="Model 3", marker='.')
plt.plot(history3.epoch,history3.history["loss"])
plt.scatter(history2.epoch,history2.history["loss"], c="red", label="Model 2", marker='.')
plt.plot(history2.epoch,history2.history["loss"])
plt.scatter(history1.epoch,history1.history["loss"], c="green", label="Model 1", marker='.')
plt.plot(history1.epoch,history1.history["loss"])
plt.scatter(history4.epoch,history4.history["loss"], c="yellow", label="Model 4", marker='.')
plt.plot(history4.epoch,history4.history["loss"])

plt.legend()
#plt.show()

X_plot = listFrom1toN(len(X_test_normal))
y_pred_1 = insurance_model.predict(X_test_normal)
y_pred_2 = insurance_model_2.predict(X_test_normal)
y_pred_3 = insurance_model_3.predict(X_test_normal)
y_pred_4 = insurance_model4.predict(X_test_normal)

results = pd.DataFrame(list(zip(y_test,y_pred_1,y_pred_2,y_pred_3,y_pred_4)),columns=["Valor Real","Modelo 1","Modelo 2","Modelo 3","Modelo 4"])

results=results.astype(float)

results.sort_values(by=['Valor Real'], inplace=True)

results['X_plot'] = X_plot

results.plot(x='X_plot',y=["Valor Real","Modelo 1","Modelo 2","Modelo 3","Modelo 4"],kind='line')
#results.plot(x='X_plot',y="Valor Real",kind='line',color="gray")
#results.plot(x='X_plot',y="Modelo 1",kind='line',color="blue")
#results.plot(x='X_plot',y="Modelo 2",kind='line',color="red")
#results.plot(x='X_plot',y="Modelo 3",kind='line',color="green")
#results.plot(x='X_plot',y="Modelo 4",kind='line',color="yellow")
plt.show()