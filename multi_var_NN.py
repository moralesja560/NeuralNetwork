import os
from unittest import result
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys


#This function sets the absolute path for the app to access its resources
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def listFrom1toN(n):
    list_from_1_to_n = []
    for x in range(1,n+1):
        list_from_1_to_n.append(x)
    return list_from_1_to_n


def plot_call(X_train,y_train,X_test,y_test,y_pred):
	#step 6: visualize the data 
	plt.figure(figsize =(10,7))
	#plot training data in blue
	plt.scatter(X_train,y_train, c="b", label="Training Data", marker='.')
	plt.plot(X_train,y_train)
	plt.scatter(X_test,y_test, c="g", label="Testing Data",marker='.')
	plt.plot(X_test,y_test)
	plt.scatter(X_test,y_pred, c="r", label="Neural Response",marker='.')
	plt.plot(X_test,y_pred)
	plt.legend()
	plt.show()


#step 1: load data from CSV
insurance_data = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

#refer to a column by using insurance_data.loc[:"age"] or  insurance_data.iloc[:,4]
#refer to independent variable as the features and depend vars as outcome or labels.

#One hot encoding serves the purpose of converting a string column into a all-number column
#suppose there is column with binary values, such as gender.
#OHE divides the info into two columns, one for female and one for male.
# when the original column says "male", a 1 will appear in the male column and a 0 will be on the female column.

###### STEP2 one hot encode the info
insurance_one_hot = pd.get_dummies(insurance_data)


#### STEP3 create the data substest to perform NN training and testing
#drop the charges column to create the X dataset
X = insurance_one_hot.drop("charges", axis=1)
#retrieve only the charges colum to create the y labels raw dataset
y = insurance_one_hot["charges"]

# ¿What is scikit learn? Scikit learn is a Machine Learning library that provides classification, regression and group analysis algorithms.

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)




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
history1 = insurance_model.fit(X_train,y_train, epochs=1000,verbose=0)



#test data set
print(insurance_model.evaluate(X_test,y_test))

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
history2 = insurance_model_2.fit(X_train, y_train, epochs=200, verbose=0)


#test data set
print(insurance_model_2.evaluate(X_test,y_test))

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
history3 = insurance_model_3.fit(X_train,y_train, epochs=1000,verbose=0)



#test data set
print(insurance_model_3.evaluate(X_test,y_test))

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
history4 = insurance_model4.fit(X_train,y_train, epochs=1000,verbose=0)
print(insurance_model4.evaluate(X_test,y_test))

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

X_plot = listFrom1toN(len(X_test))
y_pred_1 = insurance_model.predict(X_test)
y_pred_2 = insurance_model_2.predict(X_test)
y_pred_3 = insurance_model_3.predict(X_test)
y_pred_4 = insurance_model4.predict(X_test)

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