import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys

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
history4 = insurance_model4.fit(X_train,y_train, epochs=1000,verbose=1)

#test data set
print(insurance_model4.evaluate(X_test,y_test))
