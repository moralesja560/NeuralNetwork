import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
import csv

tf.random.set_seed(42)
#This function sets the absolute path for the app to access its resources
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


#step 1: load data from CSV
insurance_data = pd.read_csv (resource_path(r"resources/insurance.csv"),header=0)

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

print(len(X),len(X_test), len(X_train))


#create the model 
insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

#2 Compile the model
insurance_model.compile(loss=tf.keras.losses.mae, optimizer=tf.keras.optimizers.SGD(),metrics = ["mae"])

insurance_model.fit(tf.expand_dims(X_train, axis=-1),y_train, epochs=100)

