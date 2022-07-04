from math import sin
import os
from tabnanny import verbose
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import os
import sys
from tensorflow.keras.utils import plot_model
import csv
import matplotlib.pyplot as plt

#This function sets the absolute path for the app to access its resources
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

#step 1: load the entire data into a list
tf.random.set_seed(42)
x_raw_data = []
y_raw_data = []
x_train_data = []
y_train_data = []
x_test_data = []
y_test_data = []
y_pred = []

#step 1.1: load data from CSV
with open(resource_path("resources/data.csv")) as file:
	type(file)
	csvreader = csv.reader(file)
	header = []
	header = next(csvreader)
	header
	rows = []
	for row in csvreader:
		rows.append(row)
#step 1.2: load into lists
for row in rows:
	#print(row[0])
	x_raw_data.append( float(row[0]))
	y_raw_data.append(float(row[1]))

#Step 2: Create the TensorFlow tensors and its related data

#		80% train data / 15% test data
Tf_raw_X = tf.constant(x_raw_data)
Tf_raw_y = tf.constant(y_raw_data)
Tf_train_X = Tf_raw_X[0:80]
Tf_train_y = Tf_raw_y[0:80]
Tf_test_X = Tf_raw_X[80:]
Tf_test_y = Tf_raw_y[80:]


#step 3: visualize the data 
#plt.figure(figsize =(20,7))
#plot training data in blue
#plt.scatter(Tf_train_X,Tf_train_y, c="b", label="Training Data", marker='.')
#plt.plot(Tf_train_X,Tf_train_y)
#plt.scatter(Tf_test_X,Tf_test_y, c="g", label="Testing Data",marker='.')
#plt.plot(Tf_test_X,Tf_test_y)
#plt.legend()
#plt.show()

#step 4: visualize the data 
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(50, activation =None))
model.add(tf.keras.layers.Dense(50, activation =None))
model.add(tf.keras.layers.Dense(1))

# Step 2 .- Compile the model
#we can add as much layers as we want
model.compile(loss = tf.keras.losses.mae,
              optimizer = tf.keras.optimizers.Adam(learning_rate= 0.01),
              metrics = ["mae"]
              )
#fit the model
model.fit(tf.expand_dims(Tf_train_X, axis=-1),Tf_train_y, epochs=400,verbose=0)

#train area
input_value = 10003

correct_value = (input_value+500)

neural_network_out = round(np.double(model.predict(np.array([input_value]))),5)
print(f"valor entrada: {input_value}, \n valor salida correcto {correct_value} \n red neuronal: {neural_network_out} \n diferencia: {round(correct_value - neural_network_out,5)}")


#Step5 Predict data
y_pred = model.predict(Tf_test_X)


#step 6: visualize the data 
plt.figure(figsize =(10,7))
#plot training data in blue
plt.scatter(Tf_train_X,Tf_train_y, c="b", label="Training Data", marker='.')
plt.plot(Tf_train_X,Tf_train_y)
plt.scatter(Tf_test_X,Tf_test_y, c="g", label="Testing Data",marker='.')
plt.plot(Tf_test_X,Tf_test_y)
plt.scatter(Tf_test_X,y_pred, c="r", label="Neural Response",marker='.')
plt.plot(Tf_test_X,Tf_test_y)
plt.legend()
plt.show()