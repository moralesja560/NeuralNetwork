import tensorflow as tf
import numpy as np
import os
import sys


#This function sets the absolute path for the app to access its resources
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)



#Get the data
#create features (or X independent data)
X = np.array([-7.0,-4.0,-1.0,2.0,5.0,8.0,11.0,14.0])

#Create labels
y = np.array([3.0,6.0,9.0,12.0,15.0,18.0,21.0,24.0])

X = tf.constant(X)
y = tf.constant(y)

#Step 1.- Create a model using the Sequential API
model = tf.keras.Sequential()

model.add(tf.keras.layers.Dense(50, activation =None))
model.add(tf.keras.layers.Dense(50, activation =None))
model.add(tf.keras.layers.Dense(1))

# Step 2 .- Compile the model
#we can add as much layers as we want
model.compile(loss = tf.keras.losses.mae,
              optimizer = tf.keras.optimizers.Adam(learning_rate= 0.02),
              metrics = ["mae"]
              )
#fit the model
model.fit(tf.expand_dims(X, axis=-1),y, epochs=300)

model.save(resource_path(r"savedmodel"))

#predict data
print(model.predict(np.array([183.0])))