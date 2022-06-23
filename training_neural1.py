import tensorflow as tf
import numpy as np

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
model.add(tf.keras.layers.Dense(1))

# Step 2 .- Compile the model
#we can add as much layers as we want
model.compile(loss = tf.keras.losses.mae,
              optimizer = tf.keras.optimizers.Adam(learning_rate= 0.01),
              metrics = ["mae"]
              )
#fit the model
model.fit(tf.expand_dims(X, axis=-1),y, epochs=100)

model.save(r'C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\savedmodel')

#predict data
print(model.predict(np.array([17.0])))