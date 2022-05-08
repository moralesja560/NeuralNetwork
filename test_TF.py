import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation, Dense
import numpy as np
# X = input of our 3 input XOR gate

# set up the inputs of the neural network (right from the table)
#     X1   X2   X3   Y
#      0    0    0   1
#      0    0    1   0
#      0    1    0   0
#      0    1    1   0
#      1    0    0   0
#      1    0    1   0
#      1    1    0   0
#      1    1    1   1

X = np.array(([0,0,0],[0,0,1],[0,1,0],
[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]), dtype=float)
# y = our output of our neural network

y = np.array(([1], [0], [0], [0], [0],
[0], [0], [1]), dtype=float)

#this groups a linear stack of layers into a tf.keras.Model
model = tf.keras.Sequential()

#inside the Keras Sequential Model, we add 4 layers.
#Dense is to indicate that layers are deeply connected
	#4 layers
	#input_dim=3 means that the input layer dimension is 3,
	#
model.add(Dense(4, input_dim=3, activation='relu',use_bias=True))
#model.add(Dense(4, activation='relu', use_bias=True))
model.add(Dense(1, activation='sigmoid', use_bias=True))
model.compile(loss='mean_squared_error',
optimizer='adam',
metrics=['binary_accuracy'])
print (model.get_weights())
history = model.fit(X, y, epochs=2000,
validation_data = (X, y))
model.summary()
# printing out to file
loss_history = history.history["loss"]
numpy_loss_history = np.array(loss_history)
np.savetxt("loss_history.txt", numpy_loss_history,
delimiter="\n")
binary_accuracy_history = history.history["binary_accuracy"]
numpy_binary_accuracy = np.array(binary_accuracy_history)
np.savetxt("binary_accuracy.txt", numpy_binary_accuracy, delimiter="\n")
print(np.mean(history.history["binary_accuracy"]))
result = model.predict(X ).round()
print (result)