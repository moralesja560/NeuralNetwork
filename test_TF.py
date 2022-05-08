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

#inside the Keras Sequential Model, we add 2 layers
#Dense is to indicate that layers are deeply connected
	#4 nodes
	#input_dim=3 means that the input layer dimension is 3,
	#relu is short of Rectified Linear Unit. Serves as the most basic linear activation function
		#basically it outputs the input directly if it is positive, otherwise it will output zero.
		#it has become the default activation function for many types of neural networks
		#because it is easier to train and often achieves better performance.
		#use bias when model is small. Still recommended using bias.
model.add(Dense(4, input_dim=3, activation='relu',use_bias=True))

#This comment below adds another layer
model.add(Dense(4, activation='relu', use_bias=True))
model.add(Dense(4, activation='relu', use_bias=True))
#another add to the model.
	#Dense
	#1 layer
	#sigmoid function as activation
		#when using sigmoid function with large models, a vanishing gradient problem arises
		#this means that the partial derivative approaches to 0 and it may stop the NN training.
model.add(Dense(1, activation='sigmoid', use_bias=True))
	
	
#Model compilation
	#The Mean Squared Error is the default loss to use for regression problems.
	#ADAM is Adaptive Moment Estimation, it has recently seen broader adoption for deep learning application in vision and natural language processing
	#A metric judges the performance of your model
		#they are similar in many ways to loss functions, except that results are not used to train the model. 
model.compile(loss='mean_squared_error',optimizer='adam',metrics=['binary_accuracy'])


#returns a list of all weight tensors in the model, as Numpy arrays.
	#¿Where does this get the tensors?
print (model.get_weights())
#it's time to group the layers into an object with training and inference features
	#X as input
	#Y as output
	#2000 validation runs
	#validation data to evaluate the loss and metric models
history = model.fit(X, y, epochs=2000,validation_data = (X, y))
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

result = model.predict(X )
print (result)