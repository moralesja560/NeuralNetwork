#introduction to NN classification with TensorFlow

#In this notebook we're going to learn how to write neural networks for classification problems.

#A classification problem is where you try to classify something as one thing or another

# binary / multiclass / multilabel

## Creating data to view and fit
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import numpy as np

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def plot_decision_boundary(model, X, y):
	"""
	Plots the decision boundary created by a model predicting on X.
	This function has been adapted from two phenomenal resources:
	 1. CS231n - https://cs231n.github.io/neural-networks-case-study/
	 2. Made with ML basics - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
	"""
	# Define the axis boundaries of the plot and create a meshgrid
	x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
	y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
	xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),np.linspace(y_min, y_max, 100))

	# Create X values (we're going to predict on all of these)
	x_in = np.c_[xx.ravel(), yy.ravel()] # stack 2D arrays together: https://numpy.org/devdocs/reference/generated/numpy.c_.html

	# Make predictions using the trained model
	print(x_in.shape)
	print(x_in)
	y_pred = model.predict(x_in)

	# Check for multi-class
	if model.output_shape[-1] > 1: # checks the final dimension of the model's output shape, if this is > (greater than) 1, it's multi-class 
		print("doing multiclass classification...")
	# We have to reshape our predictions to get them ready for plotting
		y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
	else:
		print("doing binary classifcation...")
		y_pred = np.round(np.max(y_pred, axis=1)).reshape(xx.shape)

	# Plot decision boundary
	plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
	plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.show()
	







#make 1000 examples
n_samples = 1000
#create circles
X,y = make_circles(n_samples,noise=0.03,random_state=42)

#check out features
print(X)
#check labels
print(y)

circles = pd.DataFrame({"X0":X[:,0],"X1":X[:,1],"label":y})

circles.to_csv('circulos.csv')

print(circles)

#plot this
plt.scatter(circles["X0"],circles["X1"],c=y,cmap=plt.cm.RdYlBu)
#plt.show()

#inspect our data
#print(X.shape,y.shape)

# The shape of X is (1000, 2)	because it has 2 columns
# (1000,) is the shape of y because y only contains labels

### BUILD A TENSORFLOW MODEL

tf.random.set_seed(42)

#1.- Create model using the sequential API
model_1 = tf.keras.Sequential([
	tf.keras.layers.Dense(1)
])

#2 Compile the model
model_1.compile(
	loss = tf.keras.losses.BinaryCrossentropy(),
	optimizer=tf.keras.optimizers.SGD(),
	metrics=["accuracy"])

#3.Fit the model
#****#   model_1.fit(tf.expand_dims(X,axis=-1),y,epochs=100,verbose=0)
print(f"Evaluación de modelo 1: {model_1.evaluate(X,y)}")

#Since we're working on a binary classification problem and our model is getting around 50% accuracy
#it's performing as if it's guessing
# Let's add an extra layer

#1.- Create model using the sequential API
model_2 = tf.keras.Sequential([
	tf.keras.layers.Dense(1),
	tf.keras.layers.Dense(1)
])

#2 Compile the model
model_2.compile(
	loss = tf.keras.losses.BinaryCrossentropy(),
	optimizer=tf.keras.optimizers.SGD(),
	metrics=["accuracy"])
 
#3.Fit the model
#****#  model_2.fit(tf.expand_dims(X,axis=-1),y,epochs=100,verbose=0)
print(f"Evaluación de modelo 2: {model_2.evaluate(X,y)}")


#### MODEL 2 WITH AN XTRA LAYER DID NOT WORK 
	# Let's look into our bag of tricks to see how we can improve our model.
	# Create a model	by adding layers or adding hidden units
	# Compile a model - by switching to another optimization algorithm
	# fit a model - by letting the model train for longer.


########## Let's try model 3

model_3 = tf.keras.Sequential([
	tf.keras.layers.Dense(100),
	tf.keras.layers.Dense(10),
	tf.keras.layers.Dense(1)
])

#2 Compile the model
model_3.compile(
	loss = tf.keras.losses.BinaryCrossentropy(),
	optimizer=tf.keras.optimizers.Adam(),
	metrics=["accuracy"])
 
#3.Fit the model
 #****# model_3.fit(tf.expand_dims(X,axis=-1),y,epochs=100,verbose=0)
print(f"Evaluación de modelo 3: {model_3.evaluate(X,y)}")

#To visualize our model's predictions, let's create a function

#this function will:
	#take in a trained model, features and labels
	#create a meshgrid of the different X values
	#make predictions across the meshgrid
	#plot the predictions as well as a line between zones (where each unique class falls)


#check out the prediction the model is making
#plot_decision_boundary(model_3,X,y)

########## Let's try model 4

model_4 = tf.keras.Sequential([
	#tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
	tf.keras.layers.Dense(1,activation="tanh")
])

#2 Compile the model
model_4.compile(
	loss = tf.keras.losses.BinaryCrossentropy(),
	optimizer=tf.keras.optimizers.Adam(learning_rate =0.001),
	metrics=["accuracy"])
 
#3.Fit the model
#****#  history_4 = model_4.fit(tf.expand_dims(X,axis=-1),y,epochs=100,verbose=0)
print(f"Evaluación de modelo 4: {model_4.evaluate(X,y)}")

#plot_decision_boundary(model_4,X,y)


#### BUILDING OUR FIRST NEURAL NETWORK WITH NON LINEARITY

model_5 = tf.keras.Sequential([
	#tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
	tf.keras.layers.Dense(1,activation="relu"),
	tf.keras.layers.Dense(1)
])

#2 Compile the model
model_5.compile(
	loss = tf.keras.losses.BinaryCrossentropy(),
	optimizer=tf.keras.optimizers.Adam(),
	metrics=["accuracy"])
 
#3.Fit the model
history_5 = model_5.fit(tf.expand_dims(X,axis=-1),y,epochs=100,verbose=0)
print(f"Evaluación de modelo 5: {model_5.evaluate(X,y)}")

#plot_decision_boundary(model_5,X,y)


############## MODEL 6

model_6 = tf.keras.Sequential([
	#tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
	tf.keras.layers.Dense(4,activation="relu"),
	tf.keras.layers.Dense(4,activation="relu"),
	tf.keras.layers.Dense(1)
])

#2 Compile the model
model_6.compile(
	loss = tf.keras.losses.BinaryCrossentropy(),
	optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
	metrics=["accuracy"])
 
#3.Fit the model
history_6 = model_6.fit(X,y,epochs=250,verbose=0)
print(f"Evaluación de modelo 6: {model_6.evaluate(X,y)}")

#plot_decision_boundary(model_6,X,y)


			##### THE DEFINITIVE NON- LINEAR NEURAL NETWORK ##################


model_7 = tf.keras.Sequential([
	#tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
	tf.keras.layers.Dense(4,activation="tanh"),
	tf.keras.layers.Dense(4,activation="tanh"),
	tf.keras.layers.Dense(1,activation="sigmoid")
])

#2 Compile the model
model_7.compile(
	loss = tf.keras.losses.BinaryCrossentropy(),
	optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
	metrics=["accuracy"])
 
#3.Fit the model
history_7 = model_7.fit(X,y,epochs=50,verbose=0)
print(f"Evaluación de modelo 7: {model_7.evaluate(X,y)}")

model_7.save(resource_path(r"save_model_Non_Linear"))
#plot_decision_boundary(model_7,X,y)


#plot the loss or training curves

history_7_df = pd.DataFrame(history_7.history)

history_7_df.plot()
plt.title("Model 7 loss curves")
plt.show()



