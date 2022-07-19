#to find the ideal learning rate (the learning rate where the loss decreases the most during training) we're going to use
#the following steps
# * Learning Rate Callback
	#you can think as a callback as an extra piece of functionality you can add to your model while training



from gc import callbacks
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import numpy as np
from tensorflow import keras

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
#print(X)
#check labels
#print(y)

circles = pd.DataFrame({"X0":X[:,0],"X1":X[:,1],"label":y})

tf.random.set_seed(42)

		##### THE DEFINITIVE NON- LINEAR NEURAL NETWORK ##################


model_7 = tf.keras.Sequential([
	#tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
	tf.keras.layers.Dense(4,activation="relu"),
	tf.keras.layers.Dense(4,activation="relu"),
	tf.keras.layers.Dense(1,activation="sigmoid")
])

#2 Compile the model
model_7.compile(
	loss = tf.keras.losses.BinaryCrossentropy(),
	optimizer=tf.keras.optimizers.Adam(),
	metrics=["accuracy"])
 
#2.5 Create a learning rate callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4*10**(epoch/20))

#3.Fit the model
history_7 = model_7.fit(X,y,epochs=100,callbacks=[lr_scheduler], verbose=0)

print(f"Evaluación de modelo 7: {model_7.evaluate(X,y)}")

history_7_df = pd.DataFrame(history_7.history)

#model_7.save(resource_path(r"save_model_Non_Linear"))
#plot_decision_boundary(model_7,X,y)

temp_model_7_df = history_7_df.drop("accuracy",axis=1)
temp_model_7_df.plot()
#plt.title("Model 7 loss curves")
#plt.show()

#let's plot the learning rate agains the loss
lrs = 1e-4 * (10 ** (tf.range(100)/20))
plt.figure(figsize=(10,7))
plt.semilogx(lrs, history_7.history["loss"])
plt.xlabel("learning Rate")
plt.ylabel("Loss")
plt.title("Learning Rate vs Loss")
#plt.show()


#Let's try using a higher ideal learning rate with the same model as before

model_8 = tf.keras.Sequential([
	#tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
	tf.keras.layers.Dense(4,activation="relu"),
	tf.keras.layers.Dense(4,activation="relu"),
	tf.keras.layers.Dense(1,activation="sigmoid")
])

#2 Compile the model
model_8.compile(
	loss = tf.keras.losses.BinaryCrossentropy(),
	optimizer=tf.keras.optimizers.Adam(learning_rate = 0.009),
	metrics=["accuracy"])
 

#3.Fit the model
history_8 = model_8.fit(X,y,epochs=15, verbose=2)

print(f"Evaluación de modelo 8: {model_8.evaluate(X,y)}")

history_8_df = pd.DataFrame(history_8.history)

#model_87.save(resource_path(r"save_model_Non_Linear"))
#plot_decision_boundary(model_7,X,y)

temp_model_8_df = history_8_df.drop("accuracy",axis=1)
temp_model_8_df.plot()
plt.show()