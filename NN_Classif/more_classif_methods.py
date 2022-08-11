#more classification evaluation methods
# Alongside visualizing our models results as much as possible, 
# there are a handful of other classification evaluation methods & metrics you should be familiar with

# * Accuracy - 
# * Precision
# * Recall 
# * F1-Score
# * Confusion Matrix
# * Classification Report from Sci-kit Learn

#Check the accuracy of our model

from gc import callbacks
import os
from cv2 import threshold
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import sys
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

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

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

tf.random.set_seed(42)


model_8 = tf.keras.Sequential([
	#tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
	tf.keras.layers.Dense(4,activation="tanh"),
	tf.keras.layers.Dense(4,activation="tanh"),
	tf.keras.layers.Dense(1,activation="sigmoid")
])

#2 Compile the model
model_8.compile(
	loss = tf.keras.losses.BinaryCrossentropy(),
	optimizer=tf.keras.optimizers.Adam(learning_rate = 0.01),
	metrics=["accuracy"])
 

#3.Fit the model 
history_8 = model_8.fit(X_train,y_train,epochs=100, verbose=0)

plot_decision_boundary(model_8,X_test,y_test)

loss, accuracy = model_8.evaluate(X_test,y_test)

print(f"Model 8 loss on the test set: {np.round(loss,4)}\nModel Accuracy on the test set {accuracy:.2%}")


# HOW ABOUT A CONFUSION MATRIX?

#Make predictions
y_preds = tf.round(model_8.predict(X_test))
#Create a confusion matrix
print(confusion_matrix(y_test,y_preds))

#Oops looks like our prediction array has come out in prediction probability form... the standard output from sigmoid (or softmax activation functions)

# we need to convert the prediction probabilities to binary format and view the first 10 
		#used tf.round above

# This is a confusion matrix
#[[ 96   4]
#[  0 100]]

# Note: The confusion matrix code we've about to write is a remix of sci-kit learn plot confusion matrix

import itertools

def plot_confusion_matrix(test_y,pred_y):
	figsize = (10,10)
	#create the confusion matrix
	cm = confusion_matrix(test_y,tf.round(pred_y))
	cm_norm = cm.astype("float") / cm.sum(axis=1)[:,np.newaxis]
	n_classes = cm.shape[0]
	fig, ax = plt.subplots(figsize=figsize)
	#create a matrix plot 
	cax = ax.matshow(cm, cmap=plt.cm.Blues)
	fig.colorbar(cax)

	#create classes
	classes = False

	if classes:
		labels = classes
	else:
		labels = np.arange(cm.shape[0])

	# label the axes
	ax.set(title = "Confusion Matrix", 
			xlabel = "Predicted Label",
			ylabel = "True Label",
			xticks = np.arange(n_classes),
			yticks = np.arange(n_classes),
			xticklabels = labels,
			yticklabels = labels,
			)

	#more adjustments
	ax.xaxis.set_label_position("bottom")
	ax.xaxis.tick_bottom()
	ax.yaxis.label.set_size(20)
	ax.xaxis.label.set_size(20)
	ax.title.set_size(20)

	# Set threshold for different colors
	threshold = (cm.max() + cm.min())/2

	#plot the text on each cell
	for i,j in itertools.product(range(cm.shape[0]),range(cm.shape[1])):
		plt.text(j,i, f"{cm[i,j]} ({cm_norm[i,j]*100:.1f}%)",
			horizontalalignment = "center",
			color="white" if cm[i,j]>threshold else "black",
			size = 15)

	plt.show()

plot_confusion_matrix(y_test,y_preds)