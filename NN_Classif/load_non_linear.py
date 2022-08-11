import tensorflow as tf
import numpy as np
import os,sys 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

def plot_decision_boundary(model, X, y,x_pred2_0,y_pred2_0,x_pred2_1,y_pred2_1):
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
	plt.scatter(x_pred2_0,y_pred2_0,c="blue")
	plt.scatter(x_pred2_1,y_pred2_1,c="red")

	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.show()

def predict_data(x,y):
	result = float(new_model.predict([[x,y]]))
	if result > 0.5:
		inner_dicc["X_pred"].append(x)
		inner_dicc["Y_pred"].append(y)
	else:
		outer_dicc["X_pred"].append(x)
		outer_dicc["Y_pred"].append(y)


#STEP1: Prepare the train data
	#make 1000 examples
n_samples = 1000
	#create circles
X,y = make_circles(n_samples,noise=0.03,random_state=42)


#STEP2: Load the neural model
print(resource_path(r"save_model_Non_Linear"))
new_model = tf.keras.models.load_model(resource_path(r"save_model_Non_Linear"))
	# Check its architecture
new_model.summary()

#STEP3: Load the real world data 
df = pd.read_csv(r'c:\Users\SKU 80093\Documents\Python_Scripts\NeuralNetwork\resources\data_model7.csv')


#Step 4: Prepare the dictionaries to store the model output data.
inner_dicc = {"X_pred":[],"Y_pred":[]}
outer_dicc = {"X_pred":[],"Y_pred":[]}


#STEP5: use a pandas dataframe to feed real world data to a neural network
result = [predict_data(x,y) for x, y in zip(df['Coordenada X'], df['Coordenada Y'])]


#STEP6 convert the two dictionaries (0 and 1 outputs from NN) into dataframes
inner_df = pd.DataFrame(inner_dicc)
outer_df = pd.DataFrame(outer_dicc)
plot_decision_boundary(new_model,X,y,inner_df['X_pred'], inner_df['Y_pred'],outer_df['X_pred'], outer_df['Y_pred'])


