import os
from statistics import mean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier

### download the MNIST dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1,cache=True,as_frame= False)
# let's see the keys of the MNIST dictionary
print(mnist.keys())

# have a look at these arrays
X,y = mnist["data"], mnist["target"]
print(f"shape of X: {X.shape}\nShape of y: {y.shape}")

#shape of X: (70000, 784)
### Shape of X means that there are 70k images and each image has 784 features. In other words, a 28x28 pixels resolution.
#Shape of y: (70000,)
y = y.astype(np.uint8)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_7 = (y_train==7)
y_test_7 = (y_test==7)

sgd_clf = SGDClassifier(random_state=42)

X_train_tf=tf.convert_to_tensor(X_train)

tf.random.set_seed(42)
model_7 = tf.keras.Sequential([
	#tf.keras.layers.Flatten(input_shape=(28,28)),
	tf.keras.layers.Dense(4,activation="tanh"),
	tf.keras.layers.Dense(6,activation="tanh"),
	tf.keras.layers.Dense(1,activation="sigmoid")
])

#2 Compile the model
model_7.compile(
	loss = tf.keras.losses.BinaryCrossentropy(),
	optimizer=tf.keras.optimizers.Adam(),
	metrics=["accuracy"])
 
#3.Fit the model
history_7 = model_7.fit(X_train_tf,y_train_7,epochs=1,verbose=1)

reshaped_item = tf.reshape(X[15],[-1,784])
print(model_7.predict(reshaped_item))


#StandardScaler the X_train data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))




model_8 = tf.keras.Sequential([
	tf.keras.layers.Dense(100,activation="tanh"),
	tf.keras.layers.Dense(100,activation="tanh"),
	tf.keras.layers.Dense(10,activation="tanh"),
	tf.keras.layers.Dense(10,activation="softmax")
])

#2 Compile the model
model_8.compile(
	loss = tf.keras.losses.SparseCategoricalCrossentropy(),
	optimizer=tf.keras.optimizers.Adam(learning_rate=0.00039811),
	metrics=["accuracy"])
 
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4*10**(epoch/20))

#3.Fit the model
history_8 = model_8.fit(X_train_scaled,y_train,epochs=40, verbose=1)




reshaped_item = tf.reshape(X_train_scaled,[-1,784])
tf_y_train_predict_mc = model_8.predict([reshaped_item])

rounded_labels=np.argmax(tf_y_train_predict_mc, axis=1)
y_train_pred_mc = cross_val_predict(sgd_clf,X_train_scaled,y_train,cv=3)


print(f"{confusion_matrix(y_train,y_train_pred_mc)}\n{confusion_matrix(y_train,rounded_labels)}")