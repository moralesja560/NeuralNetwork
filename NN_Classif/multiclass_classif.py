#Working with a larger example (Multiclass classification)

#When you have more than two classes as an option, it's called Multiclass classification
#to practice multiclass classification, we're going to build a neural network to classify images of different items of clothing

from gc import callbacks
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

#the data has already been sorted into training and test set for us.
(train_data,train_labels),(test_data,test_labels) = fashion_mnist.load_data()

#print(train_data[0], train_labels[0])
print(train_data[0].shape, train_labels[0].shape)

#plt.imshow(train_data[7])


#Create a small list so we can index onto our training labels so they're human readable.
class_names = ["TShirt", "Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle Boot"]

#actually use the class names

def plot_example(index_of_choice):
	plt.imshow(train_data[index_of_choice],cmap=plt.cm.binary)
	plt.title(class_names[train_labels[index_of_choice]])
	plt.show()

#plot_example(17)
#plot_example(10)
#plot_example(100)

#plot some random dataset items
import random

plt.figure(figsize=(14,14))
for i in range(16):
	ax = plt.subplot(4,4,i+1)
	rand_index = random.choice(range(len(train_data)))
	plt.imshow(train_data[rand_index],cmap=plt.cm.binary)
	plt.title(class_names[train_labels[rand_index]])
	plt.axis(False)

plt.show()

##### FOR OUR MULTICLASS CLASSIFICATION MODEL, WE CAN USE A SIMILAR ARCHITECTURE TO OUR BINARY CLASSIFIERS, HOWEVER WE'RE
##### GOING TO HAVE TO TWEAK A FEW THINGS:

#DISCOVER THE SHAPE OF THE DATA
print(train_data[0].shape)
print(train_labels[0].shape)

# How many outputs does the NN need? The answer is in classes
print(len(class_names))

###Results
# Data shape is 28x28 ( the shape or resolution of one image)
# we have 10 classes, so we need an output shape of 10
# Loss function will be tk.keras.losees.CategoricalCrossentropy()
# Output activation will be softmax

##### TIME TO BUILD THE NEURAL NETWORK MODEL

				# A Note about the Flatten layer: NN like to have all the info in one long vector, but the shape of the input image is 28*28.
				# Flatten has converted that shape from (28*28) to (None,784). A long vector filled of data

tf.random.set_seed(42)
model_11 = tf.keras.Sequential([
	tf.keras.layers.Flatten(input_shape=(28,28)),
	tf.keras.layers.Dense(4, activation="relu"),
	tf.keras.layers.Dense(4, activation="relu"),
	tf.keras.layers.Dense(10, activation="softmax"),
	])

model_11.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
					optimizer=tf.keras.optimizers.Adam(),
					metrics=["accuracy"])

non_norm_history = model_11.fit(train_data,train_labels,epochs=5,validation_data=(test_data,test_labels))

#The Categorical Cross Entropy loss function documentation mentions that the labels must be one-hot encoded, 
# or we can use SparseCategoricalCrossEntropy if the labels were integers.
# please use tf.one_hot(test_labels, depth=10) to OHE the data.
# Depth 10 is the maximum value that the resulting tensor will have.

print(model_11.predict(tf.reshape(train_data[0],[-1,28,28])))


#print(tf.one_hot(test_labels, depth=10))
