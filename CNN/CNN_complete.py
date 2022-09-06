# from zero to a complete Convolutional Neural Network

"""
A convolutional Neural Network is a Deep Learning algorithm which can take an input image, assign importance to various aspects 
in the image and be able to differentiate one from the other.

The pre-processing required in a ConvNet is much lower as compared to other classification algorithms, and also, is able to understand and capture the Spacial
and Temporal dependencies in an image through the application of relevant filters. In short, a CNN can be trained to understand a sophisticated image.

An image is nothing more than an array of pixels. In an RGB image, the picture has 3 color planes (red,green,blue) each with its height and width measured in pixels.
An 8k image has a resolution of 7680x4320. That means that each color plane has 33 million pixels. In order to fully process the image, the CNN has to understand 99 million of pixels.

Things can get computationally intensive with a standard Neural Network. The role of the CNN is to reduce the image into a form that is easier to process.

In other words, a little window of pixels hovers over the picture looking for patterns to learn. The CNN grabs 32 images at the same time and processes them by using convolutional neurons.
"""

from random import shuffle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,Flatten, Conv2D,MaxPool2D,Activation
from tensorflow.keras import Sequential
import os, sys
import matplotlib.pyplot as plt
import pandas as pd


#This function sets the absolute path for the app to access its resources
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)



#This funcion plots curves

def plot_loss_curves(history):
  val_acc =history.history["val_accuracy"]
  val_loss= history.history["val_loss"]

  train_acc = history.history["accuracy"]
  train_loss = history.history["loss"]
  epochs = range(len(history.history["loss"]))

  plt.plot(epochs,train_loss,label="training loss")
  plt.plot(epochs,val_loss,label="val loss")
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()

  plt.plot(epochs,train_acc,label="training accuracy")
  plt.plot(epochs,val_acc,label="val accuracy")
  plt.title("Accuracy")
  plt.xlabel("epochs")
  plt.legend()
  plt.show()

tf.random.set_seed(42)
#get all the pixel values between 0 - 1
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

#set the directory path WORK COMPUTER
#train_dir = r'C:\\Users\\moralesja.group\\Documents\\SC_Repo\\NeuralNetwork\\resources\\dataset_pizza_steak\\pizza_steak2\\train'
#test_dir = r'C:\\Users\\moralesja.group\\Documents\\SC_Repo\\NeuralNetwork\\resources\\dataset_pizza_steak\\pizza_steak2\\test'

train_dir = r'C:\\Users\\SKU 80093\\Documents\\Python_Scripts\\NeuralNetwork\\resources\\dataset_pizza_steak\\pizza_steak2\\train'
test_dir = r'C:\\Users\\SKU 80093\\Documents\\Python_Scripts\\NeuralNetwork\\resources\\dataset_pizza_steak\\pizza_steak2\\test'


# import data from directories and turn it into batches
train_data = train_datagen.flow_from_directory(
    directory=train_dir,
    batch_size=32,
    target_size=(224,224),
    class_mode = 'binary',
    seed=42
)

valid_data = valid_datagen.flow_from_directory(
    directory=test_dir,
    batch_size=32,
    target_size=(224,224),
    class_mode = 'binary',
    seed=42
)



#####------------ A BASELINE CONVOLUTIONAL NEURAL NETWORK------------------------------#

"""
Filters: Decides how many filters should pass over an input tensor. (eg. sliding windows over an image)
    Typical values: 10,32,64,128
    Higher values lead to more complex models

Kernel Size: Determine the shape of the filters (the sliding windows over the output).
    Typical values: 3,5,7
    Lower values learn smaller features, higher values learn large features

Padding: Pads the target tensor with zeroes to preserve input shape. Or leaves in the target tensor as is,lowering the output shape.
    same or valid
    Use it if we have relevant information we might want the CNN to learn close to the edge .

Strides: The number of steps a filters takes across an image at a time.
    Typical values: 1 or 2
    The sliding window moves 1 pixel during the image "scanning"

"""



model_base = tf.keras.models.Sequential([
	# First CNN layer.
	# add the input shape to the first CNN layer only
	Conv2D(
		filters=10, # filter is the number of sliding windows going across an input. higher number means more complex alarm.
		kernel_size=3, # the size of the sliding window going across and input 
		strides=(1,1), # the size of the step the sliding window takes.
		padding= 'valid', # if "same", out shape is same as input shape, otherwise image gets compressed
		activation = "relu", 
		input_shape =(224,224,3)),

	# Second Layer
	tf.keras.layers.Conv2D(20,3,activation="relu"),
	# third layer
	tf.keras.layers.Conv2D(20,3,activation="relu"),
	# Fourth layer
	tf.keras.layers.Flatten(),
	# output layer
	tf.keras.layers.Dense(1, activation='sigmoid')
	])

model_base.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

#model_base_data = model_base.fit(train_data,
#                      epochs=5,
#                      steps_per_epoch=len(train_data), # this is amazing. Train data is an array that already have the features and labels joined.
#                      validation_data=valid_data, # 
#                      validation_steps=len(valid_data),
#                      verbose=1)


#model_base.save(r'C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\CNN_savedmodel')
#model_base.save(resource_path(r"CNN_savedmodel2"))


#------- Evaluate the model-----------#
#plot_loss_curves(model_base_data)
#model_base_df = pd.DataFrame(model_base_data.history)
#model_base_df.to_csv(resource_path("baseline.csv"))

# When a model's **validation loss** starts to increase, it's likely that the model is **overfitting** the training dataset
"""
Bias: The error rate of the training data. High bias means high error rate

Variance: The delta between the training data error rate and the test data error rate. 

Weight: A neuron is composed of an input, weights and a bias. The neuron receives an input, applies a math operation (weight) and adds a bias to produce an output.
    In other words, weight and bias are learnable parameters within the neuron that are adjusted towards the desired values and the correct ouput.


Overfitting: When the model does not make accurate predictions on testing data. When a model gets trained with so much data, it starts learning from the noise.
    It's a problem where the evaluation of the ML algorithms on training data is different from unseen data. (eg training accuracy: 98%, validation accuracy, 60%)
    
    * Increase training data
    * reduce model complexity
    * Stop the training as soon as the loss starts to increase
    * Ridge regularization, MaxPool2d Regularization and lasso regularization
    * Use dropout for NN to tackle overfitting
    

Underfitting: Refers to a model that can neither performs well on the training data nor generalize to new data. This means that there's not enough clean, 
    training data or the model is not complex enough for the presented data. (eg using a linear ML algorithm to discover non-linear patterns)
    
    * Increase model complexity
    * increase the number of features
    * remove noise from the data
    * increase the epochs

"""


 #-------------------------------chapter 121: Beating the Baseline CNN Model-------------#

model_base2 = tf.keras.models.Sequential([
	# First CNN layer.
	    # add the input shape to the first CNN layer only
	Conv2D(
		filters=16, # filter is the number of sliding windows going across an input. higher number means more complex alarm.
		kernel_size=3, # the size of the sliding window going across and input 
		strides=(1,1), # the size of the step the sliding window takes.
		padding= 'valid', # if "same", out shape is same as input shape, otherwise image gets compressed
		activation = "relu", 
		input_shape =(224,224,3)),

	# First Max Pool Layer: Takes the max value of a group of pixels. Condenses the input into a smaller output.
    # From a grid of 4 pixels, takes the pixel with the max value.
	tf.keras.layers.MaxPool2D(pool_size=1),
    # Second CNN Layer
    tf.keras.layers.Conv2D(16,3,activation="relu"),
	# Second Max Pool Layer
	tf.keras.layers.MaxPool2D(),
    # third CNN
	tf.keras.layers.Conv2D(16,3,activation="relu"),
    # Third Max Pool Layer
    tf.keras.layers.MaxPool2D(),
    # 4th CNN
	tf.keras.layers.Conv2D(16,3,activation="relu"),
    # 4th Max Pool Layer
    tf.keras.layers.MaxPool2D(),
		
    
    
    # Flatten
	tf.keras.layers.Flatten(),
	# output layer
	tf.keras.layers.Dense(1, activation='sigmoid')
	])

model_base2.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

#model_base2_data = model_base2.fit(train_data,
#                      epochs=10,
#                      steps_per_epoch=len(train_data), # this is amazing. Train data is an array that already have the features and labels joined.
#                      validation_data=valid_data, # 
#                      validation_steps=len(valid_data),
#                      verbose=1)

# Epoch 10/10
# 51/51 [==============================] - 5s 92ms/step - loss: 0.1898 - accuracy: 0.9342 - val_loss: 0.5477 - val_accuracy: 0.7947


#------- Evaluate the model-----------#
#model_base2_df = pd.DataFrame(model_base2_data.history)
#model_base2_df.to_csv(resource_path("model2.csv"))
#plot_loss_curves(model_base2_data)
#model_base2.summary()


"""
Time to reduce overfitting by generating augmented data.
Data augmentation is the process of altering our training data, leading it to have more diversity,
and in turn allowing our model to learn more generalizable patterns. The training dataset can be altered by using these parameters

* rescale: to normalize (0-1) the pixel values range.
* zoom_range: zoom in the image
* shear_range: mimic the human eye perspective by distorting the image along an axis
* rotation_range: degree range for random rotations
* width shift range: move the image randomly along the x axis (horizontal)
* height shift range: move the image randomly along the y axis (vertical)
* horizontal flip: inverts the image horizontally
* vertical flip: inverts the image vertically
"""
#get all the pixel values between 0 - 1
train_datagen_aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0.5,
    shear_range=0.5,
    zoom_range=0.5,
    width_shift_range=0.4,
    height_shift_range=0.4,
    horizontal_flip=True,
    vertical_flip=True,
    )
valid_datagen = ImageDataGenerator(rescale=1./255)



# import data from directories and turn it into batches
train_data_aug = train_datagen_aug.flow_from_directory(
    directory=train_dir,
    batch_size=32,
    target_size=(224,224),
    class_mode = 'binary',
    shuffle=True
)

valid_data = valid_datagen.flow_from_directory(
    directory=test_dir,
    batch_size=32,
    target_size=(224,224),
    class_mode = 'binary',

)
"""
# get a sample of an already processed training data batch
images, labels = train_data.next()
#check data and label
import cv2
for i in range(0,31):
    while True:
        cv2.imshow(f"processed_data {labels[i]}",images[i])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
"""

#--------------TRAINING WITH AUGMENTED DATA-----------------#


model_base3 = tf.keras.models.Sequential([
	# First CNN layer.
	    # add the input shape to the first CNN layer only
	Conv2D(
		filters=10, # filter is the number of sliding windows going across an input. higher number means more complex alarm.
		kernel_size=3, # the size of the sliding window going across and input 
		strides=(1,1), # the size of the step the sliding window takes.
		padding= 'valid', # if "same", out shape is same as input shape, otherwise image gets compressed
		activation = "relu", 
		input_shape =(224,224,3)),

	# First Max Pool Layer: Takes the max value of a group of pixels. Condenses the input into a smaller output.
    # From a grid of 4 pixels, takes the pixel with the max value.
	tf.keras.layers.MaxPool2D(pool_size=1),
    # Second CNN Layer
    tf.keras.layers.Conv2D(10,3,activation="relu"),
	# Second Max Pool Layer
	tf.keras.layers.MaxPool2D(),
    # third CNN
	tf.keras.layers.Conv2D(10,3,activation="relu"),
    # Third Max Pool Layer
    tf.keras.layers.MaxPool2D(),
    # 4th CNN
	tf.keras.layers.Conv2D(10,3,activation="relu"),
    # 4th Max Pool Layer
    tf.keras.layers.MaxPool2D(),
		  
    # Flatten
	tf.keras.layers.Flatten(),
	# output layer
	tf.keras.layers.Dense(1, activation='sigmoid')
	])

model_base3.compile(
    loss="binary_crossentropy",
    optimizer=tf.keras.optimizers.Adam(),
    metrics=['accuracy']
)

model_base3_data = model_base3.fit(train_data_aug,
                      epochs=10,
                      steps_per_epoch=len(train_data_aug), 
                      validation_data=valid_data, # 
                      validation_steps=len(valid_data),
                      verbose=1)



#------- Evaluate the model-----------#
plot_loss_curves(model_base3_data)
model_base3.save(resource_path(r"CNN_savedmodel3"))

