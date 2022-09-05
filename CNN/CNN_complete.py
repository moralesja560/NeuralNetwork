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

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,Flatten, Conv2D,MaxPool2D,Activation
from tensorflow.keras import Sequential

tf.random.set_seed(42)
#get all the pixel values between 0 - 1
train_datagen = ImageDataGenerator(rescale=1./255)
valid_datagen = ImageDataGenerator(rescale=1./255)

#set the directory path
train_dir = r'C:\\Users\\moralesja.group\\Documents\\SC_Repo\\NeuralNetwork\\resources\\dataset_pizza_steak\\pizza_steak2\\train'
test_dir = r'C:\\Users\\moralesja.group\\Documents\\SC_Repo\\NeuralNetwork\\resources\\dataset_pizza_steak\\pizza_steak2\\test'

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

#time to build the baseline model

model_base = tf.keras.models.Sequential([
	# First CNN layer.
	# add the input shape to the first CNN layer only
	Conv2D(
		filters=10, # filter is the number of sliding windows going across an input. higher number means more complex alarm.
		kernel_size=2, # the size of the sliding window going across and input 
		strides=(1,1), # the size of the step the sliding window takes.
		padding= 'valid', # if "same", out shape is same as input shape, otherwise image gets compressed
		activation = "relu", 
		input_shape =(224,224,3)),

	# Second Layer
	tf.keras.layers.Conv2D(10,3,activation="relu"),
	# third layer
	tf.keras.layers.Conv2D(10,3,activation="relu"),
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

model_base_df = model_base.fit(train_data,
                      epochs=8,
                      steps_per_epoch=len(train_data), # this is amazing. Train data is an array that already have the features and labels joined.
                      validation_data=valid_data, # 
                      validation_steps=len(valid_data),
                      verbose=1)

model_base.save(r'C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\CNN_savedmodel')