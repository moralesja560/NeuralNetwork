# How to develop a Convolutional Neural Network 

"""
A convolutional Neural Network is a Deep Learning algorithm which can take an input image, assign importance to various aspects 
in the image and be able to differentiate one from the other.

The pre-processing required in a ConvNet is much lower as compared to other classification algorithms, and also, is able to understand and capture the Spacial
and Temporal dependencies in an image through the application of relevant filters. In short, a CNN can be trained to understand a sophisticated image.

An image is nothing more than an array of pixels. In an RGB image, the picture has 3 color planes (red,green,blue) each with its height and width measured in pixels.
An 8k image has a resolution of 7680x4320. That means that each color plane has 33 million pixels. In order to fully process the image, the CNN has to understand 99 million of pixels.

Things can get computationally intensive with a standard Neural Network. The role of the CNN is to reduce the image into a form that is easier to process.

"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import matplotlib.pyplot as plt
from matplotlib.image import imread
import sys

#This function sets the absolute path for the app to access its resources
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

#define location of the dataset
folder = resource_path('resources\\dataset_dogs_vs_cats\\train\\')

##-------Visualize the data to gain insight----------------#

for i in range(9):
	#define subplot
	plt.subplot(330+1+i)
	#define filename
	filename = folder + 'dog.' + str(i) + '.jpg'
	#load images
	image = imread(filename)
	#plot raw pixel data
	plt.imshow(image)
plt.show()

#------------------Create the train and test datasets-----------------------#

from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

#create and select a different folder to store the data.
dataset_home = resource_path('resources\\dataset_dogs_vs_cats2\\')
#create the subdirs
subdirs = ['train\\','test\\']

for subdir in subdirs:
	#create train and test folders
	labeldirs = ['dogs\\', 'cats\\']
	for labeldir in labeldirs:
		newdir = dataset_home + subdir + labeldir
		makedirs(newdir, exist_ok=True)

#seed random number generator
seed(42)
#select the train/test ratio .80/.20
val_ratio = 0.20
src_directory = folder

#for each file in the original dataset folder
for file in listdir(src_directory):
	src= src_directory + file
	dst_dir = 'train\\'
	if random() < val_ratio:
		dst_dir = 'test\\'
	if file.startswith('cat'):
		dst = dataset_home + dst_dir + 'cats\\' + file
	# commented to avoid duplicating when running the code
	#	copyfile(src,dst)
	elif file.startswith('dog'):
		dst = dataset_home + dst_dir + 'dogs\\' + file
	#	copyfile(src,dst)


##---------------CREATE A CONVOLUTIONAL NEURAL NETWORK--------------------#
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def define_model():
	model = Sequential()
	model.add(Conv2D(
			32,(3,3),
			activation='relu',
			kernel_initializer='he_uniform',
			padding='same',
			input_shape=(200,200,3)
			))
	model.add(MaxPooling2D(2,2))
	model.add(Flatten())
	model.add(Dense(128,activation='relu',kernel_initializer='he_uniform'))
	model.add(Dense(1,activation='sigmoid'))
	#compile model
	opt = SGD(learning_rate=0.001, momentum=0.9)
	model.compile(optimizer=opt,loss='binary_crossentropy',metrics=['accuracy'])
	return model

# call the model
model = define_model()

####------------- PREPARE THE DATA----------------#

# data generator
datagen = ImageDataGenerator(rescale=1.0/255.0)
## prepare the iterators
train_it = datagen.flow_from_directory(
	resource_path('resources\\dataset_dogs_vs_cats2\\train\\'),
	class_mode='binary',
	batch_size=64,
	target_size=(200,200))
test_it = datagen.flow_from_directory(
	resource_path('resources\\dataset_dogs_vs_cats2\\test\\'),
	class_mode='binary',
	batch_size=64,
	target_size=(200,200))

# time to fit the model
history = model.fit(
	train_it,
	steps_per_epoch=len((train_it)),
	validation_data=(test_it),
	validation_steps=len((test_it)),
	epochs=20,
	verbose=1)
