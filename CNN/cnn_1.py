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

#------------------Step 1: Get the data-----------------------#
"""
data structure:

main_folder > class1 folder, class2 folder , etc.

In this example we have "pizza_steak" as main_folder 
and pizza and steak as class folders

"""
#please put here the path of the NEW empty folder where the split datasets will live.
processed_folder = r'C:\\Users\\moralesja.group\\Documents\\SC_Repo\\NeuralNetwork\\resources\\dataset_pizza_steak\\pizza_steak2\\'



#------------------Step 2: Create the train and test datasets-----------------------#
from os import makedirs
from os import listdir
from shutil import copyfile
from random import seed
from random import random

# as always, only train and test are necessary
subdirs = ['train\\','test\\']

for subdir in subdirs:
	#create train and test folders for each class
	labeldirs = ['pizza\\', 'steak\\']
	for labeldir in labeldirs:
		newdir = processed_folder + subdir + labeldir
		makedirs(newdir, exist_ok=True)

# by the end of this step, we will have the folder structure

#-------------Step2: End---------------------#

#-------------Step3: Split main datasets------------------------------#

"""
At this point we already have:
	* the empty folders where the train and test subsets will be stored in.
	* a folder with pictures separated by its filename. (e.g. pizza001.jpg, pizza002.jpg, steak001.jpg)
"""


#seed random number generator
seed(42)
#select the train/test ratio .80/.20
val_ratio = 0.20
# the source directory points to the entire pool of photos. The filename will decide its fate.
src_directory = r'C:\\Users\\moralesja.group\\Documents\\SC_Repo\\NeuralNetwork\\resources\\dataset_pizza_steak\\pizza_steak\\'
#for each file in the original dataset folder
for file in listdir(src_directory):
	src= src_directory + file
	dst_dir = 'train\\'
	if random() < val_ratio:
		dst_dir = 'test\\'
	if file.startswith('pizza'):
		dst = processed_folder + dst_dir + 'pizza\\' + file
	# commented to avoid duplicating when running the code
		copyfile(src,dst)
	elif file.startswith('steak'):
		dst = processed_folder + dst_dir + 'steak\\' + file
		copyfile(src,dst)