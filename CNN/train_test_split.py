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
	* a folder with pictures separated by their filenames. (e.g. pizza001.jpg, pizza002.jpg, steak001.jpg)
"""

#seed random number generator
seed(42)
#select the train/test ratio .80/.20
val_ratio = 0.20
# the source directory points to the entire pool of photos. their filename will decide its fate.
src_directory = r'C:\\Users\\moralesja.group\\Documents\\SC_Repo\\NeuralNetwork\\resources\\dataset_pizza_steak\\pizza_steak\\'
#for each file in the original dataset folder
for file in listdir(src_directory):
	src= src_directory + file
	dst_dir = 'train\\'
	if random() < val_ratio:
		dst_dir = 'test\\'
	if file.startswith('pizza'):  #change the pizza name 
		dst = processed_folder + dst_dir + 'pizza\\' + file
	# commented to avoid duplicating when running the code
		copyfile(src,dst)
	elif file.startswith('steak'):
		dst = processed_folder + dst_dir + 'steak\\' + file
		copyfile(src,dst)