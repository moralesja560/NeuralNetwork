import tensorflow as tf
import numpy as np
import sys,os
import pandas as pd

#This function sets the absolute path for the app to access its resources
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)



new_model = tf.keras.models.load_model(resource_path(r"CNN_savedmodel3"))

# Check its architecture
#print(new_model.summary())


#custom function to help process the data
def load_and_prep_image(filename,img_shape=224):
	"""
	Reads an image from filename, turns it into a tensor and reshapes it to the selected shape (eg 224)
	"""
	#read in the image
	img = tf.io.read_file(filename)
	#decode the image into a tensor
	img = tf.image.decode_image(img)
	#resize the image
	img = tf.image.resize(img, size=[img_shape,img_shape])
	#rescale the image
	img = img/255
	return img

pd_dict = {'result' : [], 'filepath' : []}
folder = r'D:\Downloads\archive\pizza_not_pizza\pizza'

for count, filename in enumerate(os.listdir(folder)):
	
	src =f"{folder}\\{filename}"  # foldername/filename, if .py file is outside folder
	#print(src)
	ready_img = load_and_prep_image(src)
	final_data = new_model.predict(tf.expand_dims(ready_img,axis=0))
	print(final_data)
	#if final_data.round() == 0:
	#	print("It's a pizza")
	#else:
	#	print(f"Not a pizza {src}")
	pd_dict['result'].append(final_data)
	pd_dict['filepath'].append(src)


pd_log = pd.DataFrame(pd_dict)
pd_log.to_csv(r'D:\Downloads\archive\pizza_not_pizza\results3.csv',index=False)

