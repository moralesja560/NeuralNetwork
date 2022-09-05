import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np

new_model = tf.keras.models.load_model(r'C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\CNN_savedmodel')

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

ready_img = load_and_prep_image(r"C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\resources\dataset_pizza_steak\image_custom1.jpg")

final_data = new_model.predict(tf.expand_dims(ready_img,axis=0))

print(final_data)

if final_data.round() == 0:
	print("It's a steak")
else:
	print("It's a pizza")

