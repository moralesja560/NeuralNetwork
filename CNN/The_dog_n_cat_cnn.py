from random import shuffle
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense,Flatten, Conv2D,MaxPool2D,Activation
from tensorflow.keras import Sequential
import os, sys
import matplotlib.pyplot as plt
import pandas as pd

tf.random.set_seed(42)
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

#set the directory path WORK COMPUTER
train_dir = r'C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\resources\dataset_dogs_vs_cats2\train'
test_dir = r'C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\resources\dataset_dogs_vs_cats2\test'

##-----------------------The baseline model--------------------------#

def baseline_model(train_dir,test_dir,train_datagen,valid_datagen):
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
	
	model_base = tf.keras.models.Sequential([
	Conv2D(filters=10, kernel_size=3, strides=(1,1),padding= 'valid', activation = "relu", input_shape =(224,224,3)),
	tf.keras.layers.Conv2D(20,3,activation="relu"),
	tf.keras.layers.Conv2D(20,3,activation="relu"),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(1, activation='sigmoid')
	])
	model_base.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])

	#callbacks
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=resource_path(r"dogcatmodel"), monitor='val_accuracy',save_best_only= True,save_weights_only=False,verbose=1)

	model_base.fit(train_data,epochs=5,steps_per_epoch=len(train_data),validation_data=valid_data,validation_steps=len(valid_data),callbacks=[cp_callback],verbose=1)
	return model_base

def model2(train_dir,test_dir,train_datagen,valid_datagen):
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
	
	model_base = tf.keras.models.Sequential([
	tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(1,1),padding= 'same', activation = "relu", input_shape =(224,224,3)),
	tf.keras.layers.Conv2D(64,3,activation="relu"),
	tf.keras.layers.Conv2D(128,3,activation="relu"),
	tf.keras.layers.MaxPool2D(pool_size=2),
	tf.keras.layers.Flatten(),
	tf.keras.layers.Dense(1, activation='sigmoid')
	])
	model_base.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])

	#callbacks
	#cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=resource_path(r"dogcatmodel"), monitor='val_accuracy',save_best_only= True,save_weights_only=False,verbose=1)

	model_base.fit(train_data,epochs=5,steps_per_epoch=len(train_data),validation_data=valid_data,validation_steps=len(valid_data),verbose=1)
	return model_base




def data_transformation(zoom,shear,flip_v,flip_h,rotation,w_shift,h_shift):
	train_datagen = ImageDataGenerator(
    	rescale=1./255,
    	rotation_range=rotation,
    	shear_range=shear,
    	zoom_range=zoom,
    	width_shift_range=w_shift,
    	height_shift_range=h_shift,
    	horizontal_flip=flip_h,
    	vertical_flip=flip_v,
	)
    
	valid_datagen = ImageDataGenerator(rescale=1./255)
	return train_datagen,valid_datagen


if __name__ == '__main__':
	train_datagen_f,valid_datagen_f = data_transformation(zoom=0.0,shear=0.0,flip_h=False,flip_v=True,rotation=0.0,w_shift=0.0,h_shift=0.0)
	#baseline_model(train_dir=train_dir,test_dir=test_dir,train_datagen=train_datagen_f,valid_datagen=valid_datagen_f)
	model2(train_dir=train_dir,test_dir=test_dir,train_datagen=train_datagen_f,valid_datagen=valid_datagen_f)


	