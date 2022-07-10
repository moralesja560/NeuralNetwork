import tensorflow as tf
import numpy as np
import os,sys 
import pandas as pd

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


new_model = tf.keras.models.load_model(resource_path(r"save_model_Non_Linear"))


def predict_data(x,y):
	print(new_model.predict([[x,y]]))



# Check its architecture
new_model.summary()

#make a prediction using the trained model 
df = pd.read_csv(r'c:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\resources\data_model7.csv')

result = [predict_data(x,y) for x, y in zip(df['Coordenada X'], df['Coordenada Y'])]



