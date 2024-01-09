import tensorflow as tf
import numpy as np
import sys,os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

#make a prediction using the trained model 

saved_model = tf.keras.models.load_model(r'C:\Users\moralesjo\OneDrive - Mubea\Documents\Python_S\NeuralNetwork\TF_model')


#smoke_pipeline = Pipeline([
#('std_scaler', StandardScaler()
# ),
#])


array_predict = np.array([[18.3,1,0,14.38,57,1016,19,15.10,45.3,3]])
print(saved_model.predict(array_predict))
