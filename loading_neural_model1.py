import tensorflow as tf
import numpy as np

new_model = tf.keras.models.load_model(r'C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\savedmodel')

# Check its architecture
new_model.summary()

#make a prediction using the trained model 
print(new_model.predict(np.array([17.0])))