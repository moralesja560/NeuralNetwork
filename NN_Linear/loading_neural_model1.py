import tensorflow as tf
import numpy as np

new_model = tf.keras.models.load_model(r'C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\savedmodel')

# Check its architecture
new_model.summary()

#make a prediction using the trained model 

sample = -7.0

for i in range(0,20):
	print(f"Valor ingresado a la red neuronal: {sample} resultado correcto: {sample+10}")
	neural_guess =  round(np.double(new_model.predict(np.array([sample]))),4)
	print(f"valor de la red neuronal {neural_guess}")
	print(f"diferencia de valores {round((sample+10)-(neural_guess),3)}")
	sample += 10
