import tensorflow as tf
import numpy as np

new_model = tf.keras.models.load_model(r'C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\saved_complete')

# Check its architecture
new_model.summary()

#make a prediction using the trained model 

sample =10000
correct_output =sample+500

for i in range(0,60):
	print(f"Valor ingresado a la red neuronal: {sample} resultado correcto: {correct_output}")
	neural_guess =  round(np.double(new_model.predict(np.array([sample]))),4)
	print(f"valor de la red neuronal {neural_guess}")
	print(f"diferencia de valores {round((correct_output)-(neural_guess),3)} \n {neural_guess/((correct_output)):.2%} ")
	sample += 100
	correct_output =sample+500