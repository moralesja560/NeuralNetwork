#let's create a toy tensor
import os
import matplotlib.pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import pandas as pd

def sigmoid_convert(orig_num):
	result = 1/(1+np.exp(-orig_num))
	return result


def relu_convert(orig_num):
	result = tf.maximum(0,orig_num).numpy()
	return result

def tanh_convert(x):
	return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

#create a tensor with 20 numbers
A = tf.cast(tf.range(-10,10), tf.float32)

#this list will store the sigmoid function
B = []
#and this will store relu
C = []
#and this will store tanh
D = []

for item in A:
	#actual sigmoid function
	res_sigmoid = sigmoid_convert(item.numpy())
	res_tanh = tanh_convert(item.numpy())
	res_relu = relu_convert(item.numpy())
	B.append(res_sigmoid)
	C.append(res_relu)
	D.append(res_tanh)


data={'Original Number':A, 'Sigmoid': B, 'RELU': C, 'TANH': D}

sigmoid_df = pd.DataFrame(data)


plt.plot(sigmoid_df)
plt.legend(sigmoid_df.columns.values)
plt.show()