import os
from statistics import mean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf


### download the MNIST dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1,cache=True)
# let's see the keys of the MNIST dictionary
print(mnist.keys())

# have a look at these arrays
X,y = mnist["data"], mnist["target"]
print(f"shape of X: {X.shape}\nShape of y: {y.shape}")

#shape of X: (70000, 784)
### Shape of X means that there are 70k images and each image has 784 features. In other words, a 28x28 pixels resolution.
#Shape of y: (70000,)




