from cProfile import label
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt

# This neural network will be trained to detect fire in many environments:
	# indoor and outdoor 
	# Gas and wood-sourced fire
	# Outdoor gas/coal/wood gas grill
	# outdoor high humidity

tf.random.set_seed(42)

#This function sets the absolute path for the app to access its resources
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)

#Load the data
	# Notice that in this particular dataset, the first column has no name. I had to name it "index" to prevent the "Unnamed:0" error
smoke_det_df = pd.read_csv(resource_path('smoke_detection_iot.csv'),index_col=['index'])

#display the data structure
print(smoke_det_df.head())
#display column data type
print(smoke_det_df.info())
#describe data statistical indicators
print(smoke_det_df.describe())

## All numbers, great. i think we may only need to scale and standardize the data, so not one-hot encoding needed

###---------Data insight
smoke_det_df.hist(bins=50, figsize=(14,10))
plt.show()


##----------------------TRAINING AND VALIDATION DATASETS: CLASSIC APPROACH------------------#
from sklearn.model_selection import train_test_split

## separate the dataset in two subsets: 80% of entire dataset will be the training data, and the remaining dataset will be the test data
train_set, test_set = train_test_split(smoke_det_df, test_size=0.2, random_state=42)

##----------------STRATIFIED SAMPLING-------------------#

# let's assume that a very important parameters is temperature. 
# We need to ensure that the test and training subsets are representative of the various ranges of temperature in the main dataset
# By using smoke_det_df.describe() we learn that the temperature ranges are -22 to 60C (82 possible values)
# First guess: 7 bins of 12 C each. (7*12 = 84)

smoke_det_df["Temperature_R"] = pd.cut(smoke_det_df["Temperature[C]"],bins=[-24,-12,0,12,24,36,48,60,np.inf],labels=[1,2,3,4,5,6,7,8])

smoke_det_df['Temperature_R'].hist()
#plt.show()

## Plot does not show a classic normal distribution. Let's try with Humidity
	# remember: Less humidity means higher risk of fire. Humidity is inversely proportional to temp.
# By using smoke_det_df.describe() we learn that the humidity ranges are 10 to 76. ¿How about 6 bins of 12? 
smoke_det_df["Humidity_S"] = pd.cut(smoke_det_df["Humidity[%]"],bins=[5,17,29,41,53,65,77,np.inf],labels=[1,2,3,4,5,6,7])

smoke_det_df['Humidity_S'].hist()
#plt.show()

## better plot shape. Let's go with Humidity

from sklearn.model_selection import StratifiedShuffleSplit

# Let's split the dataset based on Humidity categories
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(smoke_det_df, smoke_det_df['Humidity_S']):
	strat_train_set = smoke_det_df.loc[train_index]
	strat_test_set = smoke_det_df.loc[test_index]

#Let's see if it worked

strat_proportions = strat_test_set["Humidity_S"].value_counts() / len(strat_test_set)
original_proportions = smoke_det_df["Humidity_S"].value_counts() / len(smoke_det_df)

print(f" stratified proportions: {strat_proportions}")
print(f" original proportions: {original_proportions}")

#AMAZING. very similar proportions between stratified sampling and original dataset
#stratified proportions: 
# 4    0.630129
# 5    0.278141
# 1    0.031375
# 3    0.030337
# 2    0.025866
# 6    0.004151
# 7    0.000000

#original proportions: 
# 4    0.630129
# 5    0.278125
# 1    0.031407
# 3    0.030321
# 2    0.025834
# 6    0.004183
# 7    0.000000

## DROP the column used in stratified sampling: "Humidity_S"

for set_ in (strat_train_set, strat_test_set):
	set_.drop("Humidity_S", axis=1, inplace=True)

###-----------------------Gain Data Insights-----#




