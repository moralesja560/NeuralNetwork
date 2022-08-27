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
smoke_det_df = pd.read_csv(resource_path('smoke_detection_iot.csv'))
# I believe we may need to remove the UTC and CNT columns, they're not even attributes , just a chronological timestamp info and a counter.
smoke_det_df.drop('UTC', inplace=True, axis=1)
smoke_det_df.drop('CNT', inplace=True, axis=1)
smoke_det_df.drop('Unnamed: 0', inplace=True, axis=1)

#display the data structure
print(smoke_det_df.head())
#display column data type
print(smoke_det_df.info())
#describe data statistical indicators
print(smoke_det_df.describe())



## All numbers, great. i think we may only need to scale and standardize the data, so not one-hot encoding needed

###---------Data insight
smoke_det_df.hist(bins=50, figsize=(14,10))
#plt.show()


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

## DROP the columns used in stratified sampling: "Humidity_S" and "Temperature_R"

for set_ in (strat_train_set, strat_test_set):
	set_.drop("Humidity_S", axis=1, inplace=True)
	set_.drop("Temperature_R", axis=1, inplace=True)

###-----------------------GAIN DATA INSIGHTS-----#

# First, a copy to protect the original data from mishandling
smoke_df = strat_train_set.copy()


# Let's calculate the standard correlation coefficient between every pair of attributes.
# Remember: Fire Alarm is the independent variable that will be predicted by this model

corr_matrix = smoke_df.corr()

print(corr_matrix["Fire Alarm"].sort_values(ascending=False))

## Pretty interesting stuff

# Fire Alarm        1.000000
# Humidity[%]       0.396037
# Pressure[hPa]     0.243615
# Raw H2            0.105488
# NC2.5            -0.055800
# NC1.0            -0.080361
# PM2.5            -0.082439
# eCO2[ppm]        -0.093245
# PM1.0            -0.108479
# NC0.5            -0.126891
# Temperature[C]   -0.163398
# TVOC[ppb]        -0.213736
# Raw Ethanol      -0.341358

# humidity has the strongest positive correlation with Fire Alarm. Humidity goes up when Fire Alarm goes up.
# Raw Ethanol has the stronges negative correlation with Fire Alarm. Raw Ethanol goes down when Fire Alarm goes up.

## let's use pandas to visually check this correlations

from pandas.plotting import scatter_matrix
attributes = ["Fire Alarm", "Humidity[%]", "Pressure[hPa]", "Raw Ethanol", "TVOC[ppb]"]
#scatter_matrix(smoke_df[attributes], figsize=(12, 8))
#plt.show()

##----------COMBINING ATTRIBUTES-------------------#

# similar to an Excel's Pivot Table calculated column, let's try a few attributes combination to see if 
# the combination results in better correlation with the labels ("Fire Alarm" column )

# Looked up in google: Higher levels of CO2 will increase the humidity.

smoke_df["CO2xHumidity"] = smoke_df["Humidity[%]"]*smoke_df["eCO2[ppm]"]
corr_matrix = smoke_df.corr()
print(corr_matrix["Fire Alarm"].sort_values(ascending=False))

# Not much of a success: CO2xHumidity     -0.049772

smoke_df["CO2/Humidity"] = smoke_df["eCO2[ppm]"]/smoke_df["Humidity[%]"]
corr_matrix = smoke_df.corr()
print(corr_matrix["Fire Alarm"].sort_values(ascending=False))

# Same, not much succes:  CO2/Humidity     -0.095861

###------------PREPARE THE DATA FOR MACHINE LEARNING ALGORITHMS-----------------####
smoke_df2 = strat_train_set.copy()

## TIME TO SEPARATE THE DATA IN FEATURES AND LABELS

smoke_feat = smoke_df2.drop("Fire Alarm", axis=1)
smoke_labels = smoke_df2["Fire Alarm"].copy()

print(smoke_feat.head())
print(smoke_labels.head())

####-------------FILLING THE BLANKS-------------------#

# most ML algorithms cannot work if there are missing values in the training set.
# a solution might be to replace missing values with the median (not average) column value.
# this procedure is not necessary, but it's safer to apply when on a data pipeline

from sklearn.impute import SimpleImputer

# load the SimpleImputer and select the strategy
imputer = SimpleImputer(strategy="median")

#In this case, our dataset has not categorical columns. Please use pandas.drop to remove any categorical/text column
imputer.fit(smoke_feat)
#let's see what the imputer came up with:
print(imputer.statistics_)
# are these numbers really the median of every column?
print(smoke_feat.median().values)
# actually transform the data
X = imputer.transform(smoke_feat)
# optional: put the stuff in a pandas DF: smoke_f_tr = pd.DataFrame(X,columns=smoke_feat.columns)


##--------------Data Pipeline------------#

# This stuff serves as an instruction list to perform a sequence of data transformations
# in this case, we may only have two: simpleimputer and standardscaler, 
# Let's suppose that we have a column with values ranging from 0 to 100 and another with values ranging from 0 to 15.
# standard scaler will transform both columns to have a similar range of values.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


smoke_pipeline = Pipeline([
('imputer', SimpleImputer(strategy="median")),
('std_scaler', StandardScaler()),
])

#let's transform this data
smoke_feat_tr = smoke_pipeline.fit_transform(smoke_feat)

#THERE YOU GO. PREPARED DATA READY TO BE FED TO A ML ALGORITHM

model_1 = tf.keras.Sequential([
	#tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
	tf.keras.layers.Dense(4,activation="tanh"),
	tf.keras.layers.Dense(4,activation="tanh"),
	tf.keras.layers.Dense(1,activation="sigmoid")
])

#2 Compile the model
model_1.compile(
	loss = tf.keras.losses.BinaryCrossentropy(),
	optimizer=tf.keras.optimizers.Adam(learning_rate=0.02),
	metrics=["accuracy"])
 
#3.Fit the model
history_1 = model_1.fit(smoke_feat_tr,smoke_labels,epochs=10,verbose=1)
print(f"Evaluación de modelo 7: {model_1.evaluate(smoke_feat_tr,smoke_labels)}")

history_1_df = pd.DataFrame(history_1.history)

history_1_df.plot()
plt.title("Model 7 loss curves")
#plt.show()

# Model training results:

# Epoch 50/50
# 1566/1566 [==============================] - 1s 551us/step - loss: 0.0115 - accuracy: 0.9970
# 1566/1566 [==============================] - 1s 460us/step - loss: 0.0041 - accuracy: 0.9992
# Evaluación de modelo 7: [0.004119544290006161, 0.9991617202758789]

# Nice. 0.004 error and 0.9991 accuracy

####-------------Evaluate the model on the test set--------#

# divide the test set into features and labels
smoke_test = strat_test_set.drop("Fire Alarm", axis=1)
smoke_test_labels = strat_test_set["Fire Alarm"].copy()
#pipeline the data
smoke_test_tr = smoke_pipeline.transform(smoke_test)


#Use the model to predict the data
smoke_pred = model_1.predict(smoke_test_tr)
smoke_pred_df = pd.DataFrame(smoke_pred,columns=['Predict'])

df = pd.concat( [smoke_pred_df.reset_index(drop=True), smoke_test_labels.reset_index(drop=True)], axis=1)


print(df.head())
