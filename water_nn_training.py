import tensorflow as tf
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,Normalizer
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import numpy as np


#tf.random.set_seed(42)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


#put the dataset into a pandas dataframe
water = pd.read_csv('water7.csv')

#fix a couple of columns
water['ITW1_PN'] = water['ITW1_PN']/100
water['ITW2_PN'] = water['ITW2_PN']/100
water['ITW3_PN'] = water['ITW3_PN']/100

print(water.head())

## separate the dataset in two subsets: 80% of entire dataset will be the training data, and the remaining dataset will be the test data
train_set, test_set = train_test_split(water, test_size=0.2, random_state=42)

#Drop the predictors and the labels because we do not want to apply the same transformation
water_data = train_set.drop('temp_adentro',axis=1)
water_labels = train_set['temp_adentro'].copy()
"""

	ESTE CODIGO ES DE PRUEBA PARA REVISION DE LIBRO
#MinMax Scaler
min_max_scaler = MinMaxScaler(feature_range=(-1,1))
water_scaled = min_max_scaler.fit_transform(water_data)

#Std Scaler
std_scaler = StandardScaler()
water_std_scaled = std_scaler.fit_transform(water_data)

#try something with linear regression
target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(water_labels.to_frame())

model = LinearRegression()
model.fit(water_data[['Temp_Torre']],scaled_labels)
some_new_data = water_data[['Temp_Torre']].iloc[:5]

scaled_predictions = model.predict(some_new_data)
predictions = target_scaler.inverse_transform(scaled_predictions)
print(predictions)

water_pipeline = make_pipeline(StandardScaler())

water_data_prepared = water_pipeline.fit_transform(water_data)
print(water_data_prepared[:2].round(2))



# The definitive Pipeline for the data

from sklearn.compose import ColumnTransformer

num_attribs = ['timestamp','ITW1_PN','ITW2_PN','ITW3_PN','Temp_Torre','Clima_Temp','Clima_Humedad','ITW1_Spd','ITW2_Spd','ITW3_Spd','ITW1_KG','ITW2_KG','ITW3_KG']
bool_attribs = ['ITW1_Auto','ITW2_Auto','ITW3_Auto','Bomba_1','Bomba_2']

num_pipeline = make_pipeline(StandardScaler())

preprocessing = ColumnTransformer([
    ('num',num_pipeline,num_attribs),
])

water_final_prepared = preprocessing.fit_transform(water_data)
#print(water_final_prepared[:2].round(2))


lin_reg = make_pipeline(preprocessing,LinearRegression())
lin_reg.fit(water_data,water_labels)

print(water_data.head())
temp_preds = lin_reg.predict(water_data)
print(temp_preds[:5].round(2))
print(water_labels.iloc[:5].values)
"""

X_train_full, X_test, y_train_full, y_test = train_test_split(water_data, water_labels)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full)
#X_train.to_csv(resource_path("X_train_load.csv"),index=False)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)


"""
model = tf.keras.models.Sequential([
	tf.keras.layers.Dense(50, activation="relu", input_shape=X_train.shape[1:]),
#keras.layers.Dense(50, activation="relu"),
	tf.keras.layers.Dense(1)
])
model.compile(loss="mean_squared_error", optimizer="sgd")
history = model.fit(X_train_scaled, y_train, epochs=20,validation_data=(X_valid_scaled, y_valid))
mse_test = model.evaluate(X_test, y_test)

X_new = X_test[1:] # pretend these are new instances
X_new = scaler.transform(X_new)
y_pred = model.predict(X_new[:3])
print(f"{y_pred}")

"""


model_1 = tf.keras.Sequential([
	tf.keras.layers.Dense(20, activation ='tanh'),
    tf.keras.layers.Dense(20, activation ='tanh'),
    tf.keras.layers.Dense(20, activation ='tanh'),
    tf.keras.layers.Dense(20, activation ='tanh'),
    tf.keras.layers.Dense(20, activation ='tanh'),
	tf.keras.layers.Dense(1)
])

#2 Compile the model
model_1.compile(
	loss = tf.keras.losses.MeanAbsoluteError(),
	optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
	metrics=["mae"])

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=resource_path(r"TF_model_prototipe_temp"), monitor='val_mae',save_best_only= True,save_weights_only=False,verbose=1)
early_cb = tf.keras.callbacks.EarlyStopping(monitor='val_mae',min_delta=0.01,patience=18,verbose=1,mode='min')
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4*10**(epoch/20))
# 3.Fit the model
#history = model_1.fit(PP_train_feat_tr,PP_train_label,callbacks=[early_cb,cp_callback],steps_per_epoch=len(PP_train_label), validation_data=(PP_test_feat_tr,PP_test_label),validation_steps=len(PP_test_label), epochs=200)
history = model_1.fit(X_train_scaled, y_train,callbacks=[early_cb,cp_callback],steps_per_epoch=len(y_train), validation_data=(X_valid_scaled, y_valid),validation_steps=len(y_valid), epochs=100)



saved_model = tf.keras.models.load_model(r'C:\Users\moralesjo\OneDrive - Mubea\Documents\Python_S\NeuralNetwork\TF_model_prototipe')


#Temp_Torre  Bomba_1  Bomba_2  Clima_Temp  Clima_Humedad  Hour   avg_diam  acc_diam  lines_running

x_hour = 10
x_ITW1_PN = 14.2
x_ITW2_PN = 0
x_ITW3_PN = 14.2
x_ITW1_Auto = 1
x_ITW2_Auto = 0
x_ITW3_Auto = 1
x_Temp_Torre = 11.2
x_Bomba_1 = 1
x_Bomba_2 = 0
x_Clima_Temp = 14
x_Clima_Humedad = 14
x_ITW1_Spd = 229
x_ITW2_Spd = 0
x_ITW3_Spd = 250
x_ITW1_KG = 1500
x_ITW2_KG = 0
x_ITW3_KG = 1800


scaled = scaler.transform([[x_hour,x_ITW1_PN,x_ITW2_PN,x_ITW3_PN,x_ITW1_Auto,x_ITW2_Auto,x_ITW3_Auto,x_Temp_Torre,x_Bomba_1,x_Bomba_2,x_Clima_Temp,x_Clima_Humedad,x_ITW1_Spd,x_ITW2_Spd,x_ITW3_Spd,x_ITW1_KG,x_ITW2_KG,x_ITW3_KG]])

print(saved_model.predict(scaled))