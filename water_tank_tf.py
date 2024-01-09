import tensorflow as tf
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,Normalizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np


#tf.random.set_seed(42)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


#put the dataset into a pandas dataframe
water = pd.read_csv('water2.csv')

#fix a couple of columns
water['Temp_Tanque'] = water['temp_adentro']/10
water['Temp_Torre'] = water['Temp_Torre']/10
water['Clima_Temp'] = water['Clima_Temp']-273
water['ITW1_PN'] = water['ITW1_PN'].astype(str).str[:4]
water['ITW2_PN'] = water['ITW2_PN'].astype(str).str[:4]
water['ITW3_PN'] = water['ITW3_PN'].astype(str).str[:4]
water['ITW1_PN'] = water['ITW1_PN'].astype(float)
water['ITW2_PN'] = water['ITW2_PN'].astype(float)
water['ITW3_PN'] = water['ITW3_PN'].astype(float)
water['avg_diam'] = (water['ITW1_PN']+water['ITW2_PN']+water['ITW3_PN'])/3
water['acc_diam'] = water['ITW1_PN']+water['ITW2_PN']+water['ITW3_PN']
water['lines_run1'] = water.ITW1_Auto.replace({True: 1, False: 0})
water['lines_run2'] = water.ITW2_Auto.replace({True: 1, False: 0})
water['lines_run3'] = water.ITW3_Auto.replace({True: 1, False: 0})
water['lines_running'] = water['lines_run1'] +water['lines_run2'] +water['lines_run3'] 

water['Bomba_1'] = water.Bomba_1.replace({True: 1, False: 0})
water['Bomba_2'] = water.Bomba_2.replace({True: 1, False: 0})

water = water.drop('ITW1_Auto', axis=1)
water = water.drop('ITW2_Auto', axis=1)
water = water.drop('ITW3_Auto', axis=1)

water = water.drop('ITW1_PN', axis=1)
water = water.drop('ITW2_PN', axis=1)
water = water.drop('ITW3_PN', axis=1)

water = water.drop('lines_run1', axis=1)
water = water.drop('lines_run2', axis=1)
water = water.drop('lines_run3', axis=1)

water = water.drop('temp_adentro', axis=1)
water = water.drop('timestamp', axis=1)



#los que son 0 y 1 vamos a quitarlos, y luego los volvemos a poner.
waterbu = water.copy()
water = water.drop('Bomba_1', axis=1)
water = water.drop('Bomba_2', axis=1)

scaler = MinMaxScaler()

water_scaled = scaler.fit_transform(water.to_numpy())
water = pd.DataFrame(water_scaled, columns=water.columns)
water = pd.concat((water, waterbu.Bomba_1),axis=1)
water = pd.concat((water, waterbu.Bomba_2),axis=1)

print(water.head())


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(water, test_size=0.2, random_state=42)

PP_train_feat = train_set.drop("Temp_Tanque", axis=1)
PP_train_label = train_set["Temp_Tanque"].copy()

PP_test_feat = test_set.drop("Temp_Tanque", axis=1)
PP_test_label = test_set["Temp_Tanque"].copy()


print(f"pp_train feat \n{PP_train_feat.head()}")
sys.exit()
#print(f"pp_test label \n{PP_test_label.head()}")

#sys.exit()

smoke_pipeline = Pipeline([
('std_scaler', StandardScaler()
 ),
])

#Store PP_train_feat
PP_train_feat.to_csv(resource_path("PP_train_feat.csv"))
PP_train_feat_tr = smoke_pipeline.fit_transform(PP_train_feat)
PP_test_feat_tr = smoke_pipeline.transform(PP_test_feat)


saved_model = tf.keras.models.load_model(r'C:\Users\moralesjo\OneDrive - Mubea\Documents\Python_S\NeuralNetwork\TF_model')


#Temp_Torre  Bomba_1  Bomba_2  Clima_Temp  Clima_Humedad  Hour   avg_diam  acc_diam  lines_running

x_temp_torre = 10
x_bomba_1 = 1
x_bomba_2 = 0
x_clima_temp = 280.75-273
x_clima_hum = 19
x_hour = 8
x_itw1_PN = 13.80
x_itw2_PN = 14.10
x_itw3_PN = 13.50
x_avg_diam = (x_itw1_PN + x_itw2_PN +x_itw3_PN)/3
x_acc_diam = x_itw1_PN + x_itw2_PN +x_itw3_PN
x_lines_run = 2

array_predict = smoke_pipeline.transform([[x_temp_torre,x_bomba_1,x_bomba_2,x_clima_temp,x_clima_hum,x_hour,x_avg_diam,x_acc_diam,x_lines_run]])
print(saved_model.predict(array_predict))

sys.exit()







model_1 = tf.keras.Sequential([
	tf.keras.layers.Dense(10, activation ='relu'),
    tf.keras.layers.Dense(10, activation ='relu'),
	tf.keras.layers.Dense(1)
])

#2 Compile the model
model_1.compile(
	loss = tf.keras.losses.MeanAbsoluteError(),
	optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
	metrics=["mae"])

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=resource_path(r"TF_model"), monitor='val_mae',save_best_only= True,save_weights_only=False,verbose=1)
early_cb = tf.keras.callbacks.EarlyStopping(monitor='val_mae',min_delta=0.01,patience=10,verbose=1,mode='min')
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4*10**(epoch/20))
# 3.Fit the model
history = model_1.fit(PP_train_feat_tr,PP_train_label,callbacks=[early_cb,cp_callback],steps_per_epoch=len(PP_train_label), validation_data=(PP_test_feat_tr,PP_test_label),validation_steps=len(PP_test_label), epochs=200)


