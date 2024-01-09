import tensorflow as tf

import pandas as pd
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train = pd.read_csv('X_train_load.csv',index_col=False)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
saved_model = tf.keras.models.load_model(r'C:\Users\moralesjo\OneDrive - Mubea\Documents\Python_S\NeuralNetwork\TF_model_prototipe')

#Temp_Torre  Bomba_1  Bomba_2  Clima_Temp  Clima_Humedad  Hour   avg_diam  acc_diam  lines_running
x_hour = 10
x_ITW1_PN = 14.2
x_ITW2_PN = 12.2
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
