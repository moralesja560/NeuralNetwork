import os
from statistics import mean
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from posixpath import split
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
#import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import sys
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

#put the dataset into a pandas dataframe
water = pd.read_csv('water.csv')



# information about a column
#print(water["ITW1_PN"].value_counts())

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
water = water.drop('temp_adentro', axis=1)
water = water.drop('timestamp', axis=1)

"""

# information about the PD
print(water.info())
#plot
#water.hist(bins=50,figsize=(12,8))
#plt.show()

#split the dataset
train_set, test_set = train_test_split(water, test_size=0.2, random_state=42)
"""
# Stratify a column to prevent outliers.
water["torre_cat"] = pd.cut(water["Temp_Torre"], bins=[12,14,16,18,20,22,np.inf],labels=[1,2,3,4,5,6])

"""
water['torre_cat'].value_counts().sort_index().plot.bar(rot=0, grid=True)
plt.xlabel('Temp_Torre')
plt.ylabel('Frecuencia')
plt.show()
"""
splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2,random_state=42)
strat_splits = []

for train_index, test_index in splitter.split(water,water["torre_cat"]):
	strat_train_set_n = water.loc[train_index]
	strat_test_set_n = water.loc[test_index]
	strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, strat_test_set = strat_splits[0]

print(strat_test_set['torre_cat'].value_counts() / len(strat_test_set))

#drop the torre_Cat
strat_test_set = strat_test_set.drop('torre_cat', axis=1)
strat_train_set = strat_train_set.drop('torre_cat', axis=1)
"""
#Make a copy of the data
water = strat_train_set.copy()


#its time for correlations.

corr_matrix = water.corr()

print(corr_matrix['Temp_Tanque'])

from pandas.plotting import scatter_matrix

attributes = ["Temp_Torre", "Clima_Temp", "Clima_Humedad"]
scatter_matrix(water[attributes],figsize=(12,8))
plt.show()

water['avg_diam'] = (water['ITW1_PN']+water['ITW2_PN']+water['ITW3_PN'])/3
water['acc_diam'] = water['ITW1_PN']+water['ITW2_PN']+water['ITW3_PN']
water['lines_run1'] = water.ITW1_Auto.replace({True: 1, False: 0})
water['lines_run2'] = water.ITW2_Auto.replace({True: 1, False: 0})
water['lines_run3'] = water.ITW3_Auto.replace({True: 1, False: 0})
water['lines_running'] = water['lines_run1'] +water['lines_run2'] +water['lines_run3'] 
water = water.drop('lines_run1', axis=1)
water = water.drop('lines_run2', axis=1)
water = water.drop('lines_run3', axis=1)

print(water.head())

corr_matrix = water.corr()
print(corr_matrix['Temp_Tanque'].sort_values(ascending=False))
"""
water = strat_train_set.copy()
water_num = strat_train_set.drop('Temp_Tanque', axis=1)
water_labels = strat_train_set['Temp_Tanque'].copy()


#Handle text and cat attributes
# separate from original data
water_cat = water[['ITW1_Auto','ITW2_Auto','ITW3_Auto','Bomba_1','Bomba_2']]

ordinal = OrdinalEncoder()
water_cat_encoded = ordinal.fit_transform(water_cat)
print(water_cat_encoded[:8])

min_max_scaler = MinMaxScaler(feature_range=(-1,1))
water_scaled = min_max_scaler.fit_transform(water_num)

std_scaler = StandardScaler()
water_std_scaled = std_scaler.fit_transform(water_num)


