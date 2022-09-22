import tensorflow as tf
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

tf.random.set_seed(42)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


#load the data

training_PP = pd.read_csv(r'C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\Kaggle_Dataset\Training.csv')
predict_data = pd.read_csv(r'C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\Kaggle_Dataset\Testing.csv')


print(training_PP.head())


corr_matrix = training_PP.corr()
print(corr_matrix["PE"].sort_values(ascending=False))

from pandas.plotting import scatter_matrix
scatter_matrix(training_PP,figsize=(12,8))
#plt.show()

# most promising is AT (Ambient Temperature to PE (Energy Output)
# training_PP.plot(kind="scatter", x="PE",y="AT",alpha=0.1)
# plt.show()

# let's try with some combination
training_PP["TandP"] = training_PP["AT"]/training_PP["AP"]
training_PP["AT_EV"] = training_PP["EV"]*training_PP["AT"]

corr_matrix = training_PP.corr()
print(corr_matrix["PE"].sort_values(ascending=False))

#######################################  PREPARE THE DATA FOR MACHINE LEARNING ALGORITHMS

######### Data Cleaning
# Any missing values?
print(training_PP.isnull().values.any())

"""
Missing values? we can train a SimpleImputer instance to fill those blanks

Want to add new features? use sklearn.base import BaseEstimator, TransformerMixin

Categorical data? use one-hot encoding to convert text into numeric columns

Already using numerical data? let's optimize the dataset by scaling it.

"""
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(training_PP, test_size=0.2, random_state=42)

PP_train_feat = train_set.drop("PE", axis=1)
PP_train_label = train_set["PE"].copy()

PP_test_feat = test_set.drop("PE", axis=1)
PP_test_label = test_set["PE"].copy()


print(PP_train_feat.head())

print(PP_train_feat.describe())
print(PP_train_label.describe())


from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


smoke_pipeline = Pipeline([
('imputer', SimpleImputer(strategy="median")),
('std_scaler', StandardScaler()),
])


PP_train_feat_tr = smoke_pipeline.fit_transform(PP_train_feat)
PP_test_feat_tr = smoke_pipeline.transform(PP_test_feat)


model_1 = tf.keras.Sequential([
	#tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
	tf.keras.layers.Dense(20,activation="relu"),
	tf.keras.layers.Dense(20,activation="relu"),
	tf.keras.layers.Dense(20,activation="relu"),
	tf.keras.layers.Dense(20,activation="relu"),
	tf.keras.layers.Dense(20,activation="relu"),
	#tf.keras.layers.Dense(4,activation=None),
	tf.keras.layers.Dense(1)
])

#2 Compile the model
model_1.compile(
	loss = tf.keras.losses.MeanAbsoluteError(),
	optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
	metrics=["mae"])
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=resource_path(r"PP_model"), monitor='val_mae',save_best_only= True,save_weights_only=False,verbose=1)
early_cb = tf.keras.callbacks.EarlyStopping(monitor='val_mae',min_delta=0.01,patience=4,verbose=1,mode='min')
#lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4*10**(epoch/20))
#3.Fit the model
history = model_1.fit(PP_train_feat_tr,PP_train_label,callbacks=[early_cb,cp_callback],steps_per_epoch=len(PP_train_label), validation_data=(PP_test_feat_tr,PP_test_label),validation_steps=len(PP_test_label), epochs=100)
#=valid_data,validation_steps=len(valid_data)


## model has been saved, so let's use it.
saved_model = tf.keras.models.load_model(resource_path(r"PP_model"))



# Test Data truth dataframing
PP_test_label_df = pd.DataFrame(PP_test_label)
# Test data truth storage
PP_test_label_df.to_csv(resource_path(r"test_label.csv"))

#Test data prediction
predictions = saved_model.predict(PP_test_feat_tr)
#Test data storage
hi_df = pd.DataFrame(predictions)

# prediction storage
hi_df.to_csv(resource_path(r"predict.csv"))


