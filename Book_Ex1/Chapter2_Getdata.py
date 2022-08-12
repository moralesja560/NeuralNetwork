from cProfile import label
import os
from posixpath import split
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


tf.random.set_seed(42)

# SET the environmental variables
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets","housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

#dowload from internet the dataset
def fetch_housing_data(housing_url = HOUSING_URL,housing_path = HOUSING_PATH):
	if not os.path.isdir(housing_path):
		os.makedirs(housing_path)
	tgz_path = os.path.join(housing_path,"housing.tgz")
	urllib.request.urlretrieve(housing_url,tgz_path)
	housing_tgz = tarfile.open(tgz_path)
	housing_tgz.extractall(path=housing_path)
	housing_tgz.close()

#put the dataset into a pandas dataframe
def load_housing_data(housing_path=HOUSING_PATH):
	csv_path = os.path.join(housing_path,"housing.csv")
	return pd.read_csv(csv_path)

#fetch_housing_data()

housing = load_housing_data()

print(housing.info())

#Find out what categories exist and how many districts belong to each category by using the value_counts() method

print(housing["ocean_proximity"].value_counts())


#some stats from the columns
	#null values are ignored

print(housing.describe())
	
#a quick plot
housing.hist(bins=50, figsize=(20,15))
plt.show()

#there's a bit of a problem with the data: info is capped (median_income goes from 0.4999 to 15) and scaled (median_income needs to be multiplied by 10,000)

#we need to split the dataset into two subsets: ""

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


#let's cluster a column to prevent data outliers

housing["income_cat"] = pd.cut(housing["median_income"], bins=[0.,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

housing["income_cat"].hist(bins=50, figsize=(20,15))
plt.show()


#time to do some stratified sampling

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)

for train_index, test_index in split.split(housing,housing["income_cat"]):
	strat_train_set = housing.loc[train_index]
	strat_test_set = housing.loc[test_index]

print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))

#Let's imagine the following scenario: A full dataset contains random values with exact proportions.
	# just splitting it in random test and train subsets is skewed because the random train subset may not contain the same proportions
	# we stratified the information by clustering the income in 5 labels, and splitting the test/train information taking care of keeping the proportions the same

#it's time to drop the temporary column "income cat"

for set_ in (strat_test_set,strat_train_set):
	set_.drop("income_cat",axis=1,inplace=True)

