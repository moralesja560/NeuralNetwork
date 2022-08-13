
from lib2to3.pgen2.literals import simple_escapes
import os
from typing import ChainMap
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
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

#print(housing.info())

#Find out what categories exist and how many districts belong to each category by using the value_counts() method

#print(housing["ocean_proximity"].value_counts())


#some stats from the columns
	#null values are ignored

#print(housing.describe())
	
#a quick plot
housing.hist(bins=50, figsize=(20,15))
#plt.show()

#there's a bit of a problem with the data: info is capped (median_income goes from 0.4999 to 15) and scaled (median_income needs to be multiplied by 10,000)

#we need to split the dataset into two subsets: ""

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)


#let's cluster a column to prevent data outliers

housing["income_cat"] = pd.cut(housing["median_income"], bins=[0.,1.5,3.0,4.5,6.0,np.inf],labels=[1,2,3,4,5])

housing["income_cat"].hist(bins=50, figsize=(20,15))
#plt.show()


#time to do some stratified sampling

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)

for train_index, test_index in split.split(housing,housing["income_cat"]):
	strat_train_set = housing.loc[train_index]
	strat_test_set = housing.loc[test_index]

#print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))

#Let's imagine the following scenario: A full dataset contains random values with exact proportions.
	# just splitting it in random test and train subsets is skewed because the random train subset may not contain the same proportions
	# we stratified the information by clustering the income in 5 labels, and splitting the test/train information taking care of keeping the proportions the same

#it's time to drop the temporary column "income cat"

#print(strat_test_set.head())

for set_ in (strat_test_set,strat_train_set):
	set_.drop("income_cat",axis=1,inplace=True)

#print(strat_test_set.head())


#Visualizing Geographical Data
housing = strat_train_set.copy()

#plot the latitude and longitude in a scatter plot to visualize the data
# VISUALIZE VISUALIZE VISUALIZE

housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1, 
			s=housing["population"]/100, label="population", figsize=(10,7),c="median_house_value",cmap=plt.get_cmap("jet"),colorbar=True)

	# plot parameters:
	# s = scatter bubble radius
	# c = we select a numerical column to color the scatter bubbles
	# cmap = we select a predefined color map to indicate the range of values by using a gradient that goes from blue (lower) to red (higher)



#looking for correlations
# a correlation is positive  when the dependent variable goes up or down the same way as the independent variable
# a correlation is negative when the dependent variable goes up when the independent variable goes down.
# a correlation is stronger the closer is to 1.

#we create the correlation matrix
corr_matrix = housing.corr()

# we display it using a column as an independent variable
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# median_house_value    1.000000
# median_income         0.687151
# total_rooms           0.135140
# housing_median_age    0.114146
# households            0.064590
# total_bedrooms        0.047781
# population           -0.026882
# longitude            -0.047466
# latitude             -0.142673

# a small negative correlation exists between the latitude and the median house value. Prices have a slight tendency to go down when you go north.

# TIME TO USE SCATTER_MATRIX FROM PANDAS

from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age", "latitude"]
scatter_matrix(housing[attributes],figsize=(12,8))

#from the past 16 plots we saw, we concluded that the most promising attribute is the median house value / median_ income,
# let's zoom on that plot

housing.plot(kind="scatter",x="median_income", y ="median_house_value", alpha =0.1)
#plt.show()

# in this section, we create some calculated columns (similar to Excel PivotTable) in our dataframe. 

housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"] = housing["population"]/housing["households"]

#we add the newly created Calculated Columns (CC) to the correlation matrix

#we create the correlation matrix
corr_matrix = housing.corr()

# we display it using a column as an independent variable
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# this is the new correlation matrix with the CC included.

#median_house_value          1.000000
#median_income               0.687151
#rooms_per_household         0.146255
#total_rooms                 0.135140
#housing_median_age          0.114146
#households                  0.064590
#total_bedrooms              0.047781
#population_per_household   -0.021991
#population                 -0.026882
#longitude                  -0.047466
#latitude                   -0.142673
#bedrooms_per_room          -0.259952

#the new CC, "bedrooms per room" is much more correlated that its separate columns

#######################################  PREPARE THE DATA FOR MACHINE LEARNING ALGORITHMS

#drop the labels (dependent variable) to apply transformation only in the features
housing = strat_train_set.drop("median_house_value", axis=1)
# copy the train set again, but only the features
housing_labels = strat_train_set["median_house_value"].copy()

######### Data Cleaning

# in this case, total bedroom has some missing values, so we have 3 options: 
# 	get rid of that datapoints, eliminate the whole attribute or set those missing values to some value

from sklearn.impute import SimpleImputer

# this class will take care of the missing numerical values by calculating the selected strategy on every column

imputer = SimpleImputer(strategy="median")

#get rid temporarily of every non-numerical attribute

housing_num = housing.drop("ocean_proximity", axis=1)

#the actual imputer working
imputer.fit(housing_num)

# even if your dataset only have one incomplete column, it's safer to apply imputer to all numerical columns.
print(imputer.statistics_)

# let's actually replace the missing values within the dataset
X = imputer.transform(housing_num)

housing_tr= pd.DataFrame(X,columns=housing_num.columns)

#by using the print below, we can check that now all the columns have the same number of values
print(housing_tr.describe())


### handling text and categorical attributes

# Most ML algorithms prefer to work with numbers anyway, so let's convert these categories into numbers

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

#load the data into a variable
housing_cat = housing[["ocean_proximity"]]

ordinal_encoder = OrdinalEncoder()

housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

# see how the encoding went
print(housing_cat_encoded[:10])
# see the categories that were replaced by numbers
print(ordinal_encoder.categories_)

# Here, we have 5 categories of ocean proximity. A one-hot encoding will produce 5 columns
	# each column will be zero except the column that matches the data.
	# we have <1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'
	# then we will have 0,0,0,0,1  in each of these columns if the row contains "Near Ocean"

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)


# TIME TO WRITE A CLASS TO HELP DO ALL OF THIS STUFF IN AN AUTOMATED WAY





#from sklearn.compose import ColumnTransformer
#num_attribs = list(housing_num)
#cat_attribs = ["ocean_proximity"]
#full_pipeline = ColumnTransformer([
#("num", num_pipeline, num_attribs),
#("cat", OneHotEncoder(), cat_attribs),
#])
#housing_prepared = full_pipeline.fit_transform(housing)

