
import os
from statistics import mean
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
#print(imputer.statistics_)

# let's actually replace the missing values within the dataset
X = imputer.transform(housing_num)

housing_tr= pd.DataFrame(X,columns=housing_num.columns)

#by using the print below, we can check that now all the columns have the same number of values
#print(housing_tr.describe())


### handling text and categorical attributes

# Most ML algorithms prefer to work with numbers anyway, so let's convert these categories into numbers

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

#load the data into a variable
housing_cat = housing[["ocean_proximity"]]

ordinal_encoder = OrdinalEncoder()

housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)

# see how the encoding went
#print(housing_cat_encoded[:10])
# see the categories that were replaced by numbers
#print(ordinal_encoder.categories_)

# Here, we have 5 categories of ocean proximity. A one-hot encoding will produce 5 columns
	# each column will be zero except the column that matches the data.
	# we have <1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'
	# then we will have 0,0,0,0,1  in each of these columns if the row contains "Near Ocean"

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
#print(housing_cat_1hot)


# TIME TO WRITE A CLASS TO HELP DO ALL OF THIS STUFF IN AN AUTOMATED WAY
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6

class CombinedAttributesAdder(BaseEstimator,TransformerMixin):
	def __init__(self,add_bedrooms_per_room =True):
		self.add_bedrooms_per_room = add_bedrooms_per_room
	def fit(self,X,y=None):
		return self
	def transform(self,X,y=None):
		rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
		population_per_household = X[:, bedrooms_ix] / X[:, rooms_ix]
		if self.add_bedrooms_per_room:
			bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
			return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
		else:
			return np.c_[X, rooms_per_household, population_per_household]

housing_extra_attribs = CombinedAttributesAdder(add_bedrooms_per_room=False).transform(housing.values)


# this automated transformer has ONE hyperparameter: Add bedrooms per room
	# the objective is to feed the ML algorithm the info and decide if the presence of the column "bedrooms per room" benefits the ML

##### FEATURE SCALING - MIN_MAX AND STANDARDIZATION

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
				('imputer', SimpleImputer(strategy="median")),
				('attribs_adder', CombinedAttributesAdder()),
				('std_scaler', StandardScaler()),
				])


housing_num_tr = num_pipeline.fit(housing_num)  
		# housing_num is the original data minus the Ocean_proximity. It is the data from before the imputer

# The pipeline constructor takes a list of name/estimator pairs, defining a sequence of transformation steps.
# All but the last estimator must have a fit_transform method.
# Call pipeline.fit to call fit_transform on every step on the pipeline, except the final step, where the pipeline will call only fit


## THE COMPLETE PIPELINE TO HANDLE NUMERICAL (NUM_PIPELINE) AND TEXT/CAT DATA (CAT_ATTRIBS)

from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
	("num", num_pipeline, num_attribs),
	("cat", OneHotEncoder(), cat_attribs),
	])
housing_prepared = full_pipeline.fit_transform(housing)

# we import the ColumnTransformer, then we retrieve the list of numerical and non numerical column names
# in the full pipeline, we provide a step name, a transformer and a list of stuff which the transformer will be applied to 


#####----------------- TRAIN A LINEAR REGRESSION MODEL -------------------------#####

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)

#the Linear Regression Model is already trained, let's try it with some data

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print(f" Predictions: {lin_reg.predict(some_data_prepared)}")
print(f"Truth: {list(some_labels)}")


# we've received our first prediction, but it was far off from what we expected
# Let's compute the MSE

from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels,housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

# we're off by $68,721. Most house prices range between 120k and 265k, so being off by 68k is terribly bad.

#tf_housing_labels= tf.convert_to_tensor(housing_labels)
#print(tf_housing_labels)
#tf_housing_prepared = tf.convert_to_tensor(housing_prepared)
#print(tf_housing_prepared)

#housing_prepared_df = pd.DataFrame(housing_prepared)

#exp_housing = housing_prepared_df.to_csv(r"C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\Book_Ex1\housing_prepared.csv")
#exp_housing_l = housing_labels.to_csv(r"C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\Book_Ex1\housing_labels.csv")

#Let's train a DecisionTreeRegresor, a more powerful model, capable of finding complex nonlinear relationships

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(housing_prepared,housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)

tree_mse = mean_squared_error(housing_labels,housing_predictions)

tree_mse = np.sqrt(tree_mse)
print(f" Tree MSE: {tree_mse}")

#the tree_mse is 0, so this model is technically absolutely perfect. But we know it may have overfit the data

#------------------EVALUATION TECHNIQUE: CROSS VALIDATION-----------------------#

# One way to evaluate the Decision Tree model would be to use the train_test_split AGAIN to split the training set into a training/test subset
# Let's use Scikit-Learn's K-fold cross validation feature
# It will split the training set into 10 folds and train the model using 9 folds and evaluating against 1 random fold.

from sklearn.model_selection import cross_val_score

scores = cross_val_score(
		tree_reg, # decision tree regressor
		housing_prepared, #the information that got through the pipeline
		housing_labels, # dataframe containing only labels
		scoring="neg_mean_squared_error",
		cv=10)

tree_rmse_scores = np.sqrt(-scores)

print(f" tree RMSE: {tree_rmse_scores}")

#[70826.71443491 68297.28520425 69681.9349547  72449.65733286 68789.2434467  71034.30847089 73761.06066747 66600.8923391 65767.05709785 72887.81778947]

#still too bad. with scores.mean() and scores.std() we can know that the Decision Tree has an avg of 71,000 with +/- 2439

# Let's try RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()

forest_reg.fit(housing_prepared,housing_labels)

housing_predictions = forest_reg.predict(housing_prepared)

forest_mse = mean_squared_error(housing_labels,housing_predictions)

forest_mse = np.sqrt(forest_mse)
print(f" Forest MSE: {forest_mse}")

scores = cross_val_score(
		forest_reg, # forest tree regressor
		housing_prepared, #the information that got through the pipeline
		housing_labels, # dataframe containing only labels
		scoring="neg_mean_squared_error",
		cv=10)

forest_rmse_scores = np.sqrt(-scores)

print(f" Forest RMSE Scores: {forest_rmse_scores}")

#  Forest RMSE Scores: [50886.26 49312.96 46463.198 51270.35 47336.98 50154.82 50929.00 48956.73 47411.97 53530.42]
# still bad but better
# think this models are overfitting the data
# Please notice something: The goal of these tests is to select two or three models that, without much tweaking, can get promising results.

## another model: SVR (Support Vector Machine)

from sklearn.svm import SVR

SVR_reg = SVR()

SVR_reg.fit(housing_prepared,housing_labels)

housing_predictions = SVR_reg.predict(housing_prepared)

SVR_mse = mean_squared_error(housing_labels,housing_predictions)

SVR_mse = np.sqrt(SVR_mse)
print(f" SVR MSE: {SVR_mse}")

scores = cross_val_score(
		SVR_reg, # forest tree regressor
		housing_prepared, #the information that got through the pipeline
		housing_labels, # dataframe containing only labels
		scoring="neg_mean_squared_error",
		cv=10)

SVR_rmse_scores = np.sqrt(-scores)

print(f" SVR RMSE Scores: {SVR_rmse_scores}")





#------------------------FINE TUNING THE SELECTED MODELS---------------------#

# GridSearchCV is a very important tool to test all possible combinations of hyperparameters.
from sklearn.model_selection import GridSearchCV
# parameters and possible values
param_grid = [
	{	'n_estimators': [3,10,30],  # 3 possible parameters of n_estimators
		'max_features': [2,4,6,8,10]},	# 4 possible parameters of max_features
	{	'bootstrap': [False],		# change this binary parameter
		'n_estimators': [3,10,30],		# 3 params
		'max_features': [2,3,4,5,6]},]	#  3 params

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(
				forest_reg,
				param_grid=param_grid,
				cv=5, 
				scoring = 'neg_mean_squared_error',
				return_train_score=True
)

grid_search.fit(housing_prepared,housing_labels)

# get the best parameters
print(f" best parameters {grid_search.best_params_}")

# get the best estimator
print(f" best estimator {grid_search.best_estimator_}")
print(f"\n")

#get the evaluation scores
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(np.sqrt(-mean_score),params)

#RandomForestRegressor can indicate the relative importante of each attribute

feature_importances = grid_search.best_estimator_.feature_importances_

print(f" Feature importances: {feature_importances}")

# Now display the feature importances and its names
#adding the missing labels that were created in this code
extra_attribs = ["rooms per houselhold", "pop_per_households", "bedroom_per_room"]
#calling only the category encoder from the pipeline
cat_encoder = full_pipeline.named_transformers_["cat"]
#saving only the category names into cat_one_hot_attribs
cat_one_hot_attribs = list(cat_encoder.categories_[0])
#concatenate all data into attributes
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
#display the sorted data 
print(sorted(zip(feature_importances,attributes), reverse=True))

print('##_________________SVR___________________##')

param_grid2 = [
	{	'kernel': ["linear"],
		'C': [1.0],
		'degree': [1,2],
		'gamma': ['scale']},	
	{	'kernel': ["rbf"],
		'C': [1.0],
		'gamma': ['scale'],
		'degree': [1,2,3]},	]	

SVR_reg = SVR()

grid_searchSVR = GridSearchCV(
				SVR_reg,
				param_grid=param_grid2,
				cv=5, 
				scoring = 'neg_mean_squared_error',
				return_train_score=True
)

grid_searchSVR.fit(housing_prepared,housing_labels)

# get the best parameters
print(f" best parameters SVR {grid_searchSVR.best_params_}")

# get the best estimator
print(f" best estimator SVR {grid_searchSVR.best_estimator_}")
print(f"\n")

#get the evaluation scores
cvres = grid_searchSVR.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(np.sqrt(-mean_score),params)



##-----------------prepare the test data for model evaluation---------------#

X_test = strat_test_set.drop("median_house_value", axis=1)
Y_test = strat_test_set["median_house_value"].copy()

# Please note the difference between this line and line in 303. Use transform only to transform the data, not fit_transform.
	# we do not want to fit the test set
X_test_prepared = full_pipeline.transform(X_test)

##----------------------Evaluate the Forest system on the Test set ----------------#

final_model_F = grid_search.best_estimator_

final_predictions_F = final_model_F.predict(X_test_prepared)

final_mse_F = mean_squared_error(Y_test,final_predictions_F)
final_rmse_F = np.sqrt(final_mse_F)

print(f" Final MSE Forest: {final_mse_F}")
print(f" Final RMSE Forest: {final_rmse_F}")

##----------------------Evaluate the SVR system on the Test set ----------------#

final_model_SVR = grid_searchSVR.best_estimator_

final_predictions = final_model_SVR.predict(X_test_prepared)

final_mse_SVR = mean_squared_error(Y_test,final_predictions)
final_rmse_SVR = np.sqrt(final_mse_SVR)

print(f" Final MSE SVR: {final_mse_SVR}")
print(f" Final RMSE SVR: {final_rmse_SVR}")





