import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.stats import norm, skew
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn import metrics
from sklearn import ensemble
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

import os,sys
import time


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


#load the data

training_PP = pd.read_csv(r'C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\Kaggle_Dataset\Training.csv')
predict_data = pd.read_csv(r'C:\Users\moralesja.group\Documents\SC_Repo\NeuralNetwork\Kaggle_Dataset\Testing.csv')


#print(training_PP.head())

sns.heatmap(training_PP.corr(),annot=True, cbar=True,cmap="Blues",fmt='.2f')
sns.pairplot(training_PP)
#plt.show()

# Data Preparation

X = training_PP.drop('PE',axis=1)
y = training_PP['PE']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

y_train= y_train.values.reshape(-1,1)
y_test= y_test.values.reshape(-1,1)

sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
y_train = sc_X.fit_transform(y_train)
y_test = sc_y.fit_transform(y_test)

#---------------------------- Linear Regression

reg_model = LinearRegression()
reg_model.fit(X_train,y_train)
print(reg_model)

predictions = reg_model.predict(X_test)
predictions= predictions.reshape(-1,1)


plt.figure(figsize=(10,5))
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
#plt.show()

plt.figure(figsize=(10,5))
plt.plot(y_test,label ='Test')
plt.plot(predictions, label = 'predict')
#plt.show()

#print('MAE:', metrics.mean_absolute_error(y_test, predictions))
#print('MSE:', metrics.mean_squared_error(y_test, predictions))

#print('R2:', metrics.r2_score(y_test, predictions))

#-------------- Decision Tree Regression-------------------------#

tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train,y_train)

pred_tree = tree_reg.predict(X_test)
pred_tree= pred_tree.reshape(-1,1)


plt.figure(figsize=(10,5))
plt.scatter(y_test,pred_tree)
plt.xlabel('Y Test ')
plt.ylabel('Tree Pred Y')
#plt.show()

plt.figure(figsize=(10,5))
plt.plot(y_test,label ='Test')
plt.plot(predictions, label = 'Tree pred')
#plt.show()

#print('MAE:', metrics.mean_absolute_error(y_test, pred_tree))
#print('MSE:', metrics.mean_squared_error(y_test, pred_tree))

#print('R2:', metrics.r2_score(y_test, pred_tree))


#-------------- Random Forest Regression-------------------------#

rand_reg = RandomForestRegressor(bootstrap=True, max_features= 2, n_estimators=40)
rand_reg.fit(X_train,y_train)
pred_randf = rand_reg.predict(X_test)
pred_randf= pred_randf.reshape(-1,1)

#print('MAE:', metrics.mean_absolute_error(y_test, pred_randf))
#print('MSE:', metrics.mean_squared_error(y_test, pred_randf))

#print('R2:', metrics.r2_score(y_test, pred_randf))

#----------------Suport Vector Machine Regression-------------------------#

svr_reg =SVR(kernel = 'rbf')
svr_reg.fit(X_train,y_train)

svr_pred = svr_reg.predict(X_test)
svr_pred= svr_pred.reshape(-1,1)




##---------------------------Cross Validation Evaluation-----------------

from sklearn.model_selection import cross_val_score

score_lin = cross_val_score(reg_model,X_train,y_train,scoring="neg_mean_squared_error",cv=10)

tree_rmse_scores_lin = np.sqrt(-score_lin)

print(f"Linear Regression Scores:  {score_lin.round(4)}")
print(f"Linear Regression Mean:  {score_lin.mean():.4f}")
print(f"Linear Regression Std Dev:  {score_lin.std():.4f}")
print('Linear Regression Training RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)).round(4))
print(f"Linear Regression RMSE:  {tree_rmse_scores_lin.round(4)}")


score_tree = cross_val_score(tree_reg,X_train,y_train,scoring="neg_mean_squared_error",cv=10)

tree_rmse_scores = np.sqrt(-score_tree)

print(f"Decision Tree Scores:  {score_tree.round(4)}")
print(f"Decision Tree Mean:  {score_tree.mean():.4f}")
print(f"Decision Tree Std Dev:  {score_tree.std():.4f}")
print('Decision Tree Training RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_tree)).round(4))
print(f"Decision Tree RMSE:  {tree_rmse_scores.round(4)}")

score_rand = cross_val_score(rand_reg,X_train,y_train,scoring="neg_mean_squared_error",cv=10)

rmse_scores_rand = np.sqrt(-score_rand)

print(f"Random Forest Scores:  {score_rand.round(4)}")
print(f"Random Forest Mean:  {score_rand.mean():.4f}")
print(f"Random Forest Std Dev:  {score_rand.std():.4f}")
print('Random Forest Training RMSE:', np.sqrt(metrics.mean_squared_error(y_test, pred_randf)).round(4))
print(f"Random Forest RMSE:  {rmse_scores_rand.round(4)}")


svm_score = cross_val_score(rand_reg,X_train,y_train,scoring="neg_mean_squared_error",cv=10)

rmse_scores_svm = np.sqrt(-svm_score)

print(f"SVM Scores:  {svm_score.round(4)}")
print(f"SVM Mean:  {svm_score.mean():.4f}")
print(f"SVM Std Dev:  {svm_score.std():.4f}")
print('SVM Training RMSE:', np.sqrt(metrics.mean_squared_error(y_test, svr_pred)).round(4))
print(f"SVM RMSE:  {rmse_scores_svm.round(4)}")


error_rate=pd.DataFrame(np.array([
						metrics.mean_squared_error(y_test, predictions),
						metrics.mean_squared_error(y_test, pred_tree),
						metrics.mean_squared_error(y_test, pred_randf),
						metrics.mean_squared_error(y_test, svr_pred)]))

error_rate.index=['Linear','Dec Tree','RandForest','SVM']

plt.figure(figsize=(10,5))
plt.plot(error_rate)
plt.show()


#-----------------Hyperparameter tuning: RandomForestRegressor----------------#

from sklearn.model_selection import GridSearchCV


parameter_grid = [{
	'n_estimators' : [3,10,30,35,40,45,50,100],
	'max_features' : [1,2,3,4,5],
	'bootstrap' : [False,True]}]

grid_search = GridSearchCV(rand_reg,
				parameter_grid,
				cv=5,
				scoring='neg_mean_squared_error',
				return_train_score=True)

grid_search.fit(X_train,y_train)

print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Estimator: {grid_search.best_estimator_}')

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
	print(np.sqrt(-mean_score),params)


#-----------------Hyperparameter tuning: SVM----------------#

from sklearn.model_selection import GridSearchCV


parameter_grid2 = [{
	'degree' : [1,2,3,],
	'gamma' : ['scale','auto'],
	'shrinking' : [False,True],
	'cache_size' : [200,400],
	'kernel':['poly','rbf','sigmoid']}
	
	
	]

grid_search2 = GridSearchCV(svr_reg,
				parameter_grid2,
				cv=5,
				scoring='neg_mean_squared_error',
				return_train_score=True)

grid_search2.fit(X_train,y_train)

print(f'Best Parameters SVM: {grid_search2.best_params_}')
print(f'Best Estimator SVM: {grid_search2.best_estimator_}')

cvres2 = grid_search2.cv_results_
for mean_score, params in zip(cvres2["mean_test_score"], cvres2["params"]):
	print(np.sqrt(-mean_score),params)



