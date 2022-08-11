import os
import tarfile
from six.moves import urllib
import pandas as pd


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

fetch_housing_data()

housing = load_housing_data()

print(housing.info())
	

