import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def importing_dataset_returning_xy():
	dataset = pd.read_csv("../../data/Data.csv")
	x = dataset.iloc[:, :-1].values
	y = dataset.iloc[:, -1].values
	return x, y

def replacing_missing_values_with_avarage(x):
	"""This function replaces missing values with the mean value of the collumn's

	Args:
			x array: array

	Returns:
			array: array
	"""	
	from sklearn.impute import SimpleImputer	
	impute = SimpleImputer(missing_values=np.nan, strategy='mean')
	impute.fit(x[:, 1:3])
	x[:, 1:3] = impute.transform(x[:, 1:3])
	return x

def one_hot_enconding_first_collumn(x):
	"""This function is responsable for encoding the colluns with str values
	in this case onle the fisrt collumn is appliable

	Args:
			x (as array): the x array of and preprocessed dataset that will be latter
			used to train an mld or something alike.

	Returns:
			array: It returns an numpy array (x passed) with new the first's collumns encoded		
	"""
	
	from sklearn.compose import ColumnTransformer
	from sklearn.preprocessing import OneHotEncoder

	ct = ColumnTransformer(
		transformers = [('encoder',
		OneHotEncoder(), [0])],
		remainder = 'passthrough'
	)

	# forcing the return to be an numpy array after transformer to onehot encoding
	x = np.array(ct.fit_transform(x))
	
	return x

def laber_encoder_last_collumn(x):
	"""This function is responsable for laber encoder the last collumn of an
	array

	Args:
			x (array): array with the last row to be labeled
	Returns:
			array: array labeled
	"""	
	from sklearn.preprocessing import LabelEncoder

x, y = importing_dataset_returning_xy()
x = replacing_missing_values_with_avarage(x)
x = one_hot_enconding_first_collumn(x)

print(x)
