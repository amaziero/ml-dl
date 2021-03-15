import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def importing_dataset_returning_xy():
	dataset = pd.read_csv("../../data/Data.csv")
	x = dataset.iloc[:, :-1].values
	y = dataset.iloc[:, -1].values
	return x, y

def replacing_missing_values_with_avarage(x):
	impute = SimpleImputer(missing_values=np.nan, strategy='mean')
	impute.fit(x[:, 1:3])
	x[:, 1:3] = impute.transform(x[:, 1:3])
	return x

def one_hot_enconding(x):
	"""This function is responsable for encoding the colluns with str values
	in this case onle the fisrt collumn is appliable

	Args:
			x (as array): the x array of and preprocessed dataset that will be latter
			used to train an mld or something alike.

		It returns an numpy array (x passed) with new the first's collumns encoded
	"""	
	ct = ColumnTransformer(
		transformers = [('encoder',
		OneHotEncoder(), [0])],
		remainder = 'passthrough'
	)

	# forcing the return to be an numpy array after transformer to onehot encoding
	x = np.array(ct.fit_transform(x))
	
	return x

x, y = importing_dataset_returning_xy()
x = replacing_missing_values_with_avarage(x)
x = one_hot_enconding(x)

print(x)
