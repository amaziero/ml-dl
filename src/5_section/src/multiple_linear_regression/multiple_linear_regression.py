#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def importing_dataset_returning_xy():
	"""This factory imports the dataset and returns x and y to do an
	lienar regression model

	Returns:
			x, y: separated structurs to continue the model
	"""	
	dataset = pd.read_csv("../../data/50_Startups.csv")
	x = dataset.iloc[:, :-1].values
	y = dataset.iloc[:, -1].values
	return x, y

def splitting_data(x, y):
	"""
	This factory receives the x and y variables builded before for modeling and
	returns the train and test objects

	Args:
			x numpy array: numpy array with the independent variables
			y array: and array with the dependent variable

	Returns:
			array: he train and test objects to make the predictions
	"""	
	from sklearn.model_selection import train_test_split

	X_train, x_test, Y_train, y_test = train_test_split(
		x,
		y,
		test_size = 0.2,
		random_state = 1
	)

	return X_train, x_test, Y_train, y_test

def one_hot_enconding(x):
	"""This factory is responsable for encoding the colluns with str values
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
		OneHotEncoder(), [3])],
		remainder = 'passthrough'
	)

	# forcing the return to be an numpy array after transformer to onehot encoding
	x = np.array(ct.fit_transform(x))
	
	return x

def main():
	"""

	This factory is responsable to call all the other factories which are neadded
	to do the model

	"""	
	x, y = importing_dataset_returning_xy()

	x = one_hot_enconding(x)

	print(x)

	# X_train, x_test, Y_train, y_test = splitting_data(x, y)

main()