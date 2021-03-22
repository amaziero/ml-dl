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
	dataset = pd.read_csv("../../data/Salary_Data.csv")
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

def training_simple_linear_regression(X_train, Y_train, x_test):
	
	"""
		Training simple dataset to make linear regressions predictions

	Returns:
			arrays X_train, Ytrain and x_test: returns prediction based on x_test
	"""	
	
	from sklearn.linear_model import LinearRegression
	
	regressor = LinearRegression()
	regressor.fit(X_train, Y_train)

	y_prediction_test = regressor.predict(x_test)
	y_prediction_train = regressor.predict(X_train)


	return y_prediction_test, y_prediction_train

def visualising_train(X_train, y_train_test, y_prediction_train_test, x_test = None, type = 'Test'):
	if x_test is None:
		plt.scatter(X_train, y_train_test, color = 'red')
	else:
		plt.scatter(x_test, y_train_test, color = 'red')

	plt.plot(X_train, y_prediction_train_test, color = 'blue')
	plt.title(f'Salary x Experience ( ${type} Set )')

	plt.xlabel('Years of Experience')
	plt.ylabel('Salary')

	plt.show()

def main():
	"""

	This factory is responsable to call all the other factories which are neadded
	to do the model

	"""	
	x, y = importing_dataset_returning_xy()
	X_train, x_test, Y_train, y_test = splitting_data(x, y)
	
	y_prediction_test, y_prediction_train = training_simple_linear_regression(
		X_train,
		Y_train,
		x_test
	)

	visualising_train(X_train, y_train_test = Y_train, y_prediction_train_test = y_prediction_train)
	visualising_train(X_train, x_test =  x_test, y_train_test =  y_test, y_prediction_train_test = y_prediction_train, type = "Train")

main()