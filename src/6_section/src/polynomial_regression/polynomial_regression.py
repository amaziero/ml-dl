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
	dataset = pd.read_csv("../../data/Position_Salaries.csv")
	x = dataset.iloc[:, 1:-1].values
	y = dataset.iloc[:, -1].values
	return x, y

def training_simple_model(x, y):
	from sklearn.linear_model import LinearRegression
	lin_reg = LinearRegression()
	lin_reg.fit(x, y)

	return lin_reg

def training_polominial_model(x, y):
	from sklearn.linear_model import LinearRegression
	from sklearn.preprocessing import PolynomialFeatures
	poly_Regr = PolynomialFeatures(degree = 2)
	x_poly = poly_Regr.fit_transform(x)
	lin_reg_2 = LinearRegression()
	lin_reg_2.fit(x_poly, y)

	return lin_reg_2, x_poly

	

def main():
	x, y = importing_dataset_returning_xy()
	lin_reg = training_simple_model(x, y)
	lin_reg_2, x_poly = training_polominial_model(x, y)

	plt.scatter(x, y, color = 'red')
	plt.plot(x, lin_reg.predict(x), color = 'blue')

	plt.scatter(x, y, color = 'red')
	plt.plot(x, lin_reg_2.predict(x_poly), color = 'blue')
	plt.title('Truthe or Bluff (Polynomial Regression)')

	plt.xlabel('Position Levels')
	plt.ylabel('Salary')


main()
# %%
