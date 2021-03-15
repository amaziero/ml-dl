import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer

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

x, y = importing_dataset_returning_xy()
x = replacing_missing_values_with_avarage(x)
print(x)