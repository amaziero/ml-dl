import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def importing_dataset_returning_xy()
	dataset = pd.read_csv("../../data/Data.csv")
	x = dataset.iloc[:, :-1].values
	y = dataset.iloc[:, -1].values
	return x, y

x, y = importing_dataset_returning_xy()