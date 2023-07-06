#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

lin_data = pd.read_csv('https://raw.githubusercontent.com/dknife/ML/main/data/pollution.csv')

x = lin_data['input'].to_numpy()
y = lin_data['pollution'].to_numpy()

x = x[:, np.newaxis]

regr = linear_model.LinearRegression()
regr.fit(x, y)

lin_data.plot(kind='scatter', x='input', y='pollution')
y_pred = regr.predict([[0], [0.5]])
plt.plot([0, 0.5], y_pred)
# %%
