#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

lin_data = pd.read_csv('https://raw.githubusercontent.com/dknife/ML/main/data/pollution.csv')
lin_data.plot(kind = 'scatter', x = 'input', y = 'pollution')

def h(x, param):
    return param[0] * x + param[1]

learning_iteration = 1000
learning_rate = 0.0025

param = [1, 1]

x = lin_data['input'].to_numpy()
y = lin_data['pollution'].to_numpy()

for i in range(learning_iteration):
    if i % 200 == 0:
        lin_data.plot(kind = 'scatter', x = 'input', y = 'pollution')
        plt.plot([0, 1], [h(0, param), h(1, param)])
    error = (h(x, param) - y)
    param[0] -= learning_rate * (error * x).sum()
    param[1] -= learning_rate * error.sum()
# %%
