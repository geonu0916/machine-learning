#%%
import matplotlib.pyplot as plt
import pandas as pd

lin_data = pd.read_csv('https://raw.githubusercontent.com/dknife/ML/main/data/pollution.csv')
lin_data.plot(kind = 'scatter', x = 'input', y = 'pollution')

w, b = 1, 1
x0, x1 = 0.0, 1.0
def h(x, w, b):
    return w * x + b

plt.plot([x0, x1], [h(x0, w, b), h(x1, w, b)])
plt.show()
# %%
