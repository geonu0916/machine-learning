#%%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

x = np.array([0.9, 4.2, 9, 10, 13.1])
y = np.array([0, 0.2, 2.3, 5.2, 7.5])

w_list = np.arange(1.0, 0.0, -0.1)
mse_list = []
for w in w_list:
    y_hat = w * x
    mse_value = mse(y_hat, y)
    mse_list.append(mse_value)
    print('w = {:.1f}, 평균제곱 오차: {:.2f}'.format(w, mse(y_hat, y)))
plt.plot(w_list, mse_list)
    
# %%
