#%%
import numpy as np

y_hat = np.array([1.2, 2.1, 2.9, 4.1, 4.7, 6.3, 7.1, 7.7, 8.5, 10.1])
y = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
diff_square = (y_hat - y) ** 2
e_mse = diff_square.sum() / len(y)
e_mse
# %%
