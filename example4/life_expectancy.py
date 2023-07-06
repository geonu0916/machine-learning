#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import normalize

life = pd.read_csv('https://github.com/dknife/ML/raw/main/data/life_expectancy.csv')

life = life[['Life expectancy', 'Year', 'Alcohol', 'Percentage expenditure', 'Total expenditure', 'Hepatitis B', 'Measles', 'Polio', 'BMI', 'GDP', 'Thinness 1-19 years', 'Thinness 5-9 years']]

print(life.shape)
print(life.isnull().sum())
life.dropna(inplace=True)
print(life.shape)

'''
sns.set(rc = {'figure.figsize' : (12, 10)})
correction_matrix = life.corr().round(2)
sns.heatmap(data=correction_matrix, annot= True)
'''

#sns.pairplot(life[['Life expectancy', 'Alcohol', 'Percentage expenditure', 'Measles', 'Polio', 'BMI', 'GDP', 'Thinness 1-19 years']])
x = life[['Alcohol', 'Percentage expenditure', 'Polio', 'BMI', 'GDP', 'Thinness 1-19 years']]
y = life['Life expectancy']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
'''
y_hat_train = lin_model.predict(X_train)
plt.scatter(y_train, y_hat_train)
xy_range = [40, 100]
plt.plot(xy_range, xy_range)

y_hat_test = lin_model.predict(X_test)
plt.scatter(y_test, y_hat_test)
xy_range = [40, 100]
plt.plot(xy_range, xy_range)
'''

n_X = normalize(x, axis=0)

nXtrain, nXtest, y_train, y_test = train_test_split(n_X, y, test_size=0.2)
lin_model.fit(nXtrain, y_train)

y_hat_train = lin_model.predict(nXtrain)
y_hat_test = lin_model.predict(nXtest)
plt.scatter(y_train, y_hat_train, color = 'r')
plt.scatter(y_test, y_hat_test, color = 'b')
xy_range = [40, 100]
plt.plot(xy_range, xy_range)
# %%
