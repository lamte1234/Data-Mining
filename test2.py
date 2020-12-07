## bai 2

import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from dmba import regressionSummary
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df2 = pd.read_csv("../dataset/Tayko.csv")

#a
print(pd.pivot_table(df2, index=['Gender=male', 'Address_is_res', 'US'], values='Spending', aggfunc= [np.mean, np.std]))
print('-----------------------------------------------------------------------------------')
#b
df2.plot(kind='scatter', x='last_update_days_ago', y='Spending')
plt.show()
print('-----------------------------------------------------------------------------------')
#c
df_new_record = df2.iloc[0:2000]
predictors = ['Freq', 'US', 'last_update_days_ago', 'Gender=male', 'Address_is_res', 'Web order']
outcome = 'Spending'

X = df_new_record[predictors]
y = df_new_record[outcome]

train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=1)

lm = LinearRegression()
lm.fit(train_x, train_y)

print("intercept:", lm.intercept_)
print(predictors)
print(lm.coef_)

result = lm.predict(test_x)
residuals = test_y - result
print(result[0], residuals[0])

print(regressionSummary(test_y, result))
print("MSE:", mean_squared_error(test_y,result))
print("R^2:", r2_score(test_y, result))

plt.hist(residuals, bins=25)
plt.show()

