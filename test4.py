# bai 4

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from dmba import regressionSummary
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

df = pd.read_csv("../dataset/ToyotaCorolla.csv")

outcome  = 'Price'
predictors = ['Age_08_04', 'KM', 'Fuel_Type', 'HP', 'Automatic', 'Doors', 'Quarterly_Tax', 'Mfr_Guarantee', 'Guarantee_Period', 'Airco', 'Automatic_airco', 'CD_Player', 'Powered_Windows', 'Sport_Model', 'Tow_Bar']

x = pd.get_dummies(df[predictors], drop_first=True)
y = df[outcome]
print(y.mean())
train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.3, train_size=0.5, random_state=1)

model = LinearRegression()
model.fit(train_x, train_y)

regressionSummary(valid_y, model.predict(valid_x))
print('MSE:', mean_squared_error(valid_y, model.predict(valid_x)))
print('R^2:', r2_score(valid_y, model.predict(valid_x)))
