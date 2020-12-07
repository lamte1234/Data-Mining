# bai 1


import pandas as pd
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score

df = pd.read_csv("../dataset/BostonHousing.csv")

predictors = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'lstat']
outcome = 'medv'

X = df[predictors]
Y = df[outcome]

train_X, test_X, train_y, test_y = train_test_split(X, Y, test_size = 0.4, random_state = 1)

variables = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptradio', 'lstat']

predictor_a = ['crim', 'chas', 'rm']

lm = LinearRegression()
lm.fit(train_X[predictor_a], train_y)
print('intercept:', lm.intercept_)
print(predictor_a, "\n", lm.coef_)
#forward
def train_model1(variables):
    if len(variables) == 0:
        return None
    model = LinearRegression()
    model.fit(train_X[variables], train_y)
    return model

def score_model1(model, variables):
    if len(variables) == 0:
        return AIC_score(train_y, [train_y.mean()] * len(train_y), model, df=1)
    return AIC_score(train_y, model.predict(train_X[variables]), model)



#backward
def train_model2(variables):
    model = LinearRegression()
    model.fit(train_X[variables], train_y)
    return model

def score_model2(model, variables):
    return AIC_score(train_y, model.predict(train_X[variables]), model)

#ii
# df.corr().to_csv("../dataset/corr.csv") correlation matrix
print(df.corr())
#iii
print("-------------------------------FORWARD-----------------------------")
best_model1, best_variables1 = forward_selection(train_X.columns, train_model1, score_model1, verbose=True)
print(best_variables1, len(best_variables1))
regressionSummary(test_y, best_model1.predict(test_X[best_variables1]))
predictValue1 = best_model1.predict(test_X[best_variables1])
residual1 = test_y - predictValue1
plt.hist(residual1, bins=25) # hist of residual
plt.show()
print("-----------------------------BACKWARD-------------------------------")
best_model2, best_variables2 = backward_elimination(train_X.columns, train_model2, score_model2, verbose=True)
print(best_variables2, len(best_variables2))
regressionSummary(test_y, best_model2.predict(test_X[best_variables2]))
predictValue2 = best_model2.predict(test_X[best_variables2])
residual2 = test_y - predictValue2
plt.hist(residual2, bins=25) # hist of residual
plt.show()
print("------------------------------STEPWISE------------------------------")
best_model3, best_variables3 = stepwise_selection(train_X.columns, train_model1, score_model1, verbose=True)
print(best_variables3, len(best_variables3))
regressionSummary(test_y, best_model3.predict(test_X[best_variables3]))
predictValue3 = best_model3.predict(test_X[best_variables3])
residual3 = test_y - predictValue3
plt.hist(residual3, bins=25) # hist of residual
plt.show()
#
#
