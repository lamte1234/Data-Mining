# bai 3

import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from dmba import liftChart, gainsChart
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score

df = pd.read_csv("../dataset/Airfares.csv")


#a
explore_var = ['COUPON', 'NEW', 'HI', 'S_INCOME', 'E_INCOME', 'S_POP', 'E_POP', 'PAX', 'FARE']
df[explore_var].corr().to_csv("../dataset/corr2.csv")

df.plot(kind='scatter', x='COUPON', y='FARE')
plt.show()
print('-----------------------------------------------------------------------------------')
#b
cate_var = ['VACATION', 'SW', 'SLOT', 'GATE']
print(df[cate_var].describe())
print(df.pivot_table(index=cate_var, values='FARE'))
print('-----------------------------------------------------------------------------------')
#c
predictors = df.columns.drop('FARE')
outcome = 'FARE'

#i
x = pd.get_dummies(df[predictors], drop_first=True)
y = df[outcome]

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.4, random_state=1)
print('-----------------------------------------------------------------------------------')
#ii
lm = LinearRegression()
lm.fit(train_x, train_y)

#
predictor_2 = ['COUPON','NEW','VACATION','SW','HI','S_INCOME','E_INCOME','S_POP','E_POP','SLOT','GATE','DISTANCE','PAX']
x2 = pd.get_dummies(df[predictor_2], drop_first=True)
train_x2, test_x2, train_y2, test_y2 = train_test_split(x2, y, test_size=0.4, random_state=1)

#forward
def train_model1(variables):
    if len(variables) == 0:
        return None
    model = LinearRegression()
    model.fit(train_x2[variables], train_y2)
    return model

def score_model1(model, variables):
    if len(variables) == 0:
        return AIC_score(train_y2, [train_y2.mean()] * len(train_y2), model, df=1)
    return AIC_score(train_y2, model.predict(train_x2[variables]), model)

#backward
def train_model2(variables):
    model = LinearRegression()
    model.fit(train_x2[variables], train_y2)
    return model

def score_model2(model, variables):
    return AIC_score(train_y, model.predict(train_x2[variables]), model)

stepwise_model, step_var = stepwise_selection(train_x2.columns, train_model1, score_model1, verbose=True)


print('Intercept:', stepwise_model.intercept_)
print(step_var)
print(stepwise_model.coef_)
regressionSummary(test_y2, stepwise_model.predict(test_x2[step_var]))
print('-----------------------------------------------------------------------------------')
# iii
def train_model(variables):
    model = LinearRegression()
    model.fit(train_x2[list(variables)], train_y2)
    return model

def score_model(model, variables):
    pred_y = model.predict(train_x2[list(variables)])
    return -adjusted_r2_score(train_y2, pred_y, model)

exh_models = exhaustive_search(train_x2.columns, train_model, score_model)
data = []
for exh_model in exh_models:
    model = exh_model['model']
    variables = list(exh_model['variables'])
    AIC = AIC_score(train_y2, model.predict(train_x2[variables]), model)
    d = {'n': exh_model['n'], 'r2adj': -exh_model['score'], 'AIC': AIC}
    d.update({var: var in exh_model['variables'] for var in train_x2.columns})
    data.append(d)

# pd.DataFrame(data, columns=('n', 'r2adj', 'AIC') + tuple(sorted(train_x2.columns))).to_csv("../dataset/exhautive_model.csv")
print('-----------------------------------------------------------------------------------')
#iv
exh_model = exh_models[10]['model']
exh_var = exh_models[10]['variables']
print(exh_var)
regressionSummary(test_y2, exh_model.predict(test_x2[exh_var]))
pred_v = pd.Series(stepwise_model.predict(test_x2[step_var]))
pred_v = pred_v.sort_values(ascending=False)
pred_v2 = pd.Series(exh_model.predict(test_x2[exh_var]))
pred_v2 = pred_v2.sort_values(ascending=False)
fig, axes = plt.subplots(nrows=1, ncols=2)
ax = liftChart(pred_v, ax=axes[0], labelBars=False, title='Stepwise regression')
ax = liftChart(pred_v, ax=axes[1], labelBars=False, title='Exhaustive search')
ax.set_ylabel('Lift')
plt.tight_layout()
plt.show()

print('-----------------------------------------------------------------------------------')
#v
print(exh_model.predict(np.array([4442.141, 28760, 27664, 4557004, 3195503, 1976, 12782, 0, 0, 1, 1]).reshape(1, -1)))
print('-----------------------------------------------------------------------------------')
#vi
print(exh_model.predict(np.array([4442.141, 28760, 27664, 4557004, 3195503, 1976, 12782, 0, 1, 1, 1]).reshape(1, -1)))
print('-----------------------------------------------------------------------------------')
#viii
predictor_3 = ['COUPON','NEW','VACATION','SW','HI','S_POP','E_POP','SLOT','GATE','DISTANCE']
x3 = pd.get_dummies(df[predictor_3], drop_first=True)
train_x3, test_x3, train_y3, test_y3 = train_test_split(x3, y, test_size=0.4, random_state=1)

exh_models_2 = exhaustive_search(train_x3.columns, train_model, score_model)
model_index = 0
for emd in exh_models_2:
    max = 0
    if(-emd['score'] > max):
        max = -emd['score']
        model_index = emd['n'] - 1

exh_model_2 = exh_models_2[model_index]['model']
exh_2_var = exh_models_2[model_index]['variables']

print(exh_2_var)
print(exh_model_2.predict(np.array([1.202, 3, 4442.141, 4557004, 3195503, 1976, 0, 0, 1, 1]).reshape(1, -1)))
print('-----------------------------------------------------------------------------------')

regressionSummary(test_y3, exh_model_2.predict(test_x3[exh_2_var]))
