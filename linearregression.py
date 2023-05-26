import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

df = pd.read_csv('/content/regressionexample.csv')
print(df.shape)
df.describe()

target_column = ['unemploy']
predictors = list(set(list(df.columns))-set(target_column))
df[predictors] = df[predictors]/df[predictors].max()
df.describe()
X = df[predictors].values
y = df[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=40)
print(X_train.shape); print(X_test.shape)

lr = LinearRegression()
lr.fit(X_train, y_train)

c = lr.intercept_
m = lr.coef_
print(m,c)

pred_train_lr= lr.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_lr)))
pred_test_lr= lr.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_lr)))

X_new = np.array([ [603.5, 202677, 11.7, 4.4]])
Y_new = lr.predict(X_new)
print(Y_new)
