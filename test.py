#各種ライブラリのImport
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#scikit-leanよりワインのデータをインポートする
df= pd.read_csv('winequality-red.csv',sep=';')
df1 = df.drop(columns='quality')
y = df['quality'].values.reshape(-1,1)
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()

# simple regression
for column in df1:
    x = df[column].values.reshape(-1,1)
    X = scaler.fit_transform(x)
    model = linear_model.LinearRegression()
    model.fit(X,y)
    mse = mean_squared_error(model.predict(X),y)
    print(column,model.coef_,model.intercept_,mse)

# Multiple Regression
x = df1.values
X = scaler.fit_transform(x)
model = linear_model.LinearRegression()
model.fit(X,y)
print(model.coef_,model.intercept_)
'''
fig = plt.figure()
plt.scatter(x,y)
plt.plot(x.values.reshape(-1,1),model.predict(x.values.reshape(-1,1)))
fig.savefig('test.png')
'''