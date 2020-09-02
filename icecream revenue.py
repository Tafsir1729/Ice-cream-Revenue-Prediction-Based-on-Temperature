
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

IceCream = pd.read_csv("icecreamDataset.csv")
IceCream.head()
IceCream.tail()
IceCream.describe()
IceCream.info()


sns.jointplot(x='Temperature', y='Revenue', data = IceCream)
sns.pairplot(IceCream)
sns.lmplot(x='Temperature', y='Revenue', data=IceCream)


y = IceCream['Revenue']
X = IceCream[['Temperature']]

from sklearn.model_selection import train_test_split

#splitting the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


X_train.shape

from sklearn.linear_model import LinearRegression


regressor = LinearRegression(fit_intercept =True)
regressor.fit(X_train,y_train)



y_predict = regressor.predict(X_test)
y_predict
y_test

#VISUALIZE TRAIN SET RESULTS
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.ylabel('Revenue [Dollars]')
plt.xlabel('Temperature [degC]')
plt.title('Revenue Generated vs. Temperature @Ice Cream Stand(Training dataset)')

#VISUALIZE TEST SET RESULTS
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_test, regressor.predict(X_test), color = 'blue')
plt.ylabel('Revenue [dollars]')
plt.xlabel('Hours')
plt.title('Revenue Generated vs. Hours @Ice Cream Stand(Test dataset)')

#plt.show()

y_predict = regressor.predict([[24]])
y_predict

print("Result: ",y_predict)