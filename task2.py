import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv("Salary_data.csv")
x = df['YearsExperience']
y = df['Salary']
print("Before Reshaping : {}".format(x.shape))
x = np.array(x).reshape(-1, 1)
y = np.array(y).reshape(-1, 1)
print("After Reshaping : {}".format(x.shape))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
mymodel = LinearRegression()
result = mymodel.fit(x_train, y_train)
y_pred = mymodel.predict(x_test)
plt.scatter(x_test, y_test, s=10)
plt.xlabel("x")
plt.ylabel("y")
plt.plot(x_test, y_pred, color="r")
plt.show()
print(result.coef_)
print(result.intercept_)
