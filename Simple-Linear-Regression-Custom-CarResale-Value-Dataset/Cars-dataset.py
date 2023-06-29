# Simple Linear regression on custom dataset


import numpy as np
import pandas as pd
# because we are using custom dataset, we need to use pandas to import or to load the CSV over here

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

cars = pd.read_csv("CarPrice_Assignment.csv")
# print(cars.head())
print(cars.columns)

plt.figure(figsize=(16,8))
plt.scatter(
    cars["horsepower"],
    cars['price'],
    c ='black'
)

plt.xlabel=("horsepower")
plt.ylabel = ("price")
plt.show()


x=cars["horsepower"].values.reshape(-1,1)
y = cars["price"].values.reshape(-1,1)
# reshaping is basically to avoid errors


reg = LinearRegression()
reg.fit(x,y)

print(reg.coef_[0][0])
print(reg.intercept_[0])

predictions =reg.predict(x)

plt.figure(figsize=(16,8))
plt.scatter(
    cars["horsepower"],
    cars['price'],
    c='black'
)

plt.plot(
    cars['horsepower'],
    predictions,
    c='blue',
    linewidth=2
)



plt.xlabel=("horsepower")
plt.ylabel = ("price")
plt.show()
