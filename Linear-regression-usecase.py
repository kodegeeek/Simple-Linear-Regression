# pip install python / pip3 install python
# pip install -U scikit-learn scipy matplotlib
# install the above packages in virtual environment
# install packages in files > settings > python:pycharm > pycharm interpretter > add the necessary packages





import matplotlib.pyplot as plt
import numpy as np
import sklearn as sklearn
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error



disease = datasets.load_diabetes()

# print(disease)
# print the data to know what are we dealing with right now .
# shows the dataset and we have target dataset also



# Splitting the data
disease_X = disease.data[:, np.newaxis,2]

disease_X_train =disease_X[:-30]
disease_X_test = disease_X[-20:]

disease_Y_train = disease.target[:-30]
disease_Y_test =disease.target[-20:]


# Generating our model
reg = linear_model.LinearRegression()   # calling the model

# Next Fit your data inside the model using the fit function
reg.fit(disease_X_train,disease_Y_train)



# Next make a prediction variable
Y_predict = reg.predict(disease_X_test)

# checking the accuracy
accuracy = mean_squared_error(disease_Y_test, Y_predict)
print(accuracy)

weights = reg.coef_
intercept = reg.intercept_
print(weights, intercept)


# ploting
plt.scatter(disease_X_test,disease_Y_test)
plt.plot(disease_X_test, Y_predict)
plt.show()