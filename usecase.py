import matplotlib.pyplot as plt
import numpy as np
import sklearn as sklearn
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error



disease = datasets.load_diabetes()

print(disease)