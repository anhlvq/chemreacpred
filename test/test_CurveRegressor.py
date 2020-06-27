from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from core.curve_regressor import CurveRegressor
from utils.gen_random_data import create_2targetsdata

model = CurveRegressor()

""" Make data
"""
n = 300
X, Y = create_2targetsdata(n)

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.15)
print("xtrain:", xtrain.shape, "ytrain:", ytrain.shape)
print("xtest:", xtest.shape, "ytest:", ytest.shape)

model.fit(xtrain, ytrain)

ypred = model.predict(xtest)
print("y1 MSE:%.4f" % mean_squared_error(ytest[:, 0], ypred[:, 0]))
print("y2 MSE:%.4f" % mean_squared_error(ytest[:, 1], ypred[:, 1]))

x_ax = range(len(xtest))
plt.plot(x_ax, ytest[:, 0], label="y1-test", color='g')
plt.plot(x_ax, ypred[:, 0], label="y1-pred", color='c')
plt.plot(x_ax, ytest[:, 1], label="y2-test", color='r')
plt.plot(x_ax, ypred[:, 1], label="y2-pred", color='b')
plt.legend()
plt.show()
