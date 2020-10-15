from numpy import array, hstack, math
from numpy.random import uniform
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


def create_data(n):
    x1 = array([math.sin(i) * (i / 10) + uniform(-5, 5) for i in range(n)]).reshape(n, 1)
    x2 = array([math.cos(i) * (i / 10) + uniform(-9, 5) for i in range(n)]).reshape(n, 1)
    x3 = array([(i / 50) + uniform(-10, 10) for i in range(n)]).reshape(n, 1)

    y1 = [x1[i] + x2[i] + x3[i] + uniform(-1, 4) + 15 for i in range(n)]
    y2 = [x1[i] - x2[i] - x3[i] - uniform(-4, 2) - 10 for i in range(n)]
    X = hstack((x1, x2, x3))
    Y = hstack((y1, y2))
    return X, Y


n = 300
X, Y = create_data(n)

f = plt.figure()
f.add_subplot(1, 2, 1)
plt.title("Xs input data")
plt.plot(X)
plt.xlabel("Samples")
f.add_subplot(1, 2, 2)
plt.title("Ys tsOutput data")
plt.plot(Y)
plt.xlabel("Samples")
plt.show()

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, test_size=0.15)
print("xtrain:", xtrain.shape, "ytrian:", ytrain.shape)
print("xtest:", xtest.shape, "ytest:", ytest.shape)

gbr = GradientBoostingRegressor()
model = MultiOutputRegressor(estimator=gbr)
print(model)

model.fit(xtrain, ytrain)
score = model.score(xtrain, ytrain)
print("Training score:", score)

ypred = model.predict(xtest)
print("y1 MSE:%.4f" % mean_squared_error(ytest[:, 0], ypred[:, 0]))
print("y2 MSE:%.4f" % mean_squared_error(ytest[:, 1], ypred[:, 1]))

x_ax = range(len(xtest))
plt.plot(x_ax, ytest[:, 0], label="y1-test", color='c')
plt.plot(x_ax, ypred[:, 0], label="y1-pred", color='b')
plt.plot(x_ax, ytest[:, 1], label="y2-test", color='m')
plt.plot(x_ax, ypred[:, 1], label="y2-pred", color='r')
plt.legend()
plt.show()

