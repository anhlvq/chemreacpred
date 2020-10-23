import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Visualizing the Polymonial Regression results
def polyTransform(X, degree=6):
    # initialize X_transform
    X_transform = np.ones((X.size, 1))
    for j in range(degree + 1):
        if j != 0:
            x_pow = np.power(X, j)
            # append x_pow to X_transform
            X_transform = np.append(X_transform, x_pow.reshape(-1, 1), axis=1)
    return X_transform


def normalize(X):
    X[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)
    return X


def polyRegPredict(x, coefs):
    X_transform = polyTransform(x, degree=coefs.size - 1)
    # X_normalize = normalize(X_transform)
    y_pred = np.dot(X_transform, coefs)
    return y_pred


def viz_polymonial(x, y, degree=6, title='', coefs=0):
    X = x.reshape(-1, 1)
    # X = x.reshape(1,-1)
    Y = y.reshape(-1, 1)
    poly_reg = PolynomialFeatures(degree=degree)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, Y)
    plt.scatter(X, Y, color='red')
    plt.plot(x, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    if not (coefs is viz_polymonial.__defaults__[0]):
        # X_transform1 = transform(x.reshape(-1,1), degree = coefs.size-1)
        # y_pred = np.dot(X_transform1, coefs)
        y_pred = polyRegPredict(x.reshape(-1, 1), coefs)
        plt.scatter(x, y_pred, color='green')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title(title)
    plt.show()
    return
