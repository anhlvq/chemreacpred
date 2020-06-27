from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.multioutput import MultiOutputRegressor

from core.kernel_regressor import KernelRegression


class CurveRegressor(BaseEstimator, RegressorMixin):
    """" Wrapper for MultiOutputRegressor + KernelRegressor
    """

    def __init__(self, kernel="polynomial", gamma=None):
        self.kernel = kernel
        self.gamma = gamma
        kr = KernelRegression(kernel, gamma)
        self.model = MultiOutputRegressor(estimator=kr)

    def fit(self, X, y):
        """Fit the model
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
        y : array-like, shape = [n_samples, n_targets]

        Returns
        -------
        self : object
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Predict target values for X.
        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
        Returns
        -------
        y : array of shape = [n_samples, n_targets]
        """
        y_pred = self.model.predict(X)
        return y_pred
