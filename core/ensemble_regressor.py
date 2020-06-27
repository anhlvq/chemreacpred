from __future__ import print_function
import sys
import traceback
from threading import Lock

from core.curve_regressor import CurveRegressor

if sys.version_info >= (3, 0):
    import builtins as bltin
else:
    import __builtin__ as bltin

import time

from core.dummy_pool_executor import DummyPoolExecutor
from sklearn.base import RegressorMixin, MetaEstimatorMixin, BaseEstimator
import numpy as np
# from theano.gradient import np


class EnsembleRegressor(BaseEstimator, MetaEstimatorMixin, RegressorMixin):
    def __init_ensembles(self):
        # (Not static any more) Static member variables
        self._ensemble_regressors_auto = (
            CurveRegressor(),
            CurveRegressor(),
            CurveRegressor()
        )

    def __init__(self, type='auto', verbose=False):
        '''
        :param type: Possible values: 'auto', 'mlp', 'mlp_large', 'ridge', 'auto_large' (defaults to 'auto').
                     Choice of set of regressors, 'auto' will use various standard regressors (usually linear
                     regression, NW-kernel, decision trees and random forests, but subject to change).
                     'mlp' will use 5 Multi-Layer Perceptrons, each with 10 hidden units, batch_size=32 and 1000 epochs.
                     'mlp_large' will use 10 MLPs, each with 10 hidden units, batch_size=32 and only 500 epochs.
                     'ridge' will train 5 ridge regressors with different alphas.
        :param verbose:
        '''
        self.__init_ensembles()
        self._thread_lock = Lock()
        self._verbose = verbose
        self.type = type.lower()  # convert type to lowercase
        """""
        if type == 'mlp':
            self.regressors = self._ensemble_nn
        elif type == 'mlp_large':
            self.regressors = self._ensemble_nn_large
        elif type == 'mlp_xlarge':
            self.regressors = self._ensemble_nn_xlarge
        elif type == 'mlp_different':
            self.regressors = self._ensemble_nn_different
        elif type == 'ridge':
            self.regressors = self._ensemble_ridge_regression
        elif type == 'auto_large':
            self.regressors = self._ensemble_auto_large
        else:
        """""
        self.regressors = self._ensemble_regressors_auto

        # set regressor labels
        self.regressor_labels = []
        self.regressor_count = len(self.regressors)
        for i, regr in enumerate(self.regressors):
            self.regressor_labels.append(str(regr))

    def _dprint(self, *args, **kwargs):
        """overload print() function to only print when verbose=True."""
        if self._verbose:
            with self._thread_lock:
                return bltin.print(*args, **kwargs)

    def fit_one(self, regr, X, y):
        try:
            return regr.fit(X, y)
        except Exception as e:
            print('Exception caught while trying to fit {0}:\n{1}'.format(regr, e), file=sys.stderr)

    def fit(self, X_train, y_train, samples_per_regressor=None, regressor_overlap=0):
        """ Fits the model for all the regression algorithms in the ensemble.
            The models themselves can be accessed directly at EnsembleRegressor.regressors,
            and their labels is accessible in EnsembleRegressor.regressor_labels.
        :param X_train: Data matrix. Shape [# samples, # features].
        :param y_train: Target value vector.
        :param samples_per_regressor: Number of samples from X_train that each regressor will be trained on.
                                      Default 'None' will cause all regressors to be trained on all samples.
        :param regressor_overlap: If samples_per_regressor is not None, this is the number of samples overlapping for
                                  every adjacent pair of regressors. Defaults to no overlap.
        """
        start_sample = 0
        if samples_per_regressor is None:
            end_sample = None
        else:
            end_sample = samples_per_regressor

        with DummyPoolExecutor() as pool:  # use Dummy or Thread. ProcessPool requires piping the fitted model back for 'predict'
            start = time.time()
            for i, regr in enumerate(self.regressors):
                self._dprint('## ' + str(i) + '. ' + str(regr))

                X = X_train[start_sample:end_sample, :]
                y = y_train[start_sample:end_sample]
                # regr.fit(X, y)
                pool.submit(regr.fit, X, y)

                if samples_per_regressor is not None:
                    start_sample = start_sample + samples_per_regressor - regressor_overlap
                    end_sample = start_sample + samples_per_regressor

                coef = getattr(regr, 'coef_', None)  # https://hynek.me/articles/hasattr/
                if coef is not None:
                    self._dprint('\tCoefficients: ', ', '.join(['%.2f' % f for f in coef]))

                alphas = getattr(regr, 'alphas_', None)
                if alphas is not None:
                    self._dprint('\tAlphas: ', ', '.join(['%.2f' % f for f in alphas]))

        self._dprint('Total running time: %.2f' % (time.time() - start))

    def predict(self, X):
        """
        :param X: Data matrix. Shape [# samples, # features].
        :return: Ensemble predictions. Shape [# regressors, # samples].
        """
        Z = np.ndarray(shape=(len(self.regressors), X.shape[0]))
        for i, regr in enumerate(self.regressors):
            # zip the real and predicted values together, sort them, and unzip them
            try:
                Z[i, :] = regr.predict(X)
            except:
                print(regr)
                raise

        return Z

    def score(self, X_test, y_test, **kwargs):
        """
        :return: vector with the R^2 score for each regressor
        """
        s = np.zeros(self.regressor_count)
        for i, regr in enumerate(self.regressors):
            try:
                s[i] = regr.score(X_test, y_test)
            except Exception as e:
                print('Exception caught while collecting results from {0}'.format(str(regr)), file=sys.stderr)
                print('{0}'.format(e), file=sys.stderr)
                traceback.print_tb(e.__traceback__)
                raise e
        return s

    def mean_squared_error(self, X_test, y_test):
        """
        :return: vector with the MSE for each regressor
        """
        Z = self.predict(X_test)
        return np.mean((Z - y_test[None, :]) ** 2, 1)
        # y[None, :] ensures that the vector is properly oriented
        # np.mean(..., 1) does the mean along the columns returning regressor_count results
