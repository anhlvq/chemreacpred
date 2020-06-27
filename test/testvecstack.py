import glob
import os
import numpy as np
from numpy.testing import assert_array_equal

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

from vecstack import stacking

n_folds = 5
temp_dir = '../tmpdw35lg54ms80eb42'

try:
    os.rmdir(temp_dir)
except:
    print('Unable to remove temp dir')

try:
    os.mkdir(temp_dir)
except:
    print('Unable to create temp dir')

boston = load_boston()
X, y = boston.data, boston.target

np.random.seed(0)
ind = np.arange(500)
np.random.shuffle(ind)

ind_train = ind[:400]
ind_test = ind[400:]

X_train = X[ind_train]
X_test = X[ind_test]

y_train = y[ind_train]
y_test = y[ind_test]

model = LinearRegression()
S_train_1 = cross_val_predict(model, X_train, y=y_train, cv=n_folds,
                              n_jobs=1, verbose=0, method='predict').reshape(-1, 1)
_ = model.fit(X_train, y_train)
S_test_1 = model.predict(X_test).reshape(-1, 1)

models = [LinearRegression()]
S_train_2, S_test_2 = stacking(models, X_train, y_train, X_test,
                               regression=True, n_folds=n_folds, shuffle=False, save_dir=temp_dir,
                               mode='oof_pred', random_state=0, verbose=0)

file_name = sorted(glob.glob(os.path.join(temp_dir, '*.npy')))[-1]  # take the latest file
S = np.load(file_name, allow_pickle=True)
S_train_3 = S[0]
S_test_3 = S[1]

assert_array_equal(S_train_1, S_train_2)
assert_array_equal(S_test_1, S_test_2)

assert_array_equal(S_train_1, S_train_3)
assert_array_equal(S_test_1, S_test_3)

