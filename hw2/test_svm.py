"""
A few simple tests for svm.py
"""

import numpy as np
import pytest

from hw2.svm import SVM


def test_svm_basic():
    clf = SVM(SVM.Config("rbf", 100.0, 1.0))

    # make a stupid simple linearly separable 2D dataset
    xtrain_a = [[1, 0], [2, 0], [3, 0]]
    xtrain_b = [[-1, 5], [0, 5], [1, 5]]
    X = np.array(xtrain_a + xtrain_b, dtype="float")
    y = np.array([0] * len(xtrain_a) + [1] * len(xtrain_b), dtype="int")

    clf.fit(X, y)

    # poke at some test points
    xtest_a = [
        [1.5, 0],
        [4, 0],
        [4, -4],
        [3, -400],
    ]
    xtest_b = [[1.5, 5], [1.5, 7], [4, 6]]
    xtest = np.array(xtest_a + xtest_b, dtype="float")
    ytest = np.array([0] * len(xtest_a) + [1] * len(xtest_b), dtype="int")

    yhat = clf.predict(xtest)
    np.testing.assert_allclose(yhat, ytest)
