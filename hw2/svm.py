"""
Support Vector Machine implementations.
"""

import warnings
from dataclasses import dataclass
from typing import Literal
import numpy as np
from qpsolvers import solve_qp


class SVMError(Exception):
    """Base error for SVM module."""

    pass


class SVM:
    """
    Linear-kernel support vector machine.

    Implemented following the paper:

    @inproceedings{Fletcher2010SupportVM,
        title={Support Vector Machines Explained},
        author={Tristan Fletcher},
        year={2010},
        url={https://api.semanticscholar.org/CorpusID:218451389}
    }

    With API inspiration taken from
    https://scikit-learn.org/stable/modules/svm.html

    """

    @dataclass
    class Config:
        # only support rbf for now
        kernel: Literal["rbf"]
        C: float
        rbf_stddev: float
        alpha_epsilon: float = 1e-8

    def __init__(self, cfg: Config) -> None:
        self._fitted = False
        self.c = cfg

        match self.c.kernel:
            case "rbf":
                self.kernel = self.rbf_kernel
                self.kernel_matrix = self.compute_rbf_kernel_matrix
            case _:
                raise SVMError(f"Unrecognized kernel type '{self.c.kernel}'")

        # storing these in fit()
        self._X: None | np.ndarray = None
        self._y: None | np.ndarray = None
        self._svec_indices: None | np.ndarray = None
        self._b: float = 0.0
        self._alpha: None | np.ndarray = None

    @property
    def svecs(self) -> np.ndarray:
        if not self._fitted:
            raise SVMError("fit() has not been called.")

        return self._alpha[self._svec_indices]  # type: ignore

    def rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        sqdist = np.sum((x1 - x2) ** 2)
        denom = 2 * self.c.rbf_stddev**2
        return np.exp(-sqdist / denom)

    def compute_rbf_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        # vectorized version!!
        sq_dists = (
            np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * X @ X.T
        )
        K = np.exp(-sq_dists / (2 * self.c.rbf_stddev**2))
        return K

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the classifier on a dataset.

        Args:
            X (np.ndarray): training data, shape=(n_samples, n_features)
            y (np.ndarray): training labels, shape=(n_samples,)
        """
        # this implementation assumes y \elem {-1, 1}, so we need to convert {0, 1}
        y = np.where(y == 0, -1, 1)

        # create H s.t. H_ij = y_i*y_j*(phi(x_i) dot phi(x_j))
        # (vectorized after much effort)
        L = len(y)
        H = np.outer(y, y) * self.compute_rbf_kernel_matrix(X)

        # find \alpha using a QP solver
        # unlike in Fletcher, we need to phrase this as a
        # minimization problem to conform with the library; hence we
        # take the negative of the equation given there

        # qT@alpha s.t. q = [1,..1] gives us the sum of all alphas
        q = -np.ones(shape=L, dtype="float")
        P = H
        lb = np.zeros(shape=L)
        ub = np.ones(shape=L) * self.c.C
        # phrase yT @ alpha = 0 as Ax=b by setting A to diag(y)
        A = y[np.newaxis, :]
        b = np.array([0.0])
        # we don't use Gx <= h
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Converted matrix")
            alpha = solve_qp(P=P, q=q, A=A, b=b, lb=lb, ub=ub, solver="clarabel")
        if type(alpha) is not np.ndarray:
            raise SVMError(f"Could not solve QP equation; returned {alpha}")

        # note that we can't actually calculate "w" as described in Fletcher,
        # since it involves taking phi(x) directly. We actually use it with the kernel
        # trick in the prediction function.

        # extract the support vectors
        # we need to use epsilon instead of 0 cuz the numerical solver
        # never quite reaches 0.
        svec_indices = np.where((self.c.alpha_epsilon < alpha) & (alpha <= self.c.C))[0]

        # finally, calculate b
        K_sv = self.compute_rbf_kernel_matrix(X[svec_indices])
        alpha_sv = alpha[svec_indices]
        y_sv = y[svec_indices]

        # inner_sum for all support vectors at once
        inner_sum = (alpha_sv * y_sv) @ K_sv  # shape: (n_sv,)
        b = np.mean(y_sv - inner_sum)

        self._X = X
        self._y = y
        self._svec_indices = svec_indices
        self._b = b  # type: ignore
        self._alpha = alpha
        self._fitted = True

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Perform classification on inputs x

        Args:
            x (np.ndarray): one or many inputs x of shape (n_samples, n_features)

        Returns:
            np.ndarray: output classes, of shape (n_samples)
        """
        if not self._fitted:
            raise SVMError("fit() has not been called.")

        X_sv = self._X[self._svec_indices]  # type: ignore
        alpha_sv = self._alpha[self._svec_indices]  # type: ignore
        y_sv = self._y[self._svec_indices]  # type: ignore

        # pairwise squared distances between X_test and support vectors
        sq_dists = (
            np.sum(x**2, axis=1)[:, None]
            + np.sum(X_sv**2, axis=1)[None, :]
            - 2 * x @ X_sv.T
        )
        kernel = np.exp(-sq_dists / (2 * self.c.rbf_stddev**2))

        out = (kernel @ (alpha_sv * y_sv)) + self._b
        return (out > 0).astype(int)
