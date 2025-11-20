"""
Support Vector Machine implementations.
"""

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

    def __init__(self, cfg: Config) -> None:
        self._fitted = False
        self.c = cfg

        match self.c.kernel:
            case "rbf":
                self.kernel = self.rbf_kernel
            case _:
                raise SVMError(f"Unrecognized kernel type '{self.c.kernel}'")

        # storing these in fit()
        self._X: None | np.ndarray = None
        self._y: None | np.ndarray = None
        self._svec_indices: None | np.ndarray = None
        self._b: float = 0.0
        self._alpha: None | np.ndarray = None
        self._w: None | np.ndarray = None

    @property
    def svecs(self) -> np.ndarray:
        if not self._fitted:
            raise SVMError("fit() has not been called.")

        return self._alpha[self._svec_indices]  # type: ignore

    def rbf_kernel(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        num = np.linalg.norm(x1 - x2, 2, 0) ** 2
        exp = -num / 2 / self.c.rbf_stddev
        out = np.exp(exp)
        return out

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the classifier on a dataset.

        Args:
            X (np.ndarray): training data, shape=(n_samples, n_features)
            y (np.ndarray): training labels, shape=(n_samples,)
        """
        # create H s.t. H_ij = y_i*y_j*(phi(x_i) dot phi(x_j))
        L = len(y)
        H = np.empty(shape=(L, L))

        # do this non-vectorized for now
        for i in range(L):
            for j in range(L):
                H[i][j] = y[i] * y[j] * self.kernel(X[i], X[j])

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
        A = np.diag(y)
        b = np.zeros(shape=L)
        # we don't use Gx <= h
        alpha = solve_qp(P=P, q=q, A=A, b=b, lb=lb, ub=ub, solver="proxqp")
        if type(alpha) is not np.ndarray:
            raise SVMError(f"Could not solve QP equation; returned {alpha}")

        # note that this is not "w" as described in Fletcher, since we can't
        # actually calculate it (it involves taking phi(x) directly).
        # this is actually just the weight that we can pass into phi(x),
        # which we will evaluate via the kernel trick during classification.
        w = np.empty(shape=X.shape[1])
        for i in range(L):
            w += alpha[i] * y[i] * X[i]

        # TODO remove this check and just vectorize if it works
        # assert alpha * y * X == w
        self._w = w

        # extract the support vectors
        svec_indices = np.where((0 < alpha) & (alpha <= self.c.C))[0]

        # finally, calculate b
        # again, do this non-vectorized for now
        outer_sum = 0
        for s in svec_indices:
            inner_sum = 0
            for m in svec_indices:
                inner_sum += alpha[m] * y[m] * self.kernel(X[m], X[s])
            outer_sum += y[s] - inner_sum
        b = 1 / len(svec_indices) * outer_sum

        self._X = X
        self._y = y
        self._svec_indices = svec_indices
        self._b = b  # type: ignore
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

        test = self.kernel(self._w, x) + self._b  # type: ignore
        return (test > 0).astype(int)
