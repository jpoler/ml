import logging
import numpy as np
import numpy.typing as npt
from scipy.special import softmax # type: ignore
from typing import Any

from constants import convergence_epsilon
from model import Model
from fixed_basis import FixedBasisFunctionMixin, GaussianBasisMixin, PolynomialBasisMixin

class LogisticRegression(FixedBasisFunctionMixin, Model):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        N = len(X)
        k = len(y[0,:])
        if len(X.shape) == 1:
            X = X.reshape(len(X), 1)
        if N == 0:
            logging.warn("empty input to fit")
            return
        phi = self.phi(X)
        M = phi.shape[1]
        self.W = np.random.random((M, k))
        W_old: npt.NDArray[np.float64] = np.ones(self.W.shape) * np.inf
        W_new = self.W
        q = 0
        while q < 500 and not np.allclose(W_old, W_new, rtol=0., atol=convergence_epsilon):
            q += 1
            A: npt.NDArray[np.float64] = phi @ W_new
            S = softmax(A, axis=1)
            H = np.zeros((M*k, M*k))
            for n in range(N):
                phi_n = phi[n,:]
                outer = np.outer(phi_n, phi_n)
                for r in range(k):
                    for c in range(r, k):
                        row_offset = r*M
                        col_offset = c*M
                        d = 1 if r == c else 0
                        H[row_offset:row_offset+M,col_offset:col_offset+M] += S[n,c]*(d - S[n,r]) * outer
                        if r != c:
                            H[col_offset:col_offset+M, row_offset:row_offset+M] += H[row_offset:row_offset+M,col_offset:col_offset+M]


            del_W = phi.T @ (S - y)
            del_w = del_W.flatten("F")
            if np.allclose(del_W, np.zeros(del_W.shape), rtol=0, atol=convergence_epsilon):
                print(f"converged: {q}")
                break
            H = H + 0.01*np.eye(H.shape[0])
            H_inv = np.linalg.inv(H) # type: ignore
            W_old = W_new
            w_new = W_new.flatten("F") - H_inv @ del_w
            W_new = w_new.reshape(W_old.shape, order="F")

        self.W = W_new

    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        phi = self.phi(x)
        A: npt.NDArray[np.float64] = phi @ self.W
        S: npt.NDArray[np.float64] = softmax(A, axis=1)
        idx: npt.NDArray[np.float64] = S.argmax(axis=1)
        X: npt.NDArray[np.float64] = np.arange(S.shape[0])
        predictions: npt.NDArray[np.float64] = np.zeros(S.shape)
        predictions[X, idx] = 1.
        return predictions

    def soft_predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        phi = self.phi(x)
        A: npt.NDArray[np.float64] = phi @ self.W
        S: npt.NDArray[np.float64] = softmax(A, axis=1)
        return S

class PolynomialBasisLogisticRegression(PolynomialBasisMixin, LogisticRegression):
    pass

class GaussianBasisLogisticRegression(GaussianBasisMixin, LogisticRegression):
    pass
