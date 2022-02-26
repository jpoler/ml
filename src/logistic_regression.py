import logging
import numpy as np
import numpy.typing as npt
from scipy.special import softmax # type: ignore
from typing import Any

from constants import convergence_epsilon
from model import Model
from fixed_basis import FixedBasisFunctionMixin, GaussianBasisMixin, PolynomialBasisMixin

class LogisticRegression(FixedBasisFunctionMixin, Model):
    def __init__(self, k_classes: int, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.k = k_classes

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        N = len(X)
        if len(X.shape) == 1:
            X = X.reshape(len(X), 1)
        if N == 0:
            logging.warn("empty input to fit")
            return
        phi = self.phi(X)
        # print(f"phi: {phi}")
        M = phi.shape[1]
        self.W = np.random.random((M, self.k))
        # print(f"phi.shape: {phi.shape}")
        # print("before np.ones")
        W_old: npt.NDArray[np.float64] = np.ones(self.W.shape) * np.inf
        W_new = self.W
        # print("before loop")
        q = 0
        while not np.allclose(W_old, W_new, rtol=0., atol=convergence_epsilon):
            # print(f"q: {q}")
            q += 1
            # print(f"W_old: {W_old}")
            # print(f"W_new: {W_new}")
            # print(f"W_new - W_old: {W_new - W_old}")
            A: npt.NDArray[np.float64] = phi @ W_new
            S = softmax(A, axis=1)
            # print(f"A:\n{np.array_str(A, precision=1, suppress_small=True)}")
            # print(f"S:\n{np.array_str(S, precision=1, suppress_small=True)}")
            # print(f"y:\n{np.array_str(y, precision=1, suppress_small=True)}")
            # print(f"S - y:\n{np.array_str(S - y, precision=1, suppress_small=True)}")
            H = np.zeros((M*self.k, M*self.k))
            for n in range(N):
                phi_n = phi[n,:]
                # print(f"phi_n.shape: {phi_n.shape}")
                outer = np.outer(phi_n, phi_n)
                # print(f"outer {outer}")
                for r in range(self.k):
                    for c in range(self.k):
                        row_offset = r*M
                        col_offset = c*M
                        d = 1 if r == c else 0
                        # print(f"S[n,c]: {S[n,c]}, S[n,r]: {S[n,r]}, product: {S[n,c]*(d - S[n,r])}")
                        H[row_offset:row_offset+M,col_offset:col_offset+M] += S[n,c]*(d - S[n,r]) * outer
                        # print(f"H:\n{np.array_str(H, precision=1, suppress_small=True)}")
                        # if r != c:
                        #     H[col_offset:col_offset+M, row_offset:row_offset+M] += H[row_offset:row_offset+M,col_offset:col_offset+M]


            del_W = phi.T @ (S - y)
            del_w = del_W.flatten("F")
            if np.allclose(del_W, np.zeros(del_W.shape)):
                print("converged")
                return
            # print(f"del_w:\n{np.array_str(del_w, precision=1, suppress_small=True)}")
            # print(f"H:\n{np.array_str(H, precision=10, suppress_small=True)}")
            H = H + np.eye(H.shape[0])
            H_inv = np.linalg.inv(H) # type: ignore
            # print(f"H_inv:\n{np.array_str(H_inv, precision=1, suppress_small=True)}")
            W_old = W_new
            w_new = W_new.flatten("F") - H_inv @ del_w
            W_new = w_new.reshape(W_old.shape, order="F")

        # print("after loop")
        self.W = W_new

    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        phi = self.phi(x)
        A: npt.NDArray[np.float64] = phi @ self.W
        S: npt.NDArray[np.float64] = softmax(A, axis=1)
        M: npt.NDArray[np.float64] = S.argmax(axis=1)
        return M

class PolynomialBasisLogisticRegression(PolynomialBasisMixin, LogisticRegression):
    pass

class GaussianBasisLogisticRegression(GaussianBasisMixin, LogisticRegression):
    pass
