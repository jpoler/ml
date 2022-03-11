import numpy as np
import numpy.typing as npt
from scipy.special import softmax # type: ignore
from typing import Any, List

from constants import convergence_epsilon
from model import Model

class NeuralNetwork(Model):
    def __init__(self, hidden_units: List[int], batch_size: int = 1, learning_rate: float = 1., max_iterations: int = 1000) -> None:
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_units = hidden_units
        self.max_iterations = max_iterations


    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        n, x_len = X.shape
        Ws = [np.random.random((self.hidden_units[0], x_len + 1))]
        if len(self.hidden_units) > 1:
            for i in range(len(self.hidden_units)-1):
                Ws.append(np.random.random((self.hidden_units[i+1], self.hidden_units[i] + 1)))
        Ws.append(np.random.random((y.shape[1], self.hidden_units[-1] + 1)))
        print(f"Ws: {[W.shape for W in Ws]}")
        w_len = sum(W.size for W in Ws)
        for batch in range(self.max_iterations):
            # feed forward
            sample_idx = np.random.randint(n, size=self.batch_size)
            A = np.concatenate([X[sample_idx, :], np.ones((self.batch_size, 1))], axis=1) # type: ignore
            T = y[sample_idx, :]
            Zs = [A]
            As = [A]
            print(f"Z: {Zs[0]}")
            print(f"A: {As[0]}")
            for W in Ws:
                Z: npt.NDArray[np.float64] = np.concatenate([A @ W.T, np.ones((self.batch_size, 1))], axis=1) # type: ignore
                # ReLU
                A = Z * (Z > 0.)
                Zs.append(Z)
                As.append(A)
                print(f"Z: {Z}")
                print(f"A: {A}")
            S = softmax(As[-1][:, :-1], axis=1)
            print(f"S: {S}")

            # backpropogate
            del_Ws = [np.zeros(W.shape, dtype=np.float64) for W in Ws]
            for i in range(T.shape[0]):
                # deltas
                t = T[i, :].T
                s = S[i, :].T
                # D = np.zeros(t.shape[0], dtype=np.float64)
                D = s - t
                # ts = np.multiply(t, s)
                # for k in range(t.shape[0]):
                #     Dk = ts[:]
                #     Dk[k] -= t[k]
                #     D[k] = Dk.sum()
                print(f"D: {D}")
                for j in reversed(range(len(del_Ws))):
                    # print(f"j {j}")
                    W = Ws[j]
                    a = As[j][i, :-1]
                    diag_D: npt.NDArray[np.float64] = np.diag(D) # type: ignore
                    # print(f"Zs[j][i, :] {Zs[j][i, :].shape}")
                    stacked_Z: npt.NDArray[np.float64] = np.tile(Zs[j][i, :], (len(D), 1)) # type: ignore
                    # print(f"del_Ws[j]: {del_Ws[j].shape} diag_D: {diag_D.shape}, stacked_Z {stacked_Z.shape}")
                    del_Ws[j] += diag_D @ stacked_Z
                    # print(f"W shape {W.shape}, diag_D shape: {diag_D.shape}")
                    product: npt.NDArray[np.float64] = diag_D @ W[:, :-1]
                    # print(f"product {product.shape} a {a.shape}")
                    D = product.sum(axis=0) * (a > 0)

            # if all(np.allclose(del_W,
            #                    np.zeros(del_W.shape, dtype=np.float64),
            #                    rtol=0.,
            #                    atol=convergence_epsilon) for del_W in del_Ws):
            #     print(f"converged after {batch} batches")
            #     break
            # print(f"del_Ws\n{del_Ws}")
            Ws = [W - self.learning_rate*del_W for W, del_W in zip(Ws, del_Ws)]
            # print(f"Ws\n{Ws}")

        self.Ws = Ws
        # print(f"self.Ws: {self.Ws}")



    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        S = self.soft_predict(x)
        idx: npt.NDArray[np.float64] = S.argmax(axis=1)
        X: npt.NDArray[np.float64] = np.arange(S.shape[0])
        predictions: npt.NDArray[np.float64] = np.zeros(S.shape)
        predictions[X, idx] = 1.
        return predictions



    def soft_predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        A = np.concatenate([x, np.ones((x.shape[0], 1))], axis=1) # type: ignore
        for W in self.Ws:
            Z: npt.NDArray[np.float64] = np.concatenate([A @ W.T, np.ones((x.shape[0], 1))], axis=1) # type: ignore
            # ReLU
            A = Z * (Z > 0.)
        S: npt.NDArray[np.float64] = softmax(A[:, :-1], axis=1)
        return S
