import logging
import numpy as np
import numpy.typing as npt
from scipy.stats import norm, multivariate_normal # type: ignore
from typing import Any, Callable, Optional, Tuple

from constants import convergence_epsilon
from model import GaussianBayesianModel, Model
from fixed_basis import FixedBasisFunctionMixin, GaussianBasisMixin, PolynomialBasisMixin

class LeastSquaresRegression(FixedBasisFunctionMixin, Model):
    def __init__(self, regularization_coefficient: Optional[float] = None, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.regularization_coefficient = regularization_coefficient

    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        phi = self.phi(X)
        if self.regularization_coefficient:
            inner = self.regularization_coefficient * np.eye(phi.T.shape[0]) + phi.T @ phi # type: ignore
        else:
            inner = phi.T @ phi
        inv = np.linalg.inv(inner) # type: ignore
        self.W: npt.NDArray[np.float64] = inv @ phi.T @ y

    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        phi = self.phi(x)
        return self.W @ phi.T

class PolynomialBasisLeastSquaresRegression(PolynomialBasisMixin, LeastSquaresRegression):
    pass

class GaussianBasisLeastSquaresRegression(GaussianBasisMixin, LeastSquaresRegression):
    pass

class BayesianLinearRegression(FixedBasisFunctionMixin, GaussianBayesianModel):
    def __init__(self, alpha: float = 1., beta: float = 1., convergence_threshold: float = convergence_epsilon, max_evidence_iterations: Optional[int] = 100, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.alpha: np.float64 = np.float64(alpha)
        self.beta: np.float64 = np.float64(beta)
        self.n: int = 0
        self.y_t_y: np.float64 = np.float64(0.)
        self.phi_t_y: npt.NDArray[np.float64] = np.zeros(self.basis_dimensionality)
        self.phi_t_phi: npt.NDArray[np.float64] = np.zeros((self.basis_dimensionality, self.basis_dimensionality))
        self.w_mean = self.initial_prior_mean()
        self.w_precision = self.initial_prior_precision(self.alpha)
        self.w_covariance: npt.NDArray[np.float64] = np.linalg.inv(self.w_precision) # type: ignore
        self.convergence_threshold = convergence_threshold
        self.max_evidence_iterations = max_evidence_iterations

    def gamma(self, alpha: np.float64, data_precision: npt.NDArray[np.float64]) -> np.float64:
        eigenvalues, _ = np.linalg.eig(data_precision) # type: ignore
        ratio: Callable[[np.float64], np.float64] = lambda v: v / (alpha + v)
        gamma: np.float64 = sum(map(ratio, eigenvalues)) # type: ignore
        return gamma

    def initial_prior_precision(self, alpha: np.float64) -> npt.NDArray[np.float64]:
        precision: npt.NDArray[np.float64] = alpha * np.eye(self.basis_dimensionality)
        return precision

    def initial_prior_mean(self) -> npt.NDArray[np.float64]:
        return np.zeros(self.basis_dimensionality)

    def update_posterior(
            self,
            alpha: np.float64,
            beta: np.float64,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        initial_prior = self.initial_prior_precision(alpha)
        w_precision_posterior: npt.NDArray[np.float64] = initial_prior +  beta * self.phi_t_phi
        w_covariance_posterior: npt.NDArray[np.float64] = np.linalg.inv(w_precision_posterior) # type: ignore
        w_mean_posterior: npt.NDArray[np.float64] = beta * w_covariance_posterior @ self.phi_t_y
        return w_mean_posterior, w_covariance_posterior, w_precision_posterior


    def fit(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> None:
        if len(X) == 0:
            print("no data provided, returning")
            return
        phi = self.phi(X)

        # With new data, it is cheap (assuming low dim basis) to accumulate phi
        # which is MXM, where we just add the outer product of each phi(x_n) @
        # phi(x_n).T. Same for keeping track of phi.T @ y, which is m x 1, so
        # just add new information and then compute posterior mean and
        # precision from the initial prior. First update y_t_y, phi_t_y, and
        # phi_t_phi with new data.
        self.n += len(y)
        self.y_t_y += y.T @ y
        self.phi_t_y +=  phi.T @ y
        self.phi_t_phi += phi.T @ phi

        # Do one intial update that will be kept regardless of whether evidence
        # reestimation converges.
        self.w_mean, self.w_covariance, self.w_precision = self.update_posterior(self.alpha, self.beta)
        self.reestimate_evidence()


    def reestimate_evidence(self) -> None:
        if not self.max_evidence_iterations:
            return
        if self.n == 0:
            logging.warning("skipping reestimation, no data points")
            return
        alpha, beta = self.alpha, self.beta
        w_mean, w_covariance, w_precision = self.w_mean, self.w_covariance, self.w_precision
        for i in range(self.max_evidence_iterations):
            gamma = self.gamma(alpha, beta*self.phi_t_phi)
            squared_mean_length: float = w_mean.T @ w_mean
            alpha_old = alpha
            alpha = gamma / squared_mean_length
            beta_old = beta
            # Note that this is the quadratic expansion of ||y - phi.T @ w_mean||^2,
            # which allows us to take advantage of the low(er) dimensionality
            # of the projected y_t_y, phi_t_y, and phi_t_phi, so we don't have
            # to store N X M phi or N x 1 y.
            distance_term: np.float64 = self.y_t_y - 2.*self.phi_t_y.T @ w_mean + w_mean.T @ self.phi_t_phi @ w_mean
            beta_inverse = (1 / (self.n - gamma)) *  distance_term
            beta = 1 / beta_inverse
            alpha_converged = abs(alpha - alpha_old) < self.convergence_threshold
            beta_converged = abs(beta - beta_old) < self.convergence_threshold
            # Now reestimate mean, covariance, and precision
            w_mean, w_covariance, w_precision = self.update_posterior(alpha, beta)

            if alpha_converged and beta_converged:
                # TODO set log level in tests and make this info
                logging.warning("converged after %d iterations", i)
                self.alpha = alpha
                self.beta = beta
                self.w_mean = w_mean
                self.w_precision = w_precision
                self.w_covariance = w_covariance
                return
        else:
            logging.warning("evidence approximation failed to converge after %d iterations", self.max_evidence_iterations)


    def predict(self, x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        samples = []
        for i in range(len(x)):
            phi = self.phi(x[i:i+1])
            predictive_mean: npt.NDArray[np.float64] = phi @ self.w_mean
            predictive_covariance = (1 / self.beta) + phi @ self.w_covariance @ phi.T
            sample: npt.NDArray[np.float64] = multivariate_normal.rvs(mean=predictive_mean, cov=predictive_covariance)
            samples.append(sample)
        return np.array(samples)

    def posterior_mean(self) -> npt.NDArray[np.float64]:
        return self.w_mean

    def posterior_covariance(self) -> npt.NDArray[np.float64]:
        return self.w_covariance

    def posterior_probability(self, w: npt.NDArray[np.float64]) -> float:
        p: float = multivariate_normal.pdf(w, mean=self.w_mean, cov=self.w_covariance)
        return p

    def sample_posterior_parameters(self) -> npt.NDArray[np.float64]:
        sample: npt.NDArray[np.float64] = multivariate_normal.rvs(mean=self.w_mean, cov=self.w_covariance)
        return sample


    def likelihood_probability(self, X: npt.NDArray[np.float64], y: npt.NDArray[np.float64], w: npt.NDArray[np.float64]) -> float:
        if w is None:
            w = self.w_mean
        phi = self.phi(X)
        locs: npt.NDArray[np.float64] = phi @ w
        p = 1.
        for i in range(len(locs)):
            loc = locs[i]
            t = y[i]
            p *= norm.pdf(t, loc, 1./self.beta)
        return p


    def predictive_probability(self, t: npt.NDArray[np.float64], x: npt.NDArray[np.float64]) -> float:
        phi = self.phi(x)
        predictive_mean: npt.NDArray[np.float64] = self.w_mean.T @ phi
        predictive_covariance = (1 / self.beta) + phi.T @ self.w_precision @ phi
        p: float = multivariate_normal(t, mean=predictive_mean, cov=predictive_covariance)
        return p

class GaussianBasisBayesianLinearRegression(GaussianBasisMixin, BayesianLinearRegression):
    pass

class PolynomialBasisBayesianLinearRegression(PolynomialBasisMixin, BayesianLinearRegression):
    pass
