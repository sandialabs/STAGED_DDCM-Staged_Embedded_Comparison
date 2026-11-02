# EnKF.py

import numpy as np
from abc import ABC, abstractmethod

class EnKF(ABC):
    """
    Base Ensemble Kalman Filter class with diagonal model and measurement covariance.
    """

    def __init__(self, Sigma, Gamma, H, dt_int, seed=1):
        # ----------------------------
        # Model error covariance Σ
        # ----------------------------
        # If provided as a 1-D array of variances, convert to diagonal matrix
        if Sigma.ndim == 1:
            self.Sigma = np.diag(Sigma)
        else:
            self.Sigma = Sigma  # assume full matrix

        # Build sqrtSigma = diag(sqrt(diag(Σ))) to avoid sqrtm on singular Σ
        diag_sigma = (
            np.diag(self.Sigma)
            if self.Sigma.ndim == 2
            else np.asarray(self.Sigma, float)
        )
        self.sqrtSigma = np.diag(np.sqrt(diag_sigma))

        # ----------------------------
        # Measurement error covariance Γ
        # ----------------------------
        if Gamma.ndim == 1:
            self.Gamma = np.diag(Gamma)
        else:
            self.Gamma = Gamma

        # Precompute sqrtGamma = diag(sqrt(diag(Γ)))
        diag_gamma = (
            np.diag(self.Gamma)
            if self.Gamma.ndim == 2
            else np.asarray(self.Gamma, float)
        )
        self.sqrtGamma = np.diag(np.sqrt(diag_gamma))

        # Observation operator
        self.H = H

        # Forecast‐step integration timestep
        self.dt_int = dt_int

        # Dimensions
        self.num_inputs = self.Sigma.shape[0]
        self.num_outputs = self.Gamma.shape[0]

        # Evaluation counter and RNG
        self.num_eval = 0
        self.rng = np.random.default_rng(seed)

    def set_Sigma(self, Sigma):
        """
        Reset Σ and rebuild sqrtSigma. Used in likelihood estimation.
        """
        if Sigma.ndim == 1:
            self.Sigma = np.diag(Sigma)
        else:
            self.Sigma = Sigma

        diag_sigma = (
            np.diag(self.Sigma)
            if self.Sigma.ndim == 2
            else np.asarray(self.Sigma, float)
        )
        self.sqrtSigma = np.diag(np.sqrt(diag_sigma))

        if self.Sigma.shape[0] != self.num_inputs:
            raise ValueError(
                f"Sigma dimension {self.Sigma.shape[0]} != expected {self.num_inputs}"
            )

    def get_model_evaluations(self):
        """Number of ensemble‐forecast evaluations done so far."""
        return self.num_eval

    @abstractmethod
    def _ODEs(self, t_cur, x_cur):
        """
        Define the model ODEs. To be implemented by subclasses.
        """
        pass

    @abstractmethod
    def model_solve(self, x_cur, t_cur, t_next):
        """
        One‐step ODE solver from t_cur to t_next. To be implemented by subclasses.
        """
        pass

    def data_assimilation(
        self, x0, d, t, t_obs, N_ensembles=50, xa_init=None, smoother=False
    ):
        """
        Run Ensemble Kalman Filter assimilation.

        Returns (t_samples, x_mu, x_sd, x_ens).
        """
        num_obs = len(t_obs)
        num_forecast = len(t)
        n_eval = num_forecast + num_obs + 1

        x_mu = np.zeros((n_eval, self.num_inputs))
        x_sd = np.zeros((n_eval, self.num_inputs))
        x_ens = np.zeros((n_eval, N_ensembles, self.num_inputs))
        t_samples = np.zeros(n_eval)

        # Initialize analysis ensemble
        if xa_init is None:
            xa = x0 * np.ones((N_ensembles, self.num_inputs))
        else:
            xa = xa_init
        xf = xa.copy()
        x_ens[0] = xa

        idx_eval = 0
        idx_fore = 0
        idx_obs = 0

        x_mu[idx_eval] = xa.mean(axis=0)
        x_sd[idx_eval] = xa.std(axis=0)
        t_samples[idx_eval] = t[0]
        idx_eval += 1
        t_cur = t[0]

        while idx_fore < num_forecast or idx_obs < num_obs:
            t_f = t[idx_fore] if idx_fore < num_forecast else np.inf
            t_o = t_obs[idx_obs] if idx_obs < num_obs else np.inf
            t_next = min(t_f, t_o)

            # Forecast step
            xf = self._forecast_step(xa, xf, t_cur, t_next, N_ensembles)

            # Record forecast
            if idx_fore < num_forecast and np.isclose(t_next, t[idx_fore], rtol=1e-5):
                x_mu[idx_eval] = xf.mean(axis=0)
                x_sd[idx_eval] = xf.std(axis=0)
                x_ens[idx_eval] = xf
                t_samples[idx_eval] = t_next
                idx_fore += 1
                idx_eval += 1

            # Analysis step if observation is available
            if idx_obs < num_obs and np.isclose(t_next, t_obs[idx_obs], rtol=1e-5):
                if not smoother:
                    xa = self._analysis_step(xf, d[:, idx_obs], N_ensembles)
                    x_mu[idx_eval] = xa.mean(axis=0)
                    x_sd[idx_eval] = xa.std(axis=0)
                    x_ens[idx_eval] = xa
                else:
                    x_ens[idx_eval] = xf
                    x_ens = self._smoothing_step(
                        x_ens, xf, d[:, idx_obs], N_ensembles, idx_eval
                    )
                    x_mu[: idx_eval + 1] = x_ens[: idx_eval + 1].mean(axis=1)
                    x_sd[: idx_eval + 1] = x_ens[: idx_eval + 1].std(axis=1)
                    xa = x_ens[idx_eval]

                t_samples[idx_eval] = t_next
                idx_obs += 1
                idx_eval += 1
            else:
                xa = xf

            t_cur = t_next

        self.num_eval += N_ensembles
        return t_samples, x_mu, x_sd, x_ens

    def _forecast_step(self, xa, xf, t_cur, t_next, N_ensembles):
        """
        Propagate each ensemble member from t_cur to t_next, adding model noise.
        """
        for i in range(N_ensembles):
            x_t = xa[i].copy()
            ti = t_cur
            while ti < t_next and abs(ti - t_next) > 1e-7:
                dt = min(self.dt_int, t_next - ti)
                x_t = self.model_solve(x_t, ti, ti + dt)
                rand = self.rng.normal(0, 1, size=self.num_inputs)
                x_t += np.sqrt(dt) * (self.sqrtSigma @ rand)
                ti += dt
            xf[i] = x_t
        return xf

    def _analysis_step(self, xf, d_obs, N_ensembles):
        """
        Standard EnKF analysis: perturb observations with sqrtGamma.
        """
        A = xf - xf.mean(axis=0, keepdims=True)
        C = (A.T @ A) / (N_ensembles - 1)

        D = (
            d_obs[np.newaxis, :]
            + self.rng.normal(0, 1, size=(N_ensembles, self.num_outputs))
            @ self.sqrtGamma
        )
        gamma_pert = D - d_obs[np.newaxis, :]
        R = (gamma_pert.T @ gamma_pert) / (N_ensembles - 1)

        CHt = C @ self.H.T
        K = CHt @ np.linalg.inv(self.H @ CHt + R)
        return xf + (K @ (D.T - self.H @ xf.T)).T

    def _smoothing_step(self, x_ens, xf, d_obs, N_ensembles, idx_eval):
        """
        Rauch–Tung–Striebel smoother on the ensemble.
        """
        A_f = xf - xf.mean(axis=0, keepdims=True)
        HA_f = A_f @ self.H.T

        D = (
            d_obs[np.newaxis, :]
            + self.rng.normal(0, 1, size=(N_ensembles, self.num_outputs))
            @ self.sqrtGamma
        )
        gamma_pert = D - d_obs[np.newaxis, :]
        R = (gamma_pert.T @ gamma_pert) / (N_ensembles - 1)

        P = np.linalg.inv((HA_f.T @ HA_f) / (N_ensembles - 1) + R)
        K_post = (HA_f @ P) / (N_ensembles - 1)
        innovation = (D.T - self.H @ xf.T)

        for k in range(idx_eval + 1):
            A_past = x_ens[k] - x_ens[k].mean(axis=0, keepdims=True)
            K_past = A_past.T @ K_post
            x_ens[k] += (K_past @ innovation).T
        return x_ens

    def compute_log_likelihood(
        self, x0, t0, d, t_obs, Sigma, N_ensembles=50, sample=True
    ):
        """
        Monte Carlo or DA‐based log‐likelihood. Calls set_Sigma internally.
        """
        self.set_Sigma(Sigma)

        xa = x0 * np.ones((N_ensembles, self.num_inputs))
        xf = np.zeros_like(xa)

        log_like = np.zeros(len(t_obs))
        t_cur = t0
        obs_i = 0

        while obs_i < len(t_obs):
            t_next = t_obs[obs_i]
            xf = self._forecast_step(xa, xf, t_cur, t_next, N_ensembles)

            if np.isclose(t_next, t_obs[obs_i], rtol=1e-5):
                if sample:
                    log_like[obs_i] = self._sample_log_likelihood(
                        xf, d[:, obs_i], N_ensembles
                    )
                else:
                    log_like[obs_i] = self._DA_estimate_log_likelihood(
                        xf, d[:, obs_i], N_ensembles
                    )
                xa = self._analysis_step(xf, d[:, obs_i], N_ensembles)
                obs_i += 1
            else:
                xa = xf

            t_cur = t_next

        self.num_eval += N_ensembles
        return log_like

    def _sample_log_likelihood(self, xf, d_obs, N_ensembles):
        """
        Compute log‐likelihood by Monte Carlo sampling of perturbed observations.
        """
        R = self.Gamma
        prefactor = (2 * np.pi) ** (-self.num_outputs / 2) * np.linalg.det(
            R
        ) ** (-0.5)
        like_array = np.zeros(N_ensembles)
        Rinv = np.linalg.inv(R)
        for i in range(N_ensembles):
            diff = d_obs - (self.H @ xf[i, :])
            like_array[i] = np.exp(-0.5 * diff.T @ Rinv @ diff)
        return np.log(prefactor * like_array.mean())

    def _DA_estimate_log_likelihood(self, xf, d_obs, N_ensembles):
        """
        Compute log‐likelihood via analytic EnKF posterior covariance.
        """
        R = self.Gamma
        # forecast mean in observation space
        x_mean = self.H @ xf.mean(axis=0)
        # forecast covariance in observation space
        x_var = self.H @ np.cov(xf, rowvar=False) @ self.H.T
        C = R + x_var
        diff = d_obs - x_mean
        Cinv = np.linalg.inv(C)
        sign, logdet = np.linalg.slogdet(C)
        return (
            -0.5 * self.num_outputs * np.log(2 * np.pi)
            - 0.5 * logdet
            - 0.5 * diff.T @ Cinv @ diff
        )