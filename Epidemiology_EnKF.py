import numpy as np
from scipy.integrate import solve_ivp

from EnKF import EnKF

class Epidemiology_EnKF(EnKF):
    def __init__(
        self,
        Sigma,
        Gamma,
        H,
        dt_int,
        model_params=(None,),
        caseid=None,
    ):
        super().__init__(Sigma, Gamma, H, dt_int)

        self.model_params = model_params

        if caseid is None:
            self.caseid = 1
        else:
            self.caseid = caseid

    def _ODEs(self, t_cur, x_cur):
        """
        Epidemiology ODEs for augmented SIQR state:

            x = [S, I, Q, R, xi]
        """
        beta, gamma, delta = self.model_params
        S, I, Q, R, xi = x_cur

        dSdt = -beta * S * I

        if self.caseid == 1:
            dIdt = beta * S * I - gamma * I - xi * I
            dQdt = xi * I - delta * Q
        elif self.caseid == 2:
            dIdt = beta * S * I - gamma * I - xi
            dQdt = xi - delta * Q
        else:
            raise ValueError(f"Unknown epidemiology caseid: {self.caseid}")

        dRdt = gamma * I + delta * Q
        dXidt = 0.0

        return dSdt, dIdt, dQdt, dRdt, dXidt

    def model_trajectory(self, x0, t_span, t_eval=None):
        """
        Forward solve the ODE for a time span and return all model evaluations.
        """
        soln = solve_ivp(
            self._ODEs,
            t_span,
            x0,
            method="RK45",
            t_eval=t_eval,
        )

        return soln.y

    def model_solve(self, x_cur, t_cur, t_next):
        """
        Forward Euler solve from t_cur to t_next.

        This intentionally keeps the ODE timestep equal to the EnKF model-error
        injection timestep.
        """
        derivs = np.array(self._ODEs(t_cur, x_cur), dtype=float)

        return x_cur + (t_next - t_cur) * derivs