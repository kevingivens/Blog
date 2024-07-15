import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score, mean_squared_error

def svi_total_variance(a, b, rho, m, sigma, forward: ArrayLike, strike: ArrayLike) -> ArrayLike:
    """
    SVI total variance formula
    TODO: add assertions
    """
    k = np.log(forward / strike)
    return a + b * (rho*(k-m) + np.sqrt((k-m)**2 + sigma**2))


def cost_function(market_tvar: ArrayLike, model_tvar: ArrayLike) -> ArrayLike:
    """ L2 norm on total variance tvar = t*vol**2"""
    return mean_squared_error(market_tvar, model_tvar)


def calibrate():
    """
    calibrate vol surface
    """
    x0 = np.array([0.5, 0])
    res = minimize(
        cost_function, 
        x0, 
        method='trust-constr', 
        jac=rosen_der, 
        hess=rosen_hess,
        constraints=[linear_constraint, nonlinear_constraint],
        options={'verbose': 1}, 
        bounds=bounds
    )
    return res