"""Utility functions for working with distributions and models created in R."""

import warnings
from typing import Union, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import ecdf
from sklearn.covariance import graphical_lasso as sklearn_graphical_lasso
from scipy.stats import norm

import R.univariate as R_PY


class ECDF:
    """
    class object to store emperical CDF (ECDF)
    """
    def __init__(self, data: np.array, n: int = 1000) -> None:
        self.rclass = ['ecdf']
        self.data = data
        # self.ecdf = ecdf(data)
        quants = np.linspace(0, 1, n + 1)
        self.ecdf = np.quantile(data, quants)
        self.approxfun = interp1d(quants, self.ecdf)

    def quantfun(self, data: np.array) -> np.array:
        return self.approxfun(data)

def qgpd(dist: R_PY.GPD, x: np.array) -> np.array:
    """Wrapper for Rsafd qgpd function; gets quantiles at all values in x."""

    try:
        return dist.qgpd(x)
        # return np.array(Rsafd.qgpd(dist, robjects.FloatVector(x)))

    except:
        # compute quantiles using PDF
        ll = min(dist.data)
        rr = max(dist.data)
        xx = np.linspace(ll, rr, 1001)

        # # rule=2 for extrapolation of values outside min and max
        # ff = stats.approxfun(Rsafd.pgpd(dist, xx), xx, rule=2)
        ff = interp1d(dist.pgpd(xx), xx, fill_value = (ll, rr))

        return ff(x)


def fit_gpd(data: np.array) -> Union[R_PY.GPD, ECDF]:
    """Fit a GPD if possible (fitting converge), otherwise use emperical distribution function."""

    try:
        ## try fit two tails
        dist = R_PY.fit_gpd(data, tail='two', plot=False)
        upper = dist.upper_tail.upper_converged
        lower = dist.lower_tail.lower_converged



        if upper and lower:
            return dist
        elif upper:
            return R_PY.fit_gpd(data, tail='upper', plot=False)
        elif lower:
            return R_PY.fit_gpd(data, tail='lower', plot=False)
        else:
            warnings.warn(f'no tail has been detected, using ECDF instead', RuntimeWarning)
            return ECDF(data)
        
    except:
        warnings.warn(f'unable to fit GPD, using ECDF instead', RuntimeWarning)
        return ECDF(data)


def qdist(dist: Union[R_PY.GPD, ECDF], x: np.array, gpd_max_extension: float=0.15) -> np.array:
    """Compute the quantiles of the distribution.
    If input distribution is GPD, output will be clipped. 
    """
    
    if hasattr(dist, "qgpd"):
        data_min, data_max = np.min(dist.data), np.max(dist.data)
        clip_min = data_min - gpd_max_extension * (data_max - data_min)
        clip_max = data_max + gpd_max_extension * (data_max - data_min)
        return np.clip(qgpd(dist, x), clip_min, clip_max)
    else:
        return np.quantile(dist.ecdf, x)


def standardize(table: pd.DataFrame,
                ignore_pointmass: bool = True) -> Tuple[pd.Series, pd.Series,
                                                        pd.DataFrame]:
    avg, std = table.mean(), table.std()

    if ignore_pointmass:
        std[std < 1e-2] = 1.

    else:
        if (std < 1e-2).any():
            raise RuntimeError(f'encountered point masses in '
                               f'columns {std[std < 1e-2].index.tolist()}')

    return avg, std, (table - avg) / std


def gaussianize(df: pd.DataFrame, 
                dist_dict: Optional[dict]=None, 
                gpd: bool = False) -> Tuple[dict, pd.DataFrame]:
    """Transform the data to fit a Gaussian distribution."""

    unif_df = pd.DataFrame(columns=df.columns, index=df.index)

    if dist_dict == None:
        dist_dict = dict()
        fit_dist = True
    else:
        fit_dist = False

    for col in df.columns:

        data = np.ascontiguousarray(df[col].values)

        if fit_dist:
            if gpd:
                dist_dict[col] = fit_gpd(data)
            else:
                dist_dict[col] = ECDF(data)

        if hasattr(dist_dict[col], "pgpd"):
            unif_df[col] = dist_dict[col].pgpd(data)
        else:
            unif_df[col] = R_PY.ecdf(data)

    unif_df.clip(lower=1e-5, upper=0.99999, inplace=True)
    gauss_df = unif_df.apply(norm.ppf)

    return dist_dict, gauss_df


def graphical_lasso(df: pd.DataFrame, m: int, rho: float):
    """
    Wrapper for the glasso model.

    Arguments
    ---------
        df
            The input dataset.
        m
            Number of input dimensions.
        rho
            LASSO regularization penalty.

    """
    assert df.shape[1] == m, (
        "Expected a DataFrame with {} columns, got {}".format(m, df.shape[1]))

    res = sklearn_graphical_lasso(df.cov().values, alpha=rho, max_iter=2000)
    return np.linalg.inv(res[0])


def gemini(df: pd.DataFrame,
           m: int, f: int, pA: float, pB: float) -> Tuple[np.array, np.array]:
    """
    A wrapper for the GEMINI model.

    Arguments
    ---------
        df
            The input dataset.
        m, f
            The number of spatial and temporal dimensions respectively.
        pA, pB
            The spatial and temporal regularization penalties.

    Returns
    -------
        A, B
            The spatial and temporal precision matrices.

    """
    assert df.shape[1] == m * f, (
        "Expected a DataFrame with {} columns, found {} "
        "columns instead!".format(f * m, df.shape[1])
        )

    n = len(df)
    XTX = np.zeros((m, m))
    XXT = np.zeros((f, f))

    for _, row in df.iterrows():
        X = np.reshape(row.values, (f, m), order='F')
        XTX += X.T @ X
        XXT += X @ X.T

    WA = np.diag(XTX)
    WB = np.diag(XXT)
    GA = XTX / np.sqrt(np.outer(WA, WA))
    GB = XXT / np.sqrt(np.outer(WB, WB))

    rA = sklearn_graphical_lasso(GA.cov().values, alpha=pA, max_iter=2000)
    rB = sklearn_graphical_lasso(GB.cov().values, alpha=pB, max_iter=2000)
    Arho = np.linalg.inv(rA[0])
    Brho = np.linalg.inv(rB[0])
    fact = np.sum(np.multiply(df.values, df.values)) / n

    WA = np.diag(np.sqrt(n / WA))
    WB = np.diag(np.sqrt(n / WB))
    A = np.sqrt(fact) * WA @ Arho @ WA
    B = np.sqrt(fact) * WB @ Brho @ WB

    return A, B


def split_actuals_hist_future(actual_df, timesteps, in_sample=False):
    if in_sample:
        hist_index = ~actual_df.index.isin(timesteps)
    else:
        hist_index = actual_df.index < timesteps[0]

    return actual_df[hist_index], actual_df[~hist_index]


def split_forecasts_hist_future(forecast_df, timesteps,
                                in_sample=False):
    if in_sample:
        hist_index = ~forecast_df.Forecast_time.isin(timesteps)
    else:
        hist_index = forecast_df.Forecast_time < timesteps[0]

    return forecast_df[hist_index], forecast_df[~hist_index]


def set_seed(obj):
    """
    This function sets an RNG seed to obj, if it has a seed set.
    """

    seed = getattr(obj, "seed", None)
    if not seed:
        return
    np.random.seed(seed)
    obj.seed += 1
