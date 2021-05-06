"""Evaluation for different explainers."""
import copy
import numpy as np
import pandas as pd
from biokit.viz import corrplot

from . import LimeBase
from sampling import replace as repl

def mask_percentile(x, upper_percentile=90, lower_percentile=10):
    # 1/on/keep and 0/off/disabled
    # n_steps, features = x
    upper_percentile = 90
    
    upper = np.percentile(x, upper_percentile, axis=0)
    lower = np.percentile(x, lower_percentile, axis=0)
    
    # Get important points if 
    #   x larger than 90 percentile if positives
    #   x smaller than 10 percentile if negatives
    m = (x.round(2) != 0)
    m *= ((x > upper) * (x > 0) + (x < lower) * (x < 0))
    
    # reverse to have 1 = on, 0 = off
    m = 1 - m   
    assert m.shape == x.shape, "Not matching shape between mask and x"
    
    return m


def mask_random(m, method="all"):
    # Random the masked percentile-90. 
    _m = np.array(m)
    if method == "all":
        # shuffle all values, without regarding to features
        _m = _m.ravel()
        np.random.shuffle(_m)
        _m = _m.reshape(m.shape)
    if method == "each":
        # inplace shuffle for each feature
        _ = np.apply_along_axis(np.random.shuffle, axis=0, arr=_m)
    return _m

def perturb_instance(x, m, replace_method="zeros"):
    """Perturb and instance in percentile 90/10. 

    Args:
        x (ndarray): time series with shape (n_steps, features)
        m (ndarray): a masking of 1s and 0s of x in shape of (n_steps, features)
        replace_method (str, optional): Replacement methods to replace disabled points. Defaults to "zeros".
            - All methods is built-in functions in sampling.replace module.

    Returns:
        ndarray: a new perturbed instance of x.
    """
    repl_fn = getattr(repl, replace_method)
    r = repl_fn(x)
    assert x.shape == m.shape == r.shape
    z = x * m + r * (1 - m)
    return z


def perturb_instances(X, relevance, replace_method="zeros", random_method=None, **kwargs):
    """Perturb multiple instances X, given their relevance/explainations.
    
        :param X: list of ndarray of shape (n_steps, features)
        :param relevance: if it is None, then random is used.
        :param method: replacement method for disabled/off relevance at point t
        :param random_method: ("all" | "each") random method on each feature or all. Default None
            For all, an instance x in X is raveled, shorted, and then reshaped back.
    """
    for x in X:
        m = mask_percentile(x)
        z = perturb_instance(x, method=replace_method, **kwargs)

    def _f(t, x_coef):
        # m = mask_percentile(x_coef, percentile, axis=axis, random=random)
        m = mask(x_coef, axis=axis, random=random)
        r = replacements(t, method=replace_method, **kwargs)
        z = perturb(t, m, r) # 1: on, 0: off/disabled/perturbed
        return z

    Z = np.array([_f(x, relevance[i]) for i, x in enumerate(X)])

    return Z


def corr_matrix(coef_or_models, names=None, **corr_kwargs):
    assert isinstance(coef_or_models, list)
    if all([isinstance(m, LimeBase) for m in coef_or_models]):
        coef = [m.coef for m in coef_or_models]
    else:
        coef = coef_or_models

    if names is None:
        names = [i for i in range(len(coef))]

    assert len(names) == len(coef), \
        f"Not matching length of names and coef_or_models, {len(names)} and {len(coef)}"

    df = pd.DataFrame({n: c for n, c in zip(names, coef)})
    return df.corr(**corr_kwargs)


def plot_corr(df_corr, method='square'):
    c = corrplot.Corrplot(df_corr)
    c.plot(method=method, shrink=.9, rotation=45)
    # c.plot(method=method, fontsize=8, colorbar=False)
