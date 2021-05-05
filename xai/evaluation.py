"""Evaluation for different explainers."""
import numpy as np
import pandas as pd
from biokit.viz import corrplot

from . import LimeBase

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
    else:
        # inplace shuffle for each feature
        _ = np.apply_along_axis(np.random.shuffle, axis=0, arr=_m)
    return _m

def perturb_instance(x, m, r):
    assert x.shape == m.shape == r.shape
    
    # m == 1 still, m == 0 disabled and replaced
    z = x * m + r * (1 - m)
    return z


def perturb_instances(X, x_coef, method="zeros", axis=None, percentile=90, random=False, **kwargs):
    """ Perturb X based on coefficient or explanations.
        :param X: list of ndarray of shape (nfeatures, ncolumns). In time series it will be (nfeatures, nsteps)
        :param x_coef: if it is None, then random is used.
        :param axis: 
            - 1 : get percentile rowwise -> get 90 percentile per feature
            - None: consider x_coef as an array -> get 90 percentile overall
        :param method: perturbation method to generate perturbed test set.
        :param random:
            - 0/False: no random, use percentile
            - 1/True: use random in general
            - 2 : random only variables having segments changed
    """

    def _f(t, x_coef):
        # m = mask_percentile(x_coef, percentile, axis=axis, random=random)
        m = mask(x_coef, axis=axis, random=random)
        r = replacements(t, method=method, **kwargs)
        z = perturb(t, m, r) # 1: on, 0: off/disabled/perturbed
        return z

    Z = np.array([_f(x, x_coef[i]) for i, x in enumerate(X)])

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
