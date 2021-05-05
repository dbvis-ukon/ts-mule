"""Implementation of LIME for Time Series."""
import logging
import numpy as np
import pandas as pd
import copy

from abc import ABC, abstractclassmethod
from sklearn import linear_model
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from ..sampling import Perturbation
from ..sampling import MatrixProfileSegmentation

class Kernels:
    """Kernels for perturbation-based XAI method.

    Notice that we use scikit-learn linear-regression kernels. 
    There are two main fitting types in fitting the model: fit, and partial fit. 
    Fit means we see all samples as a whole, while partial fit (or online learning with mini batches) is 
    more benefitial in scaling. 
    
    reference:
        - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
        - https://scikit-learn.org/stable/modules/linear_model.html#linear-model
        - https://scikit-learn.org/0.15/modules/scaling_strategies.html
    """
    # Fit only
    Lasso = linear_model.Lasso(alpha=.5, fit_intercept=True)
    Ridge = linear_model.Ridge(alpha=.5, fit_intercept=True)
    
    # Fit and Partial fit
    SGDClassifier = linear_model.SGDClassifier()
    SGDRegressor = linear_model.SGDRegressor()


class AbstractXAI(ABC):
    """Abstract Module for explainable AI. """
    @abstractclassmethod
    def __init__(self) -> None:
        pass
    
    @abstractclassmethod
    def explain(self, x, predict_fn, **kwargs):
        pass

class BaseLIME(AbstractXAI):
    """Module of LIME in explaining a model."""
    
    def __init__(self, kernel=None, sampler=None, segmenter=None) -> None:
        self._kernel = kernel
        self._sampler = sampler
        self._segmenter = segmenter
        
        self.logger = logging.getLogger(f"::{self.__class__.__name__}::")
    
    @property
    def kernel(self):
        return self._kernel

    @kernel.setter
    def kernel(self, v):
        if not isinstance(v, BaseEstimator) or 'fit' not in dir(v):
            raise ValueError("The estimator not supported by sklearn.")
        self._kernel = v

    @property
    def coef(self):
        return self._kernel.coef_

    @property
    def sampler(self):
        return self._sampler

    @sampler.setter
    def sampler(self, v):
        if 'perturb' not in dir(v):
            raise ValueError("Not found perturb function in the class.")
        self._sampler = v

    @staticmethod
    def _explain(samples, kernel, predict_fn):
        # Unpack samples
        new_x, z_prime, pi = list(zip(*samples))
        
        # get the predictions
        z_hat = list(map(predict_fn, new_x))

        # Try to approximate g(z') ~ f(new_x) <=> g(z') = Z'* W ~ Z_hat
        #   or z_prime ~ X, z_hat ~ y, pi ~ sample weight (sw)
        _t = train_test_split(z_prime, z_hat, pi, test_size=0.3, random_state=42)
        X, X_test, y, y_test, sw, sw_test = _t
        
        kernel.fit(X, y, sample_weight=np.nan_to_num(sw))

        # Evaluation Score
        y_pred = kernel.predict(X_test)
        # score = kernel.score(y_test, y_pred)
        score = metrics.r2_score(y_test, y_pred)
        
        return kernel, score
    
    def explain(self, x, predict_fn, n_samples=100, **kwargs):
        # Segmentation
        seg_m = self._segmenter.segment(x)
        samples = self._sampler.perturb(x, seg_m, n_samples)
        kernel = self._kernel
        
        # Fitting into the model/kernel
        self._kernel, self.score = self._explain(samples, kernel, predict_fn)
        
        return self.coef

    def explain_instances(self, instances, predict_fn, **kwargs):
        # Todo add to use explain, and in default n_instance = 1
        #   reshape to (n_instances, n_features, n_steps)
        coef = []
        score = []
        for x in instances:
            m = self.explain(x, predict_fn, **kwargs)
            coef.append(m.coef)
            score.append(m.score)

        coef = np.stack(coef)
        score = np.stack(score)

        coef_mean = coef.mean(axis=0)
        score_mean = score.mean(axis=0)
        assert self.coef.shape == coef_mean.shape, \
            "Not same shape between 2 coefficients"

        self.xai_estimator.coef_ = coef_mean
        self.score = score_mean
        return copy.deepcopy(self)