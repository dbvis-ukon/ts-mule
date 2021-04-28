"""Implementation of LIME for Time Series."""
import logging
import numpy as np
import pandas as pd
import copy

from abc import ABC
from sklearn import linear_model
from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split

from ..perturbation import (TimeSeriesPerturbation, 
                            SyncTimeSlicer,
                            ASyncTimeSlicer)


class XAIModels:
    """Supporting Estimators for XAI.

    reference:
        - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
        - https://scikit-learn.org/stable/modules/linear_model.html#linear-model
    """
    # Regression (Forecasting)
    Lasso = linear_model.Lasso(alpha=.5, fit_intercept=True)

    # Classifier
    Ridge = linear_model.Ridge(alpha=.5, fit_intercept=True)


class LIMEAbstract(ABC):
    """Abstract module of LIME which include all methods needs to implemented."""

    def __init__(self, sample_size=100, **_kwargs):
        self.sample_size = sample_size
        self.logger = logging.getLogger(self.__class__.__name__)

        self._xai_estimator = None
        self._perturbator = None

        self._z_prime = []
        self._z = []
        self._z_hat = []
        self._sample_weight = []

    def explain(self, x, predict_fn, **kwargs):
        raise NotImplementedError()

    @property
    def xai_estimator(self):
        return self._xai_estimator

    @xai_estimator.setter
    def xai_estimator(self, v):
        if not isinstance(v, BaseEstimator) or 'fit' not in dir(v):
            raise ValueError("The estimator not supported by sklearn.")
        self._xai_estimator = v

    @property
    def coef(self):
        return self._xai_estimator.coef_

    @property
    def perturb_obj(self):
        return self._perturbator

    @perturb_obj.setter
    def perturb_obj(self, v):
        if 'perturb' not in dir(v):
            raise ValueError("Not found perturb function in the class.")
        self._perturbator = v


class LIMETimeSeries(LIMEAbstract):
    """LIME for time series witch time slicing."""

    def __init__(self, scale='async', perturb_method='zeros',
                 window_size=3, off_prob=0.5, sample_size=10, **kwargs):
        super().__init__(sample_size, **kwargs)

        # MTS Time Series Initialization
        self.scale = scale
        self.window_size = window_size
        self.n_steps = 0
        self.n_segments = 0
        self.n_features = 0

        # General perturbation Initialization
        self.perturb_method = perturb_method
        self.off_p = off_prob

        self.xai_estimator = XAIModels.Lasso
        self.score = np.nan

        # Set-up Perturbator
        if scale == "async":
            self.perturb_obj = ASyncTimeSlicer(window_size, off_prob, perturb_method)
        elif scale == 'sync':
            self.perturb_obj = SyncTimeSlicer(window_size, off_prob, perturb_method)
        else:
            ValueError(f"Scale {scale} currently is not supported.")

    def _get_samples(self, x, predict_fn, sample_size, **kwargs):
        samples = self._perturbator.perturb(x, n_samples=sample_size, **kwargs)

        z_prime = []
        z = []
        z_hat = []
        sample_weight = []

        # Todo: any way of using fit as generators to save memory?
        for _prime, _z, _pi_z in samples:
            z_prime.append(_prime)
            z.append(_z)
            z_hat.append(predict_fn(_z))
            sample_weight.append(_pi_z)
        return z_prime, z, z_hat, sample_weight

    def explain(self, x, predict_fn, **kwargs):
        assert np.ndim(x) == 2, \
            "Only 2 dimension accepted. If univariate time series please use np.reshape(-1, 1)"
        self.n_features, self.n_steps = x.shape
        self.n_segments = (self.n_steps // self.window_size) + int(bool(self.n_steps % self.window_size))

        self._z_prime, self._z, self._z_hat, self._sample_weight = self._get_samples(x,
                                                                                     predict_fn,
                                                                                     self.sample_size,
                                                                                     **kwargs)
        z_prime = np.stack(self._z_prime)
        z_hat = np.stack(self._z_hat)
        weight = np.stack(self._sample_weight)

        X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
            z_prime,
            z_hat,
            weight,
            test_size=0.3,
            random_state=42
        )

        # Fit to XAI estimator
        self.xai_estimator.fit(X_train, y_train, sw_train)
        self.logger.info("Updated xai estimator.")

        # Evaluate XAI estimator
        # Reference: https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
        y_pred = self.xai_estimator.predict(X_test)
        # self.score = metrics.mean_squared_error(y_test, y_pred, sw_test)
        self.score = metrics.r2_score(y_test, y_pred)

        return copy.deepcopy(self)

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

    def plot_coef(self, feature_names=None, scaler=None, **kwargs):
        coef = self.xai_estimator.coef_.copy()
        if self.scale == 'async':
            coef = coef.reshape(self.n_segments, self.n_features)
        coef_df = pd.DataFrame(coef)
        if feature_names:
            coef_df.columns = feature_names
        if scaler:
            scaler.fit(coef_df.values)
            coef_df = pd.DataFrame(data=scaler.transform(coef_df.values),
                                   columns=coef_df.columns)
        kwargs['kind'] = kwargs.get('kind') or 'bar'
        kwargs['subplots'] = kwargs.get('subplots') or 1
        coef_df.plot(**kwargs)

    def get_a_local_sample(self):
        if len(self._z_prime) > 0:
            idx = np.random.choice(self.sample_size)
            z_prime = self._z_prime[idx]
            if self.scale == 'sync':
                z_prime = np.broadcast_to(z_prime, (self.n_features, len(z_prime)))
            else:
                z_prime = z_prime.reshape(self.n_features, self.n_segments)
            z = self._z[idx].reshape(self.n_features, self.n_steps)
            z_hat = self._z_hat[idx]
            w = self._sample_weight[idx]
            return (z_prime, z, z_hat, w)
