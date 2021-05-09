"""Implementation of LIME for Time Series."""
import logging
from abc import ABC, abstractclassmethod
import numpy as np

from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split

from ..sampling.perturb import Perturbation
from ..sampling.segment import MatrixProfileSegmentation

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

class LimeBase(AbstractXAI):
    """Module of LIME in explaining a model."""
    
    def __init__(self, kernel=None, sampler=None, segmenter=None) -> None:
        self._kernel = kernel
        self._sampler = sampler
        self._segmenter = segmenter

        self.logger = logging.getLogger(f"::{self.__class__.__name__}::")
        self._coef = None
        self._xcoef = None
        
    @property
    def segment_coef(self):
        return self._coef

    @property
    def coef(self):
        return self._xcoef
    
    @staticmethod
    def _explain(samples, kernel, predict_fn):
        # Unpack samples
        new_x, z_prime, pi = list(zip(*samples))
        # get the predictions
        z_hat = list(map(predict_fn, new_x))
        
        # Try to approximate g(z') ~ f(new_x) <=> g(z') = Z'* W ~ Z_hat
        _t = train_test_split(z_prime, z_hat, pi, test_size=0.3, random_state=42)
        X, X_test, y, y_test, sw, sw_test = _t
        
        # Avoid nan in similarity
        sw = np.nan_to_num(np.abs(sw), 0.01)
        sw_test = np.nan_to_num(np.abs(sw_test), 0.01)
        
        # Fit g(z') ~ f(new_x)
        kernel.fit(X, y, sample_weight=sw)
        
        # Evaluation Score
        y_pred = kernel.predict(X_test)
        score = metrics.r2_score(y_test, y_pred)
        
        return kernel, score
    
    def explain(self, x, predict_fn, segment_method="slopes", **kwargs):
        
        n_steps, features = x.shape
        # Get segmentation masks
        seg_m = self._segmenter.segment(x, segment_method=segment_method)
        
        # Generate samples
        samples = self._sampler.perturb(x, seg_m)
        
        # Fitting into the model/kernel
        kernel = self._kernel
        self._kernel, self.score = self._explain(samples, kernel, predict_fn)
        
        # Set coef of segments
        coef = self._kernel.coef_
        self._coef = coef.reshape(-1, features)
        
        self._xcoef =  self.to_original(coef, seg_m)
        return self._xcoef

    @staticmethod
    def to_original(coef, segments):
        """Convert coef per segment to coef per point.
        """
        x_coef = np.zeros_like(segments).astype(float)
        
        # Get labels vectors from segmentation
        seg_unique_labels = np.unique(segments)
        assert coef.shape == seg_unique_labels.shape
        
        for i, l in enumerate(seg_unique_labels):
            idx = (segments == l)
            x_coef[idx] = coef[i]
        return x_coef

class LimeTS(LimeBase):
    def __init__(self, 
                 kernel=None, 
                 segmenter=None, 
                 sampler=None, 
                 partitions=10, 
                 win_length=-1,
                 p_off=0.5,
                 replace_method="zeros",
                 n_samples=100,
                 **kwargs) -> None:
        kernel = kernel or Kernels.Lasso
        sampler = sampler or Perturbation(p_off, replace_method, n_samples)        
        segmenter = segmenter or MatrixProfileSegmentation(partitions, win_length)

        super().__init__(kernel=kernel, sampler=sampler, segmenter=segmenter)