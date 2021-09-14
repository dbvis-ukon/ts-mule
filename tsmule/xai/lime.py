"""Explain a time series with LIME algorithm."""
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
    Lasso = linear_model.Lasso(alpha=.01)
    Ridge = linear_model.Ridge(alpha=.01)

    # Fit and Partial fit
    SGDClassifier = linear_model.SGDClassifier()
    SGDRegressor = linear_model.SGDRegressor()


class AbstractXAI(ABC):
    """Abstract module for explainable AI."""


    @abstractclassmethod
    def __init__(self) -> None:
        """Abstract construct."""
        pass


    @abstractclassmethod
    def explain(self, x, predict_fn, **kwargs):
        """Generate explaination for time series x, given a model or predict function.

        Args:
            x (ndarray): Time series x with (n_steps, n_features)
            predict_fn (function): Predict function of the model.
        """
        pass


class LimeBase(AbstractXAI):
    """Module of LIME in explaining a model."""


    def __init__(self, kernel=None, sampler=None, segmenter=None) -> None:
        """Construct perturbation base explainer.

        Args:
            kernel (obj, optional): The sklearn.linear_model for infering output of explaining model.
                Defaults to None.
            segmenter (obj, optional): Segmenation object from tsmule.sampling.segment.
                Defaults to None.
            sampler (obj, optional): Perturbation object from tsmule.sampling.perturb.
                Defaults to None.
        """
        self._kernel = kernel
        self._sampler = sampler
        self._segmenter = segmenter

        self.logger = logging.getLogger(f'::{self.__class__.__name__}::')
        self._coef = None
        self._xcoef = None


    @property
    def segment_coef(self):
        """Coefficient per segment (array).

        Returns:
            array: Array of coefficient/relevance.
        """
        return self._coef


    @property
    def coef(self):
        """Coefficient of all time points.

        Returns:
            ndarray: All coefficients of the time series. It has same shape with the time series explained.
        """
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


    def explain(self, x, predict_fn, segmentation_method='slopes-max', **kwargs):
        """Generate explaination for a time series.

        Args:
            x (ndarays): a time series with shape n_steps, n_features
            predict_fn (function): Function to make the prediction given the input of x.
                For keras models, the predict_fn is `keras_model.predict`.
                Because the keras model accept input of shape (n_sample, n_steps, n_features),
                so it is recommended that the model should handle both single or multiple instances.
            segmentation_method (str, optional): Segmentation method as cited in the paper.
                Defaults to 'slopes-max'.

        Returns:
            ndarray: Coefficients of all points in the time series. Same shape with the time series.
        """
        _, features = x.shape
        # Get segmentation masks
        seg_m = self._segmenter.segment(x, segmentation_method=segmentation_method)

        # Generate samples
        samples = self._sampler.perturb(x, seg_m)

        # Fitting into the model/kernel
        kernel = self._kernel
        self._kernel, self.score = self._explain(samples, kernel, predict_fn)

        # Set coef of segments
        coef = np.array(self._kernel.coef_)
        xcoef = self.to_original(coef, seg_m)

        return xcoef


    @staticmethod
    def to_original(coef, segments):
        """Convert coef per segment to coef per point.

        Args:
            coef (array): Coefficients of unique segments.
            segments (ndarray): Original segmentations of its time series.

        Returns:
            ndarray: coefficients of each point and have same shape with the time series (n_steps, n_features).
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
    """LIME explainer for time series."""

    def __init__(self,
                 kernel=None,
                 segmenter=None,
                 sampler=None,
                 partitions=10,
                 win_length=-1,
                 p_off=0.5,
                 replace_method='zeros',
                 n_samples=100,
                 **kwargs) -> None:
        """Construct LIME explainer for time series.

        Args:
            kernel (obj, optional): The sklearn.linear_model for infering output of explaining model.
                Defaults to None.
            segmenter (obj, optional): Segmenation object from tsmule.sampling.segment.
                Defaults to None.
            sampler (obj, optional): Perturbation object from tsmule.sampling.perturb.
                Defaults to None.
            partitions (int, optional): number of partitions.
                Defaults to 10.
            win_length (int, optional): window/subspace length.
                Defaults to -1.
            p_off (float, optional): Off probability when perturbing.
                Defaults to 0.5.
            replace_method (str, optional): Method to perturbation in tsmule.sampling.replace.
                Defaults to 'zeros'.
            n_samples (int, optional): Number of samples in perturbation.
                Defaults to 100.
        """
        kernel = kernel or Kernels.Lasso
        sampler = sampler or Perturbation(p_off, replace_method, n_samples)
        segmenter = segmenter or MatrixProfileSegmentation(partitions, win_length)

        super().__init__(kernel=kernel, sampler=sampler, segmenter=segmenter)
