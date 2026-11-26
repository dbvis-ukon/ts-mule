"""Explain a time series with the LIME algorithm."""
from abc import ABC, abstractmethod
import logging

import numpy as np
from sklearn import base as sk_base
from sklearn import linear_model, metrics
from sklearn.model_selection import train_test_split

from ..sampling.perturb import Perturbation
from ..sampling.segment import MatrixProfileSegmentation


def _clone_kernel(kernel):
    """Clone a scikit-learn estimator so Lime instances do not share fitted state."""
    if kernel is None:
        return None
    if hasattr(kernel, 'get_params'):
        return sk_base.clone(kernel)
    return kernel


class Kernels:
    """Kernels for perturbation-based XAI method.

    Notice that we use scikit-learn linear-regression kernels.
    There are two main fitting types in fitting the model: fit, and partial fit.
    Fit means we see all samples as a whole, while partial fit (or online learning with mini batches) is
    more beneficial in scaling.

    Reference:
        - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
        - https://scikit-learn.org/stable/modules/linear_model.html#linear-model
    """

    Lasso = linear_model.Lasso(alpha=.01)
    Ridge = linear_model.Ridge(alpha=.01)

    SGDClassifier = linear_model.SGDClassifier()
    SGDRegressor = linear_model.SGDRegressor()


class AbstractXAI(ABC):
    """Abstract module for explainable AI."""

    @abstractmethod
    def __init__(self) -> None:
        """Abstract construct."""
        pass

    @abstractmethod
    def explain(self, x, predict_fn, **kwargs):
        """Generate explanation for time series x, given a model or predict function.

        Args:
            x (ndarray): Time series x with shape (n_steps,) or (n_steps, n_features).
            predict_fn (function): Predict function of the model.
        """
        pass


class LimeBase(AbstractXAI):
    """Module of LIME for explaining time series forecasting and classification models."""

    def __init__(self, kernel=None, sampler=None, segmenter=None, random_state=None) -> None:
        """Construct perturbation-based explainer.

        Args:
            kernel (obj, optional): A scikit-learn linear_model for inferring surrogate weights.
                Defaults to None.
            sampler (obj, optional): Perturbation object from ``tsmule.sampling.perturb``.
                Defaults to None.
            segmenter (obj, optional): Segmentation object from ``tsmule.sampling.segment``.
                Defaults to None.
            random_state (int, optional): Random seed for train/test splitting in surrogate fit.
                Defaults to None.
        """
        self._kernel = _clone_kernel(kernel)
        self._sampler = sampler
        self._segmenter = segmenter
        self._random_state = random_state

        self.logger = logging.getLogger(f'::{self.__class__.__name__}::')
        self._coef = None
        self._xcoef = None
        self._target_class = None
        self.score = None

    @property
    def segment_coef(self):
        """Coefficient per segment (array).

        Returns:
            ndarray: Array of coefficient/relevance per segment.
        """
        return self._coef

    @property
    def coef(self):
        """Coefficient of all time points.

        Returns:
            ndarray: All coefficients of the time series with the same shape as input x.
        """
        return self._xcoef

    @property
    def target_class(self):
        """Target class index explained during the last classification call, or None."""
        return self._target_class

    @staticmethod
    def _extract_targets(predictions, target_class=None):
        """Normalize single-sample or batched predictions into a 1D float target vector."""
        arr = np.asarray(predictions, dtype=float)
        if arr.ndim == 0:
            raise ValueError('Expected at least 1D sequence of predictions.')
        if arr.ndim == 1:
            if target_class is not None:
                raise ValueError(
                    f'target_class={target_class} requested, but predictions are 1D scalars.'
                )
            return arr
        if arr.ndim == 2:
            if arr.shape[1] == 1 and target_class is None:
                return arr[:, 0]
            if target_class is None:
                raise ValueError(
                    'target_class must be provided when predict_fn returns multi-class outputs.'
                )
            return arr[:, target_class]
        raise ValueError(f'Unsupported prediction output shape {arr.shape}.')

    @staticmethod
    def _explain(
        samples,
        kernel,
        predict_fn,
        target_class=None,
        batch_predict=False,
        random_state=42,
    ):
        new_x, z_prime, pi = list(zip(*samples))

        if batch_predict:
            raw_preds = predict_fn(np.asarray(new_x))
        else:
            raw_preds = [predict_fn(sample) for sample in new_x]

        z_hat = LimeBase._extract_targets(raw_preds, target_class=target_class)
        z_prime_arr = np.asarray(z_prime, dtype=float)
        pi_arr = np.asarray(pi, dtype=float)

        split_seed = 42 if random_state is None else random_state
        X, X_test, y, y_test, sw, sw_test = train_test_split(
            z_prime_arr, z_hat, pi_arr, test_size=0.3, random_state=split_seed
        )

        sw = np.nan_to_num(np.abs(sw), nan=0.01)

        kernel.fit(X, y, sample_weight=sw)

        y_pred = kernel.predict(X_test)
        score = float(metrics.r2_score(y_test, y_pred))

        return kernel, score

    def _infer_target_class(self, x, predict_fn, target_class, batch_predict):
        """Determine target_class automatically for classifier outputs."""
        probe_input = np.expand_dims(x, axis=0) if batch_predict else x
        probe_pred = np.asarray(predict_fn(probe_input))
        squeezed = np.squeeze(probe_pred)
        if squeezed.ndim == 0:
            if target_class is not None:
                raise ValueError(
                    f'target_class={target_class} given for a scalar-output predict_fn.'
                )
            return None
        if squeezed.ndim == 1:
            if target_class is None:
                return int(np.argmax(squeezed))
            if not (-squeezed.shape[0] <= target_class < squeezed.shape[0]):
                raise IndexError(
                    f'target_class={target_class} out of range for {squeezed.shape[0]} classes.'
                )
            return int(target_class)
        raise ValueError(
            f'Unexpected prediction shape {probe_pred.shape} for a single instance.'
        )

    def explain(
        self,
        x,
        predict_fn,
        segmentation_method=None,
        target_class=None,
        batch_predict=False,
        **kwargs,
    ):
        """Generate explanation for a time series (forecasting or classification).

        Args:
            x (ndarray): Time series with shape ``(n_steps,)`` or ``(n_steps, n_features)``.
            predict_fn (function): Function returning scalar predictions (regression/forecasting)
                or 1D class probabilities/logits (classification). When ``batch_predict=True``,
                ``predict_fn`` receives a batch array of shape ``(n_samples, *x.shape)``.
            segmentation_method (str, optional): Segmentation variant passed to the segmenter
                (e.g. ``'slopes-sorted'``, ``'slopes-not-sorted'``, ``'bins-min'``,
                ``'bins-max'``, ``'uniform'``, ``'exponential'``). When ``None``, the segmenter's
                default method is used.
            target_class (int, optional): Class index to explain for classification models.
                If ``None`` and ``predict_fn`` outputs class scores, defaults to ``argmax`` on ``x``.
            batch_predict (bool, optional): If True, invoke ``predict_fn`` once on all perturbed
                samples stacked along axis 0. Defaults to False.

        Returns:
            ndarray: Relevance coefficients for every point, matching ``x.shape``.
        """
        x_arr = np.asarray(x, dtype=float)
        was_1d = x_arr.ndim == 1
        if was_1d:
            x_2d = x_arr.reshape(-1, 1)
        elif x_arr.ndim == 2:
            x_2d = x_arr
        else:
            raise ValueError(f'Expected 1D or 2D time series, got shape {x_arr.shape}')

        resolved_target = self._infer_target_class(
            x_arr, predict_fn, target_class, batch_predict
        )
        self._target_class = resolved_target

        if segmentation_method is None:
            seg_m = self._segmenter.segment(x_2d)
        else:
            seg_m = self._segmenter.segment(x_2d, segmentation_method=segmentation_method)

        samples_2d = self._sampler.perturb(x_2d, seg_m)
        if was_1d:
            samples = (
                (sample_x.ravel(), z_prime, pi)
                for sample_x, z_prime, pi in samples_2d
            )
        else:
            samples = samples_2d

        kernel = _clone_kernel(self._kernel)
        self._kernel, self.score = self._explain(
            samples,
            kernel,
            predict_fn,
            target_class=resolved_target,
            batch_predict=batch_predict,
            random_state=self._random_state,
        )

        coef = np.asarray(self._kernel.coef_, dtype=float).ravel()
        xcoef_2d = self.to_original(coef, seg_m)
        xcoef = xcoef_2d.ravel() if was_1d else xcoef_2d

        self._coef = coef
        self._xcoef = xcoef
        return xcoef

    @staticmethod
    def to_original(coef, segments):
        """Convert coef per segment to coef per point.

        Args:
            coef (array): Coefficients of unique segments.
            segments (ndarray): Original segmentations of the time series.

        Returns:
            ndarray: Coefficients of each point with the same shape as ``segments``.
        """
        coef_arr = np.asarray(coef, dtype=float).ravel()
        x_coef = np.zeros_like(segments, dtype=float)

        seg_unique_labels = np.unique(segments)
        assert coef_arr.shape == seg_unique_labels.shape, (
            f'Coefficient shape {coef_arr.shape} does not match '
            f'unique segments shape {seg_unique_labels.shape}'
        )

        for i, label in enumerate(seg_unique_labels):
            x_coef[segments == label] = coef_arr[i]
        return x_coef


class LimeTS(LimeBase):
    """LIME explainer for time series forecasting and classification."""

    def __init__(
        self,
        kernel=None,
        segmenter=None,
        sampler=None,
        partitions=10,
        win_length=-1,
        p_off=0.5,
        replace_method='zeros',
        n_samples=100,
        random_state=None,
        **kwargs,
    ) -> None:
        """Construct LIME explainer for time series.

        Args:
            kernel (obj, optional): A scikit-learn linear_model for surrogate regression.
                Defaults to a cloned ``Kernels.Lasso``.
            segmenter (obj, optional): Segmentation object from ``tsmule.sampling.segment``.
                Defaults to ``MatrixProfileSegmentation(partitions, win_length)``.
            sampler (obj, optional): Perturbation object from ``tsmule.sampling.perturb``.
                Defaults to ``Perturbation(p_off, replace_method, n_samples, random_state)``.
            partitions (int, optional): Number of partitions. Defaults to 10.
            win_length (int, optional): Window/subsequence length (``-1`` infers automatically).
                Defaults to -1.
            p_off (float, optional): Off probability when perturbing. Defaults to 0.5.
            replace_method (str, optional): Replacement strategy in ``tsmule.sampling.replace``.
                Defaults to ``'zeros'``.
            n_samples (int, optional): Number of perturbation samples. Defaults to 100.
            random_state (int, optional): Random seed for reproducibility. Defaults to None.
        """
        kernel = kernel if kernel is not None else Kernels.Lasso
        sampler = sampler if sampler is not None else Perturbation(
            p_off=p_off,
            method=replace_method,
            n_samples=n_samples,
            random_state=random_state,
        )
        segmenter = segmenter if segmenter is not None else MatrixProfileSegmentation(
            partitions=partitions, win_length=win_length
        )

        super().__init__(
            kernel=kernel,
            sampler=sampler,
            segmenter=segmenter,
            random_state=random_state,
        )
