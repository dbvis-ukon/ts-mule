"""Module to generate samples through perturbation."""
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats

from . import replace as repl

_STOCHASTIC_REPLACE_METHODS = {'random', 'local_noise', 'global_noise'}


class AbstractPerturbation(ABC):
    """Abstract Perturbation with abstract methods."""

    @abstractmethod
    def __init__(self, **kwargs):
        """Abstract construction."""
        self.p_off = None
        self.repl_method = None
        self.n_samples = None

    @abstractmethod
    def perturb(self, ts, segments):
        """Perturb a time series to create new samples with the same shape.

        Args:
            ts (np.ndarray): A time series of shape (n_steps,) or (n_steps, n_features).
            segments (np.ndarray): Segment labels with the same shape as ts.

        Yields:
            tuple: ``(new_sample, on_off_segments, similarity)``
        """
        pass


class Perturbation(AbstractPerturbation):
    """Base Perturbation module."""

    def __init__(self, p_off=0.5, method='zeros', n_samples=10, random_state=None):
        """Construct perturbation base module.

        Args:
            p_off (float, optional): Probability of disabling a segment. Defaults to 0.5.
            method (str, optional): Method to replace disabled segments:
                ``'zeros' | 'global_mean' | 'local_mean' | 'inverse_mean' | 'inverse_max' | 'random' | 'local_noise' | 'global_noise'``.
                Defaults to ``'zeros'``.
            n_samples (int, optional): Number of perturbed samples to generate. Defaults to 10.
            random_state (int or np.random.Generator, optional): Random seed or generator.
                Defaults to None.
        """
        self.p_off = p_off
        self.repl_method = method
        self.n_samples = n_samples
        self.random_state = random_state

    @staticmethod
    def _get_on_off_segments(segm, p=0.5, random_state=None):
        n_seg = len(np.unique(segm))
        rng = np.random.default_rng(random_state)
        # 0 = off/disabled/replaced, 1 = on/keep/unchanged
        return rng.choice([0, 1], size=n_seg, p=[p, 1 - p])

    @staticmethod
    def _get_segment_mask(segm, on_off_segments):
        labels = np.unique(segm)
        mask = np.ones_like(segm)
        for i, label in enumerate(labels):
            mask[segm == label] = on_off_segments[i]
        return mask

    @staticmethod
    def _get_similarity(x, z, method='kendalltau'):
        pi = 1.0
        if method in ('pearsonr', 'spearmanr', 'kendalltau'):
            fn = getattr(stats, method)
            pi, _ = fn(np.asarray(x).ravel(), np.asarray(z).ravel())
            pi = float(np.nan_to_num(pi, nan=0.01))
        return pi

    @classmethod
    def get_sample(cls, x, segm, r=None, p_off=0.5, random_state=None):
        """Get a perturbed sample of x by replacing disabled segments with r.

        Args:
            x (ndarray): Input time series.
            segm (ndarray): Segmentation mask with the same shape as x.
            r (ndarray, optional): Replacement values with the same shape as x. Defaults to zeros.
            p_off (float, optional): Probability of disabling a segment. Defaults to 0.5.
            random_state (int or np.random.Generator, optional): Random seed or generator.

        Yields:
            tuple: ``(new_x, z_prime, pi)``
        """
        x = np.asarray(x, dtype=float)
        if r is None:
            r = np.zeros_like(x, dtype=float)
        assert r.shape == x.shape == segm.shape

        z_prime = cls._get_on_off_segments(segm, p_off, random_state=random_state)
        mask = cls._get_segment_mask(segm, z_prime)

        new_x = x * mask + r * (1 - mask)
        pi = cls._get_similarity(x, new_x)
        yield new_x, z_prime, pi

    @classmethod
    def get_samples(
        cls,
        x,
        segm,
        replace_method='zeros',
        p_off=0.5,
        n_samples=10,
        random_state=None,
    ):
        """Perturb and generate sample sets from a time series and its segmentation.

        Args:
            x (np.ndarray): A time series with shape (n_steps,) or (n_steps, n_features).
            segm (np.ndarray): A segmentation mask with the same shape as x.
            replace_method (str): Method name in ``tsmule.sampling.replace``.
            p_off (float): Probability of disabling a segment. Defaults to 0.5.
            n_samples (int): Number of samples to generate. Defaults to 10.
            random_state (int or np.random.Generator, optional): Random seed or generator.

        Yields:
            tuple: ``(new_sample, on_off_segments, similarity)``
        """
        fn = getattr(repl, replace_method, None)
        if fn is None or not callable(fn) or replace_method.startswith('_'):
            raise ValueError(f"Unknown replace_method '{replace_method}'")

        rng = np.random.default_rng(random_state)
        is_stochastic = replace_method in _STOCHASTIC_REPLACE_METHODS
        r = None if is_stochastic else fn(x, segm)

        for _ in range(n_samples):
            sample_r = fn(x, segm, random_state=rng) if is_stochastic else r
            yield from cls.get_sample(x, segm, sample_r, p_off, random_state=rng)

    def perturb(self, ts, segments):
        """Perturb and generate sample sets from a time series and its segmentation.

        Args:
            ts (np.ndarray): A time series with shape (n_steps,) or (n_steps, n_features).
            segments (np.ndarray): A segmentation mask with the same shape as ts.

        Yields:
            tuple: ``(new_sample, on_off_segments, similarity)``
        """
        return self.get_samples(
            ts,
            segments,
            self.repl_method,
            self.p_off,
            self.n_samples,
            random_state=self.random_state,
        )
