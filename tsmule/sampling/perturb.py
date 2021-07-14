"""Module to generate samples through perturbation."""
import numpy as np
from scipy import stats
from abc import ABC, abstractmethod

from . import replace as repl


class AbstractPerturbation(ABC):
    """Abstract Pertubation with abstract methods."""


    @abstractmethod
    def __init__(self, **kwargs):
        """Abstract construction."""
        self.p_off = None
        self.repl_method = None
        self.n_samples = None


    @abstractmethod
    def perturb(self, ts, segments):
        """Perturb a time series to create new sample with same shape.

        :param ts: (np.array) A time series must be (n_steps, n_features)
        :param segments: (np.array) A segments with labels of the time series must be (n_steps, n_features)

        Yields:
            Generator: tuple of (new sample, on/off segments, similarity to original)
        """
        pass


class Perturbation(AbstractPerturbation):
    """Base Perturbation module."""

    def __init__(self, p_off=0.5, method='zeros', n_samples=10):
        """Construct perturbation base module.

        Args:
            p_off (float, optional): Probability of disabling a segment. Default is 0.5
            method (str, optional): Methods to replace parts of segmentation, including:
                'zeros | global_mean | local_mean | inverse_mean | inverse_max'
                Defaults to 'zeros'.
            n_samples (int, optional): [description]. Defaults to 10.
        """
        self.p_off = p_off
        self.repl_method = method
        self.n_samples = n_samples


    @staticmethod
    def _get_on_off_segments(segm, p=0.5):
        # Get n_seg
        n_seg = len(np.unique(segm))

        # Get on off segments
        # 0 = off/disabled/replaced, 1 = on/keep/unchanged
        v = np.random.choice([0, 1], size=n_seg, p=[p, 1 - p])
        return v


    @staticmethod
    def _get_segment_mask(segm, on_off_segments):
        # Get binary on/off masks for segments
        labels = np.unique(segm)
        n_segs = len(labels)
        mask = np.ones_like(segm)
        for i in range(n_segs):
            idx = (segm == labels[i])
            mask[idx] = on_off_segments[i]
        return mask


    @staticmethod
    def _get_similarity(x, z, method='kendalltau'):
        # Calculate pi/similarity between x and y:
        pi = 1
        if method in ['pearsonr', 'spearmanr', 'kendalltau']:
            fn = getattr(stats, method)
            pi, _ = fn(x.ravel(), z.ravel())
            # avoid nan
            pi = np.nan_to_num(pi, 0.01)
        return pi


    @classmethod
    def get_sample(cls, x, segm, r=None, p_off=0.5):
        """Get sample of x based on replace segments of x with r.

        Args:
            x (ndarray): A multivariate time series
            segm (ndarray): A segmentation of x, having same shape with x
            r (ndarray): A replacements of x when create a new sample
            p_off (float, optional): Probility of disabling a segmentation. Defaults to 0.5.
        Yields:
            Generator: a tuple of (new sample, on/off segments, similarity to original)
        """
        if r is None:
            r = np.zeros_like(x)
        assert r.shape == x.shape == segm.shape

        # On/off vector z', used to fit into XAI linear regression
        z_prime = cls._get_on_off_segments(segm, p_off)
        mask = cls._get_segment_mask(segm, z_prime)

        # get new x sample, when mask = 1, then keep x, else replace it
        new_x = x * mask + r * (1 - mask)
        pi = cls._get_similarity(x, new_x)
        yield new_x, z_prime, pi


    @classmethod
    def get_samples(cls, x, segm, replace_method='zeros', p_off=0.5, n_samples=10):
        """Perturb and generate sample sets from given time series and its segmentation.

        Args:
            ts (np.ndarray): A time series with shape (n_steps, n_features)
            segments (np.ndarray): A segmentation of the time series with shape (n_steps, n_features)
            replace_method (str): Method to replace off/disabled segment
            p_off (float): Probability of disabling a segment. Default is 0.5
            n_samples (int): Number of samples to be generated.

        Yields:
            Generator: tuples of (new sample, on/off segments, similarity to original)
        """
        fn = getattr(repl, replace_method)
        r = fn(x, segm)

        for _ in range(n_samples):
            yield from cls.get_sample(x, segm, r, p_off)


    def perturb(self, ts, segments):
        """Perturb and generate sample sets from given time series and its segmentation.

        Args:
            ts (np.ndarray): A time series with shape (n_steps, n_features)
            segments (np.ndarray): A segmentation of the time series with shape (n_steps, n_features)

        Yields:
            Generator: tuple of (new sample, on/off segments, similarity to original)
        """
        return self.get_samples(ts, segments,
                                self.repl_method,
                                self.p_off,
                                self.n_samples)
