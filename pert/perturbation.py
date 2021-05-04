"""Perturbation and Sampling for time series."""
import numpy as np
from abc import ABC, abstractmethod
from pyts.utils import segmentation


# __all__ = ['Perturbation', 'TimeSeriesPerturbation', 'SyncTimeSlicer']

class Perturbation(ABC):
    """Base Perturbation with abstract methods."""

    def __init__(self, off_p=0.5, replacement_method='zeros'):
        """Initialize perturbation module."""
        self.off_p = off_p
        self._replacement_fn = replacement_method

        self.labels = None
        self.x_segmented = None
        self.replacements = None

        if isinstance(replacement_method, str):
            # Todo: Try to use mapping
            self._replacement_fn = eval(f"self.{replacement_method}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.labels = None
        self.x_segmented = None
        self.replacements = None

    @abstractmethod
    def __segment__(self, x):
        # Todo general segmentation method for matrix.
        raise NotImplementedError()

    def __get_replacements__(self, x, x_segmented=None, labels=None, **_kwargs):
        """Prepare replacement vectors corresponding to each segments/labels.

        Notice:
            - replacement r_i same shape with z'
        """
        if x_segmented is None:
            x_segmented = self.x_segmented
            labels = self.labels

        _replacement_fn = self._replacement_fn

        r = _replacement_fn(x=x, x_segmented=x_segmented, labels=labels, **_kwargs)
        return r

    def __get_z_prime__(self, **_kwargs):
        """Sampling based on number of segments/features.

        Sampling z_comma with shape (n_samples, n_windows)
        """
        p = self.off_p
        n_segments = len(self.labels)

        z_prime = np.random.choice([0, 1], size=n_segments, p=[p, 1 - p])
        return z_prime

    def __get_z__(self, x, z_prime, replacements=None, *_args, **_kwargs):
        """Convert from z_prime to z with same format with x.

        :param z_prime: (np.array) a binary vector
            if z_prime[i] == 0, replacements[i] will be used to perturb the segment.
        """
        if replacements is None:
            replacements = self.replacements

        labels = self.labels
        x_segmented = self.x_segmented
        n_segments = len(labels)

        if isinstance(replacements, (int, float)):
            replacements = np.full_like(labels, fill_value=replacements)
        assert len(z_prime) == len(replacements) == len(labels), \
            f"Replacements length {len(replacements)} not match with windows features {len(z_prime)}."
        assert x_segmented.shape == x.shape, \
            f"Not matching shape of segmented {x_segmented.shape} and instance x {x.shape}."

        # Todo: try to use numpy native function instead of for loop
        z = np.zeros_like(x)
        for i in range(n_segments):
            _idx = x_segmented == labels[i]
            z[_idx] = z_prime[i] * x[_idx] + replacements[i] * (1 - z_prime[i])
        return z

    def __get_pi__(self, x, z, gamma=0.01, **kwargs):
        """Calculate distance/similarity from z to x.

        Because z was built from x, hence x, and z must have same shape and length.
        We could simply use 2-norm in np.linalg.norm to calculate the distance with default
        option is to use Frobenius Norm.

        Alternatively, we could use pairwise_distance function from sklearn.metrics to calculate
        the distance but must reshape to (1, -1) for 1 sample x and z only.

        To convert to similarity, we could use np.exp given gamma. gamma ~ 1 / num_features
        Reference:
            - https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.norm.html
            - https://mathworld.wolfram.com/FrobeniusNorm.html
            - https://scikit-learn.org/stable/modules/metrics.html#metrics

        :param x: (np.array) Instance X with shape (n_features, n_steps) to be explained.
            With univariate time series it could be (n_steps, ) or (n_steps, 1)
        :param z: An perturbed sample from z. z must have same shape with x.
        :param gamma: (float) A parameter used to convert to similarity.
            one heuristic for choosing gamma is 1 / num_features
        :param kwargs: Other options in np.linalg.norm()
        """
        assert x.shape == z.shape, \
            f"Not matching shape of segmented {x.shape} and instance x {z.shape}."
        d = np.linalg.norm(x - z, **kwargs)
        pi = np.exp(-d * gamma)
        return pi

    def perturb(self, x, n_samples=10, replacements=None, **_kwargs):
        """Perturb x."""
        self.labels, self.x_segmented = self.__segment__(x)

        if replacements is not None:
            self.replacements = replacements
        else:
            self.replacements = self.__get_replacements__(x, self.x_segmented, self.labels)

        # Todo: try to use numpy native function instead of for loop
        for i in range(n_samples):
            z_prime = self.__get_z_prime__()
            z = self.__get_z__(x, z_prime, self.replacements)
            pi_z = self.__get_pi__(x, z)
            yield z_prime, z, pi_z

    @staticmethod
    def zeros(labels, **_kwargs):
        return np.zeros_like(labels)

    @staticmethod
    def local_mean(x, x_segmented, labels, **_kwargs):
        n_segments = len(labels)
        r = np.zeros_like(labels)

        # Todo: try to use numpy native function instead of for loop
        for i in range(n_segments):
            _idx = x_segmented == labels[i]
            r[i] = np.average(x[_idx])
        return r

    @staticmethod
    def global_mean(x, labels, **_kwargs):
        r = np.zeros_like(labels)
        r.fill(np.average(x))
        return r

    @staticmethod
    def local_noise(x, x_segmented, labels, **_kwargs):
        pass

    @staticmethod
    def global_noise(x, labels, **_kwargs):
        pass

    @staticmethod
    def reference_set(ref_set, labels, **_kwargs):
        pass


class TimeSeriesPerturbation(Perturbation):
    """Perturbation for Time Series. Supporting also multivariate time series."""

    # Todo: Implement segmentations with
    #   1. window-size: logarithm vs equally-distributed
    #   2. slicing: Frequency vs Time slice
    def __init__(self, window_size, off_p=0.5, replacement_method='zeros'):
        super().__init__(off_p=off_p, replacement_method=replacement_method)

        self.window_size = window_size

    def __segment__(self, x):
        """Segmentation instance X into segments/labels.

        Time Slices (or time slicing segmentations on normal scale)

        :param x: (np.array) x must be (n_features, n_steps)
        """
        w_size = self.window_size
        n_features, n_steps = x.shape

        # Todo: try to use numpy native function instead of for loop
        x_segmented = np.zeros_like(x)
        start, end, n_windows = segmentation(n_steps, w_size, overlapping=False)
        for i in range(n_features):
            for j in range(n_windows):
                x_segmented[i, start[j]:end[j]] = i * n_windows + j

        labels = np.unique(x_segmented)

        return labels, x_segmented

    def __get_pi__(self, x, z, **kwargs):
        """Override distance function for Time Series."""
        n_features, n_steps = x.shape
        gamma = 1 / n_steps

        return super().__get_pi__(x, z, gamma)


class SyncTimeSlicer(TimeSeriesPerturbation):
    """Perturbation for Time Series. Supporting also multivariate time series."""

    def _slices(self, x):
        w_size = self.window_size
        n_features, n_steps = x.shape

        start, end, n_windows = segmentation(n_steps, w_size, overlapping=False)
        return zip(start, end, range(n_windows))

    @staticmethod
    def _mask(arr, z_prime, slices):
        m = np.zeros_like(arr)
        for s, e, l in slices:
            m[s:e] = z_prime[l]
        return m

    def _x_masked(self, x, z_prime):
        slices = list(self._slices(x))
        mask = np.apply_along_axis(self._mask, 1, x, z_prime, slices)
        return mask

    def _x_segmented(self, x, slices=None):
        x_segmented = np.zeros_like(x)
        if slices is None:
            slices = list(self._slices(x))
        for s, e, l in slices:
            x_segmented[:, s: e] = l
        return x_segmented

    def _x_replacements(self, x, fn='zeros', slices=None, **fn_kwargs):
        if slices is None:
            slices = list(self._slices(x))
        if isinstance(fn, str):
            fn = eval(f"self.{fn}")
        r = np.apply_along_axis(fn, axis=1, arr=x, slices=slices, **fn_kwargs)

        return r

    def _z_prime(self, x, **_kwargs):
        """Sampling based on number of segments/features.

        Sampling z_comma with shape (n_samples, n_windows)
        """
        x_segmented = self._x_segmented(x)
        labels = np.unique(x_segmented)

        p = self.off_p
        n_segments = len(labels)

        z_prime = np.random.choice([0, 1], size=n_segments, p=[p, 1 - p])
        return z_prime

    def _z(self, x, z_prime, replacements, **_kwargs):
        mask = self._x_masked(x, z_prime)
        assert x.shape == replacements.shape == mask.shape, "Not matching shape between x, replacements and mask."
        z = x * mask + replacements * (1 - mask)
        return z


    def to_original(self, x, z_prime, repl_fn='zeros'):
        r = self._x_replacements(x, fn=repl_fn)
        z = self._z(x, z_prime, r)
        return z

    @staticmethod
    def zeros(x, **kwargs):
        return np.zeros_like(x)

    @staticmethod
    def local_mean(x, slices, **kwargs):
        r = np.zeros_like(x).astype(float)
        for s, e, l in slices:
            r[s:e] = np.average(x[s:e])
        return r

    @staticmethod
    def global_mean(x, **kwargs):
        r = np.zeros_like(x).astype(float)
        r.fill(np.average(x))
        return r

    @staticmethod
    def constant(x, value, **kwargs):
        r = np.zeros_like(x).fill(value)
        return r

    def perturb(self, x, n_samples=10, replacements=None, **_kwargs):
        """Perturb x."""
        _slices = list(self._slices(x))
        self.x_segmented = self._x_segmented(x)
        self.labels = np.unique(self.x_segmented)

        if isinstance(replacements, (int, float, bool)):
            r = self._x_replacements(x, fn='constant', value=replacements)
        else:
            r = self._x_replacements(x, fn=self._replacement_fn)

        for i in range(n_samples):
            z_prime = self._z_prime(x)
            z = self._z(x, z_prime, r)
            pi_z = self.__get_pi__(x, z)
            yield z_prime, z, pi_z

class ASyncTimeSlicer(SyncTimeSlicer):

    def _x_segmented(self, x, slices=None):
        x_segmented = np.zeros_like(x)
        n_features, _steps = x.shape

        if slices is None:
            slices = list(self._slices(x))
        n_segments = len(slices)        
        # assign label as increasing function
        for i in range(n_features):
            for s, e, l in slices:
                x_segmented[i, s: e] =  i * n_segments + l
        return x_segmented

    # Temp not using _mask()
    def _x_masked(self, x, z_prime):
        slices = list(self._slices(x))
        n_features, _steps = x.shape
        mask = np.ones_like(x)
        _z_prime = z_prime.reshape(n_features, -1)

        for i in range(n_features):
            for s, e, l in slices:
                mask[i, s:e] = _z_prime[i, l]
        return mask