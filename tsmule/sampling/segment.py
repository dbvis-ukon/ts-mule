"""Segmentation for time series."""
from abc import ABC, abstractmethod

import numpy as np
from pyts.approximation import SymbolicAggregateApproximation
import stumpy


def _ensure_2d(time_series_sample):
    """Convert 1D or 2D time series input to a float64 2D array."""
    arr = np.asarray(time_series_sample, dtype=np.float64)
    if arr.ndim == 1:
        return arr.reshape(-1, 1), True
    if arr.ndim == 2:
        return arr, False
    raise ValueError(
        f'Expected 1D or 2D time series array, got shape {arr.shape}'
    )


def _restore_shape(mask, was_1d):
    """Restore mask to 1D if the original input was 1D."""
    return mask.ravel() if was_1d else mask


class AbstractSegmentation(ABC):
    """Abstract Segmentation with abstract methods."""

    @abstractmethod
    def __init__(self):
        """Abstract construct."""
        pass

    @abstractmethod
    def segment(self, time_series_sample, **kwargs):
        """Time series instance segmentation into segments.

        Args:
            time_series_sample (ndarray): Time series data (n_steps,) or (n_steps, n_features)

        Returns:
            ndarray: Integer segmentation mask with the same shape as time_series_sample.
        """
        pass


class MatrixProfileSegmentation(AbstractSegmentation):
    """Matrix Profile Segmentation using a matrix profile on every feature."""

    def __init__(self, partitions=10, win_length=3):
        """Construct segmenter of a time series using matrix profile.

        Args:
            partitions (int, optional): Number of partitions. Defaults to 10.
            win_length (int, optional): Subsequence window length, or -1 to infer
                from ``n_steps / partitions``. Defaults to 3.
        """
        self.partitions = max(1, int(partitions))
        if win_length == -1:
            self.win_length = -1
        else:
            self.win_length = max(3, int(win_length))

    @staticmethod
    def _resolve_mp_win_len(n_steps, m, k):
        """Resolve and validate the matrix profile subsequence window length."""
        if n_steps < 4:
            raise ValueError(
                f'MatrixProfileSegmentation requires at least 4 time steps, got {n_steps}'
            )
        if m == -1:
            return max(3, min(n_steps - 1, int(n_steps / max(1, k))))
        m_int = int(m)
        if m_int < 3 or m_int >= n_steps:
            raise ValueError(
                f'Matrix profile window length m must be in [3, {n_steps - 1}], got {m_int}'
            )
        return m_int

    @classmethod
    def _segment_with_slopes(cls, time_series_sample, m=4, k=10, profile='sorted'):
        """Time series instance segmentation into segments.

        Idea:
         - Compute the matrix profile of each feature (optionally sorted).
         - Calculate slopes of the matrix profile and take the k largest jumps as split points.

        Args:
            time_series_sample (ndarray): Time series data (n_steps,) or (n_steps, n_features).
            m (int, optional): Subsequence window length for matrix profile, or -1 for auto.
                Defaults to 4.
            k (int, optional): Number of partitions. Defaults to 10.
            profile (str, optional): Whether to sort matrix profile before computing slopes
                ('sorted' or 'not-sorted'). Defaults to 'sorted'.

        Returns:
            ndarray: Integer segmentation mask with the same shape as time_series_sample.
        """
        if profile not in ('sorted', 'not-sorted'):
            raise ValueError(
                f"Unknown profile '{profile}'. Expected 'sorted' or 'not-sorted'."
            )

        ts, was_1d = _ensure_2d(time_series_sample)
        n_steps, n_features = ts.shape
        mp_win_len = cls._resolve_mp_win_len(n_steps, m, k)

        segmentation_mask = np.zeros((n_steps, n_features), dtype=int)
        win_idx = 0

        for feature in range(n_features):
            col = np.ascontiguousarray(ts[:, feature], dtype=np.float64)
            mp = stumpy.stump(col, mp_win_len)
            mp_ = np.nan_to_num(
                mp[:, 0].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
            )

            if len(mp_) < 2:
                segmentation_mask[:, feature] = win_idx
                win_idx += 1
                continue

            if profile == 'sorted':
                mp_idx_sorted = np.argsort(mp_)
                temp_mp = mp_[mp_idx_sorted]
                slopes = np.diff(temp_mp)
                slopes_sorted = np.argsort(slopes)[::-1][:k]
                partitions_idx_sorted = sorted(
                    {
                        int(mp_idx_sorted[part])
                        for part in slopes_sorted
                        if int(mp_idx_sorted[part]) > 0
                    }
                )
            else:
                slopes = np.abs(np.diff(mp_))
                slopes_sorted = np.argsort(slopes)[::-1][:k]
                partitions_idx_sorted = sorted(
                    {int(part) for part in slopes_sorted if int(part) > 0}
                )

            partitions_idx_sorted.append(n_steps)

            start = 0
            for end in partitions_idx_sorted:
                if end > start:
                    segmentation_mask[start:end, feature] = win_idx
                    win_idx += 1
                    start = end

        return _restore_shape(segmentation_mask, was_1d)

    @classmethod
    def _segment_with_bins(cls, time_series_sample, m=4, k=10, distance_method='max'):
        """Segment a time series by using matrix profile distance and its bins.

        For shared points between two windows, it can minimize or maximize the nearest distance.

        Args:
            time_series_sample (ndarray): Time series data (n_steps,) or (n_steps, n_features).
            m (int, optional): Subsequence window length for matrix profile, or -1 for auto.
                Defaults to 4.
            k (int, optional): Initial max number of partitions. Defaults to 10.
            distance_method (str, optional): Options can be ``'min'`` or ``'max'``.
                Defaults to ``'max'``.

        Returns:
            ndarray: Integer segmentation mask with the same shape as time_series_sample.
        """
        if distance_method not in ('max', 'min'):
            raise ValueError(
                f"Unknown distance_method '{distance_method}'. Expected 'min' or 'max'."
            )

        ts, was_1d = _ensure_2d(time_series_sample)
        n_steps, n_features = ts.shape
        mp_win_len = cls._resolve_mp_win_len(n_steps, m, k)

        segmentation_mask = np.zeros((n_steps, n_features), dtype=int)
        seg_start = 0

        for feature in range(n_features):
            col = np.ascontiguousarray(ts[:, feature], dtype=np.float64)
            mp = stumpy.stump(col, mp_win_len)
            mp_d = np.nan_to_num(
                mp[:, 0].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
            )
            mp_d_min = min(0.0, float(mp_d.min()))
            mp_d_max = float(mp_d.max())

            if mp_d_max <= mp_d_min:
                segments = np.full(len(mp_d), seg_start, dtype=int)
            else:
                n_bins = max(2, int(k))
                bins = np.linspace(mp_d_min, mp_d_max, n_bins)
                segments = np.clip(np.digitize(mp_d, bins) - 1, 0, n_bins - 1)
                segments = seg_start + segments.astype(int)

            if distance_method == 'max':
                init_v = int(segments.min())
                _fn = np.maximum
            else:
                init_v = int(segments.max())
                _fn = np.minimum

            seg_m = np.full(n_steps, init_v, dtype=int)
            for i, s in enumerate(segments):
                seg_m[i:i + mp_win_len] = _fn(seg_m[i:i + mp_win_len], int(s))

            segmentation_mask[:, feature] = seg_m
            seg_start = int(seg_m.max()) + 1

        return _restore_shape(segmentation_mask, was_1d)

    def segment(self, time_series_sample, segmentation_method='slopes-sorted'):
        """Time series instance segmentation into segments.

        Args:
            time_series_sample (ndarray): Time series data (n_steps,) or (n_steps, n_features).
            segmentation_method (str, optional): Segmentation method to be used.
                Defaults to ``'slopes-sorted'``.
                Possible: ``'slopes-sorted'``, ``'slopes-not-sorted'``,
                ``'bins-min'``, ``'bins-max'``.

        Returns:
            ndarray: Integer segmentation mask with the same shape as time_series_sample.
        """
        if segmentation_method == 'slopes-sorted':
            return self._segment_with_slopes(
                time_series_sample,
                m=self.win_length,
                k=self.partitions,
                profile='sorted',
            )
        if segmentation_method == 'slopes-not-sorted':
            return self._segment_with_slopes(
                time_series_sample,
                m=self.win_length,
                k=self.partitions,
                profile='not-sorted',
            )
        if segmentation_method == 'bins-max':
            return self._segment_with_bins(
                time_series_sample,
                m=self.win_length,
                k=self.partitions,
                distance_method='max',
            )
        if segmentation_method == 'bins-min':
            return self._segment_with_bins(
                time_series_sample,
                m=self.win_length,
                k=self.partitions,
                distance_method='min',
            )

        raise ValueError(
            f"Unknown segmentation_method '{segmentation_method}'. "
            "Expected one of: 'slopes-sorted', 'slopes-not-sorted', 'bins-min', 'bins-max'."
        )


class SAXSegmentation(AbstractSegmentation):
    """SAX Segmentation applied on every feature."""

    def __init__(self, partitions=10, win_length=3):
        """Construct segmenter for SAX algorithm.

        Args:
            partitions (int, optional): Target number of partitions. Defaults to 10.
            win_length (int, optional): Unused, kept for API compatibility. Defaults to 3.
        """
        self.partitions = max(1, int(partitions))

    def segment(self, time_series_sample, **_kwargs):
        """Time series instance segmentation into segments using SAX.

        Args:
            time_series_sample (ndarray): Time series data (n_steps,) or (n_steps, n_features).

        Returns:
            ndarray: Integer segmentation mask with the same shape as time_series_sample.
        """
        ts, was_1d = _ensure_2d(time_series_sample)
        n_steps, n_features = ts.shape
        segmentation_mask = np.zeros((n_steps, n_features), dtype=int)

        partitions = self.partitions
        win_idx = 0

        for feature in range(n_features):
            col = ts[:, feature]

            if np.min(col) == np.max(col):
                win_len = max(1, int(n_steps / partitions))
                for i in range(n_steps):
                    if i > 0 and i % win_len == 0:
                        win_idx += 1
                    segmentation_mask[i, feature] = win_idx
                win_idx += 1
                continue

            if n_steps < 3:
                segmentation_mask[:, feature] = win_idx
                win_idx += 1
                continue

            max_bins = min(n_steps, 26)
            best_mask = None
            best_diff = float('inf')

            for n_bins in range(3, max_bins + 1):
                try:
                    sax = SymbolicAggregateApproximation(
                        n_bins=n_bins, strategy='quantile', alphabet='ordinal'
                    )
                    sax_transformation = sax.fit_transform(col.reshape(1, -1))
                except ValueError:
                    sax = SymbolicAggregateApproximation(
                        n_bins=n_bins, strategy='uniform', alphabet='ordinal'
                    )
                    sax_transformation = sax.fit_transform(col.reshape(1, -1))

                candidate = np.empty(n_steps, dtype=int)
                cur_win_idx = win_idx
                old_value = None
                for i, value in enumerate(sax_transformation.reshape(-1)):
                    if old_value is not None and value != old_value:
                        cur_win_idx += 1
                    candidate[i] = cur_win_idx
                    old_value = value

                n_segs = cur_win_idx - win_idx + 1
                diff = abs(n_segs - partitions)
                if diff < best_diff:
                    best_diff = diff
                    best_mask = candidate

                if (0.8 * partitions <= n_segs <= 1.1 * partitions) or (
                    n_segs > 1.1 * partitions
                ):
                    break

            if best_mask is None:
                segmentation_mask[:, feature] = win_idx
                win_idx += 1
            else:
                segmentation_mask[:, feature] = best_mask
                win_idx = int(best_mask.max()) + 1

        return _restore_shape(segmentation_mask, was_1d)


class WindowSegmentation(AbstractSegmentation):
    """Window segmentation with non-overlapping windows."""

    def __init__(self, partitions=10, win_length=3):
        """Construct the window segmenter for time series.

        A time series is segmented into uniform or exponential windows.

        Args:
            partitions (int, optional): Number of partitions. Defaults to 10.
            win_length (int, optional): Window length. Defaults to 3.
        """
        self.partitions = max(1, int(partitions))
        self.win_length = max(1, int(win_length))

    @staticmethod
    def _segment_with_uniform(time_series_sample, m=4):
        """Segment a time series into uniform non-overlapping windows of size m."""
        ts, was_1d = _ensure_2d(time_series_sample)
        n_steps, features = ts.shape
        m = int(m)
        if m < 1 or m > n_steps:
            raise ValueError(
                f'Window size m must be between 1 and n_steps ({n_steps}), got {m}'
            )

        starts = list(np.arange(0, n_steps, m))
        ends = starts[1:] + [n_steps]

        segmentation_mask = np.zeros((n_steps, features), dtype=int)
        win_idx = 0
        for feature in range(features):
            for i, j in zip(starts, ends):
                segmentation_mask[i:j, feature] = win_idx
                win_idx += 1

        return _restore_shape(segmentation_mask, was_1d)

    @staticmethod
    def _segment_with_exponential(time_series_sample):
        """Segment a time series into exponentially growing windows."""
        ts, was_1d = _ensure_2d(time_series_sample)
        n_steps, features = ts.shape

        if n_steps <= 1:
            segmentation_mask = np.arange(features, dtype=int).reshape(1, features)
            return _restore_shape(np.broadcast_to(segmentation_mask, (n_steps, features)).copy(), was_1d)

        x = np.arange(max(1, int(np.ceil(np.log(n_steps)))))
        win_sizes = [int(np.ceil(np.exp(i))) for i in x]
        while len(win_sizes) > 1 and sum(win_sizes[:-1]) >= n_steps:
            win_sizes.pop()
        win_sizes[-1] = n_steps - sum(win_sizes[:-1])

        starts = [0]
        for i in range(len(win_sizes) - 1):
            starts.append(starts[-1] + win_sizes[i])
        ends = starts[1:] + [n_steps]

        segmentation_mask = np.zeros((n_steps, features), dtype=int)
        win_idx = 0
        for feature in range(features):
            for i, j in zip(starts, ends):
                segmentation_mask[i:j, feature] = win_idx
                win_idx += 1

        return _restore_shape(segmentation_mask, was_1d)

    def segment(self, time_series_sample, segmentation_method='uniform'):
        """Time series instance segmentation into segments.

        Args:
            time_series_sample (ndarray): Time series data (n_steps,) or (n_steps, n_features).
            segmentation_method (str, optional): Segmentation method to be used.
                Defaults to ``'uniform'``. Possible: ``'uniform'`` | ``'exponential'``.

        Returns:
            ndarray: Integer segmentation mask with the same shape as time_series_sample.
        """
        if segmentation_method == 'uniform':
            return self._segment_with_uniform(
                time_series_sample, m=self.win_length
            )
        if segmentation_method == 'exponential':
            return self._segment_with_exponential(time_series_sample)

        raise ValueError(
            f"Unknown segmentation_method '{segmentation_method}'. "
            "Expected 'uniform' or 'exponential'."
        )


class SegmentationPicker:
    """Picker for segmentation method."""

    @staticmethod
    def select(method, partitions, win_length=3):
        """Construct a segmentation method family.

        Args:
            method (str): Segmentation method family: ``'sax'``, ``'window'``,
                or ``'matrix'`` (e.g. ``'matrix-slope'``, ``'matrix-bins'``).
            partitions (int): Number of partitions.
            win_length (int, optional): Window length. Defaults to 3.

        Returns:
            AbstractSegmentation: An instance of AbstractSegmentation.
        """
        if method == 'sax':
            return SAXSegmentation(partitions)
        if method == 'window':
            return WindowSegmentation(partitions, win_length)
        if 'matrix' in method:
            return MatrixProfileSegmentation(partitions, win_length)
        raise ValueError(
            f"Unknown segmentation family '{method}'. "
            "Expected 'sax', 'window', or a 'matrix*' method."
        )
