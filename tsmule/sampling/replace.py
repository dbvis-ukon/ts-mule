"""Calculate replacements for a time series."""
import numpy as np


def _to_2d(x, segments=None):
    """Validate and reshape 1D/2D arrays to 2D while remembering original shape."""
    x_arr = np.asarray(x, dtype=float)
    was_1d = x_arr.ndim == 1
    if was_1d:
        x_2d = x_arr.reshape(-1, 1)
    elif x_arr.ndim == 2:
        x_2d = x_arr
    else:
        raise ValueError(f'Expected 1D or 2D array for x, got shape {x_arr.shape}')

    if segments is None:
        return x_2d, None, was_1d

    seg_arr = np.asarray(segments)
    if x_arr.shape != seg_arr.shape:
        raise AssertionError(
            f'{x_arr.shape} does not match with segments shape {seg_arr.shape}'
        )
    seg_2d = seg_arr.reshape(-1, 1) if was_1d else seg_arr
    return x_2d, seg_2d, was_1d


def _restore(arr, was_1d):
    """Restore 1D shape if the original input was 1D."""
    return arr.ravel() if was_1d else arr


def zeros(x, *_args, **_kwargs):
    """Zeros replacement.

    Args:
        x (ndarray): A time series with any shape.

    Returns:
        ndarray: Zeros like x.
    """
    return np.zeros_like(x, dtype=float)


def local_mean(x, segments, *_args, **_kwargs):
    """Local mean replacements for each segment.

    Args:
        x (ndarray): A time series with shape (n_steps,) or (n_steps, n_features).
        segments (ndarray): Segmentation of the time series. Same shape as x.

    Returns:
        ndarray: The average per segment per feature in x.
    """
    x_2d, seg_2d, was_1d = _to_2d(x, segments)
    _, features = x_2d.shape

    r = np.zeros_like(x_2d, dtype=float)
    for i in range(features):
        for s in np.unique(seg_2d[:, i]):
            idx = seg_2d[:, i] == s
            r[idx, i] = np.mean(x_2d[idx, i])
    return _restore(r, was_1d)


def global_mean(x, *_args, **_kwargs):
    """Global mean replacements per feature.

    Args:
        x (ndarray): A time series with shape (n_steps,) or (n_steps, n_features).

    Returns:
        ndarray: The average of whole x per feature.
    """
    x_2d, _, was_1d = _to_2d(x)
    mean_vals = np.mean(x_2d, axis=0, keepdims=True)
    r = np.broadcast_to(mean_vals, x_2d.shape).copy()
    return _restore(r, was_1d)


def local_noise(x, segments, *_args, random_state=None, **_kwargs):
    """Local Gaussian noise replacements for each segment.

    Args:
        x (ndarray): A time series with shape (n_steps,) or (n_steps, n_features).
        segments (ndarray): Segmentation of the time series. Same shape as x.
        random_state (int or np.random.Generator, optional): Random seed or generator.

    Returns:
        ndarray: Gaussian noise matching each segment's mean and standard deviation.
    """
    x_2d, seg_2d, was_1d = _to_2d(x, segments)
    _, features = x_2d.shape
    rng = np.random.default_rng(random_state)

    r = np.zeros_like(x_2d, dtype=float)
    for i in range(features):
        feat_std = float(np.std(x_2d[:, i]))
        fallback_std = feat_std if feat_std > 0 else 1e-6
        for s in np.unique(seg_2d[:, i]):
            idx = seg_2d[:, i] == s
            seg_vals = x_2d[idx, i]
            mu = float(np.mean(seg_vals))
            sigma = float(np.std(seg_vals))
            if sigma <= 0:
                sigma = fallback_std
            r[idx, i] = rng.normal(loc=mu, scale=sigma, size=int(np.sum(idx)))
    return _restore(r, was_1d)


def global_noise(x, *_args, random_state=None, **_kwargs):
    """Global Gaussian noise replacements per feature.

    Args:
        x (ndarray): A time series with shape (n_steps,) or (n_steps, n_features).
        random_state (int or np.random.Generator, optional): Random seed or generator.

    Returns:
        ndarray: Gaussian noise matching each feature's global mean and std.
    """
    x_2d, _, was_1d = _to_2d(x)
    n_steps, features = x_2d.shape
    rng = np.random.default_rng(random_state)

    r = np.zeros_like(x_2d, dtype=float)
    for i in range(features):
        mu = float(np.mean(x_2d[:, i]))
        sigma = float(np.std(x_2d[:, i]))
        if sigma <= 0:
            sigma = 1e-6
        r[:, i] = rng.normal(loc=mu, scale=sigma, size=n_steps)
    return _restore(r, was_1d)


def inverse_max(x, *_args, **_kwargs):
    """Inverse max replacement.

    Args:
        x (ndarray): A time series with shape (n_steps,) or (n_steps, n_features).

    Returns:
        ndarray: ``max(x, axis=0) - x`` per feature.
    """
    x_2d, _, was_1d = _to_2d(x)
    max_v = np.max(x_2d, axis=0, keepdims=True)
    r = max_v - x_2d
    return _restore(r, was_1d)


def inverse_mean(x, *_args, **_kwargs):
    """Inverse mean replacement.

    Args:
        x (ndarray): A time series with shape (n_steps,) or (n_steps, n_features).

    Returns:
        ndarray: ``global_mean(x) - x`` per feature.
    """
    x_arr = np.asarray(x, dtype=float)
    mean_v = global_mean(x_arr)
    return mean_v - x_arr


def random(x, *_args, random_state=None, **_kwargs):
    """Uniform random [0, 1) replacements.

    Args:
        x (ndarray): A time series with any shape.
        random_state (int or np.random.Generator, optional): Random seed or generator.

    Returns:
        ndarray: Uniform random array with the same shape as x.
    """
    x_arr = np.asarray(x)
    rng = np.random.default_rng(random_state)
    return rng.random(x_arr.shape)
