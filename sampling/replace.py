"""Module to calculate replacement."""
import numpy as np

def _reshape_x(x, segments):
    assert x.shape == segments.shape, (
            f"{x.shape} does not match with segments shape {segments.shape}")
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
        segments = segments.reshape(-1, 1)
    return x, segments


def zeros(x, *_args, **_kwargs):
    return np.zeros_like(x)


def local_mean(x, segments, *_args, **_kwargs):
    x, segments = _reshape_x(x, segments)
    _, features = x.shape

    r = np.zeros_like(x).astype(float)
    for i in range(features):
        # Get average per segment
        for l in np.unique(segments):
            idx = (segments[:, i] == l)
            r[idx, i] = np.average(x[idx, i])
    return r


def global_mean(x, *_args, **_kwargs):
    r = local_mean(x, np.ones_like(x))
    return r


def local_noise(x, x_segmented, segments, *_args, **_kwargs):
    pass


def global_noise(x, segments, *_args, **_kwargs):
    pass


def reference_set(ref_set, segments, *_args, **_kwargss):
    pass

def inverse_max(x, *_args, **_kwargs):
    _, n_features = x.shape
    v = x.max(axis=0)
    r = v.reshape(-1, n_features) - x
    return r

def random(x, *_args, **_kwargs):
    r = np.random.rand(*x.shape)
    return r