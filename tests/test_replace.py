"""Test module for replacement calculations."""
import warnings

import numpy as np
import pytest

import tsmule.sampling.replace as repl

ts = np.array([[1, 3, 9, 5, 4, 6, 7, 5, 9, 2, 6, 6, 7, 4, 0, 0]]).T
mts = np.array([
    [1, 3, 9, 5, 4, 6, 7, 5, 9, 2, 6, 6, 7, 4, 0, 0],
    [5, 9, 6, 0, 5, 8, 8, 1, 0, 2, 5, 4, 4, 5, 8, 0],
]).T

segm_ts = np.array([[0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2]]).T
segm_ts2 = np.array([[7, 7, 7, 7, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6]]).T
segm_mts = np.stack([segm_ts.ravel(), segm_ts2.ravel()], axis=1)


def test_repl_zeros():
    x = repl.zeros(ts)
    assert np.array_equal(x, np.zeros_like(ts))

    x = repl.zeros(mts)
    assert np.array_equal(x, np.zeros_like(mts))

    x_1d = repl.zeros(ts.ravel())
    assert x_1d.shape == (len(ts),)


def test_repl_local_mean_no_warnings():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        x = repl.local_mean(ts, segm_ts)
        x_mts = repl.local_mean(mts, segm_mts)
    assert len(caught) == 0

    expected = np.array(
        [[4.3, 4.3, 4.3, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 3.4, 3.4, 3.4, 3.4, 3.4]]
    ).T
    assert np.array_equal(x.round(1), expected)

    expected_mts = np.array([
        [4.3, 4.3, 4.3, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 3.4, 3.4, 3.4, 3.4, 3.4],
        [5.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2, 3.2],
    ]).T
    assert np.array_equal(x_mts.round(1), expected_mts)


def test_global_mean():
    x = repl.global_mean(ts).round(1)
    expected = np.full_like(ts, 4.6, dtype=float)
    assert np.array_equal(x, expected)

    x = repl.global_mean(mts).round(1)
    expected = np.zeros_like(mts, dtype=float)
    expected[:, 0].fill(4.6)
    expected[:, 1].fill(4.4)
    assert np.array_equal(x, expected)


def test_inverse_max_and_inverse_mean():
    inv_max = repl.inverse_max(mts)
    assert inv_max.shape == mts.shape
    assert np.allclose(inv_max[:, 0], 9.0 - mts[:, 0])
    assert np.allclose(inv_max[:, 1], 9.0 - mts[:, 1])

    inv_max_1d = repl.inverse_max(ts.ravel())
    assert inv_max_1d.shape == (len(ts),)
    assert np.allclose(inv_max_1d, 9.0 - ts.ravel())

    inv_mean = repl.inverse_mean(mts)
    assert inv_mean.shape == mts.shape
    assert np.allclose(inv_mean, repl.global_mean(mts) - mts)


def test_noise_and_random_replacements():
    r_rand = repl.random(mts, random_state=42)
    assert r_rand.shape == mts.shape
    assert np.all((r_rand >= 0.0) & (r_rand < 1.0))

    r_gnoise = repl.global_noise(mts, random_state=42)
    assert r_gnoise.shape == mts.shape
    assert np.all(np.isfinite(r_gnoise))

    r_lnoise = repl.local_noise(mts, segm_mts, random_state=42)
    assert r_lnoise.shape == mts.shape
    assert np.all(np.isfinite(r_lnoise))

    # Constant series should also work without zero-std errors
    const = np.ones(10)
    r_const = repl.global_noise(const, random_state=0)
    assert r_const.shape == (10,)
    assert np.all(np.isfinite(r_const))


def test_shape_mismatch_raises():
    with pytest.raises(AssertionError):
        repl.local_mean(ts, segm_mts)
