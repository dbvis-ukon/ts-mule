"""Test module for perturbation sampling."""
import numpy as np
import pytest

from tsmule.sampling.perturb import Perturbation

ts = np.array([[1, 3, 9, 5, 4, 6, 7, 5, 9, 2, 6, 6, 7, 4, 0, 0]]).T
mts = np.array([
    [1, 3, 9, 5, 4, 6, 7, 5, 9, 2, 6, 6, 7, 4, 0, 0],
    [5, 9, 6, 0, 5, 8, 8, 1, 0, 2, 5, 4, 4, 5, 8, 0],
]).T

segm_ts = np.array([[0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2]]).T
segm_ts2 = np.array([[7, 7, 7, 7, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6]]).T
segm_mts = np.stack([segm_ts.ravel(), segm_ts2.ravel()], axis=1)


def test_get_segment_mask():
    on_off = np.array([1, 0, 1, 0])
    mask = Perturbation._get_segment_mask(segm_ts, on_off)
    assert mask.shape == segm_ts.shape
    # Segment labels sorted: [0, 1, 2, 3] -> [1, 0, 1, 0]
    assert np.all(mask[segm_ts == 0] == 1)
    assert np.all(mask[segm_ts == 1] == 0)
    assert np.all(mask[segm_ts == 2] == 1)
    assert np.all(mask[segm_ts == 3] == 0)


def test_get_sample():
    x, z_prime, pi = next(Perturbation.get_sample(ts, segm_ts, p_off=0.5, random_state=7))
    assert x.shape == ts.shape
    assert z_prime.shape == np.unique(segm_ts).shape
    assert isinstance(pi, float)
    assert -1.0 <= pi <= 1.0


def test_similarity_nan_fallback():
    const = np.ones((10, 1))
    pi = Perturbation._get_similarity(const, const)
    assert pi == pytest.approx(0.01)


def test_perturb_deterministic_and_stochastic():
    sampler = Perturbation(method='zeros', n_samples=5, random_state=42)
    samples = list(sampler.perturb(mts, segm_mts))
    assert len(samples) == 5
    for new_x, z_prime, pi in samples:
        assert new_x.shape == mts.shape
        assert z_prime.shape == np.unique(segm_mts).shape
        assert -1.0 <= pi <= 1.0

    # Stochastic method produces distinct samples when random_state is set
    noise_sampler = Perturbation(method='local_noise', p_off=1.0, n_samples=2, random_state=123)
    s1, s2 = list(noise_sampler.perturb(mts, segm_mts))
    assert not np.allclose(s1[0], s2[0])

    with pytest.raises(ValueError):
        list(Perturbation(method='nonexistent').perturb(mts, segm_mts))
