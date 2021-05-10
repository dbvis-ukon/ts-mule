import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../tsmule')))

from tsmule.sampling.perturb import Perturbation


ts = np.array([[1, 3, 9, 5, 4, 6, 7, 5, 9, 2, 6, 6, 7, 4, 0, 0]]).T
mts = np.array([[1, 3, 9, 5, 4, 6, 7, 5, 9, 2, 6, 6, 7, 4, 0, 0],
                [5, 9, 6, 0, 5, 8, 8, 1, 0, 2, 5, 4, 4, 5, 8, 0]]).T

segm_ts = np.array([[0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2]]).T
segm_ts2 = np.array([[7, 7, 7, 7, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6]]).T
segm_mts = np.stack([segm_ts.ravel(), segm_ts2.ravel()], axis=1)


def test_get_segment_mask():
    pass


def test_get_sample():
    pass


def test_pertub():
    sampler = Perturbation()
    samples = sampler.get_samples(ts, segm_ts)
    x, z_prime, pi = next(samples)

    assert pi < 1 and len(z_prime.shape) == 1 and x.shape == ts.shape

    sampler = Perturbation()
    samples = sampler.get_samples(mts, segm_mts)
    x, z_prime, pi = next(samples)

    assert pi < 1 and len(z_prime.shape) == 1 and x.shape == mts.shape
