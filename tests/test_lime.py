"""Test module for LIME in generate explainations."""
import numpy as np
import pytest

from sampling import Perturbation
from xai.lime import LimeBase, Kernels, LimeTS


ts = np.array([[1, 3, 9, 5, 4, 6, 7, 5, 9, 2, 6, 6, 7, 4, 0, 0]]).T
mts = np.array([[1, 3, 9, 5, 4, 6, 7, 5, 9, 2, 6, 6, 7, 4, 0, 0],
                [5, 9, 6, 0, 5, 8, 8, 1, 0, 2, 5, 4, 4, 5, 8, 0]]).T

segm_ts = np.array([[0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2]]).T
segm_ts2 = np.array([[7, 7, 7, 7, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6]]).T
segm_mts = np.stack([segm_ts.ravel(), segm_ts2.ravel()], axis=1)

def test_samples_unpack():
    sampler = Perturbation()
    
    # UTS
    samples = sampler.get_samples(ts, segm_ts, n_samples=100)
    new_x, z_prime, pi = list(zip(*samples))
    
    assert(len(new_x) == len(z_prime) == len(pi) == 100)
    assert new_x[0].shape == ts.shape
    assert isinstance(pi[0], float)
    assert z_prime[0].shape == np.unique(segm_ts).shape
    
    # MTS
    samples = sampler.get_samples(mts, segm_mts, n_samples=100)
    new_x, z_prime, pi = list(zip(*samples))
    
    assert(len(new_x) == len(z_prime) == len(pi) == 100)
    assert new_x[0].shape == mts.shape
    assert isinstance(pi[0], float)
    assert z_prime[0].shape == np.unique(segm_mts).shape
    
def test_base_explain():
    # Arguments
    sampler = Perturbation()
    kernel =  Kernels.Lasso
    predict_fn = lambda x: 1
    lime = LimeBase()

    samples = sampler.get_samples(ts, segm_ts, n_samples=100)
    _kernel, score = lime._explain(samples, kernel, predict_fn)
    assert score == 1 and _kernel.coef_ is not None


    samples = sampler.get_samples(mts, segm_mts, n_samples=100)
    _kernel, score = lime._explain(samples, kernel, predict_fn)
    assert score == 1 and _kernel.coef_ is not None
    

def test_ts_explain():
    lime_ts = LimeTS(n_samples=1000)
    predict_fn = lambda x: x.sum()
    
    coef = lime_ts.explain(mts, predict_fn)
    assert all(coef.ravel() >= 0)
    assert coef.shape == mts.shape
    