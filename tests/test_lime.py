"""Test module for LIME in generate explainations."""

import numpy as np
from sampling import Perturbation
from xai.lime import BaseLIME, Kernels


ts = np.array([[1, 3, 9, 5, 4, 6, 7, 5, 9, 2, 6, 6, 7, 4, 0, 0]]).T
mts = np.array([[1, 3, 9, 5, 4, 6, 7, 5, 9, 2, 6, 6, 7, 4, 0, 0],
                [5, 9, 6, 0, 5, 8, 8, 1, 0, 2, 5, 4, 4, 5, 8, 0]]).T

segm_ts = np.array([[0, 0, 0, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 2]]).T
segm_ts2 = np.array([[7, 7, 7, 7, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6]]).T
segm_mts = np.stack([segm_ts.ravel(), segm_ts2.ravel()], axis=1)



def test_explain():
    # Arguments
    sampler = Perturbation()
    samples = sampler.get_samples(mts, segm_mts, n_samples=100)
    kernel =  Kernels.Lasso
    predict_fn = lambda x: 1

    lime = BaseLIME()
    t = lime._exp_explain(samples, kernel, predict_fn)
    t.coef_
