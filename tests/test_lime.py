""" Test module for LIME in generate explainations."""
import numpy as np
import pytest

from tsmule.sampling.perturb import Perturbation
from tsmule.xai.lime import Kernels, LimeBase, LimeTS


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
    kernel = Kernels.Lasso

    def predict_fn(x):
        return 1

    lime = LimeBase()

    samples = sampler.get_samples(ts, segm_ts, n_samples=100)
    _kernel, score = lime._explain(samples, kernel, predict_fn)
    assert score == 1 and _kernel.coef_ is not None

    samples = sampler.get_samples(mts, segm_mts, n_samples=100)
    _kernel, score = lime._explain(samples, kernel, predict_fn)
    assert score == 1 and _kernel.coef_ is not None


def test_ts_explain():
    lime_ts = LimeTS(n_samples=1000)

    def predict_fn(x):
        return x.sum()

    coef = lime_ts.explain(mts, predict_fn)
    assert all(coef.ravel() >= 0)
    assert coef.shape == mts.shape


@pytest.mark.skip("Manuel test")
def test_explain_cnn():
    import dill
    import matplotlib.pyplot as plt
    from tensorflow import keras

    data_dir = "demo/beijing_air_2_5"
    cnn_model = keras.models.load_model(
        f'{data_dir}/beijing_air_2_5_cnn_model.h5')
    with open(f'{data_dir}/beijing_air_2_5_test_data.dill', 'rb') as f:
        dataset_test = dill.load(f)

    # Define a predict fn/model
    def predict_(x):
        return cnn_model.predict(x[np.newaxis]).ravel()

    # Get a sample
    sample = dataset_test[0][0]

    # First insight of the sample
    n_steps, features = sample.shape
    fig = plt.figure()
    for i in range(features):
        fig.add_subplot(features, 1, i + 1)
        plt.plot(sample[:, i])
    plt.show()

    # Define an explainer
    explainer = LimeTS()

    # Segmentation with slopes-max (default)
    seg_m = explainer._segmenter.segment(sample, "slopes-max")
    fig = plt.figure()
    for i in range(features):
        fig.add_subplot(features, 1, i + 1)
        plt.scatter(range(n_steps), sample[:, i], c=seg_m[:, i])
    plt.show()

    # Samples and binary
    perturbed_samples = explainer._sampler.perturb(sample, seg_m)
    new_s, z_prime, pi = next(perturbed_samples)
    fig = plt.figure()
    for i in range(features):
        fig.add_subplot(features, 1, i + 1)
        plt.plot(new_s[:, i])
    plt.show()

    # Plot an on/off binary vector
    fig, ax = plt.subplots()
    ax.imshow(z_prime.reshape(-1, features).T)
    plt.show()

    # Explain the model
    from sklearn import linear_model
    explainer = LimeTS(n_samples=100)

    Lasso = linear_model.Lasso(alpha=.01)
    explainer._kernel = Lasso
    xcoef = explainer.explain(sample, predict_)

    fig = plt.figure()
    for i in range(features):
        fig.add_subplot(features, 1, i + 1)
        plt.scatter(range(n_steps), xcoef[:, i], c=seg_m[:, i], marker="*")
    plt.show()
