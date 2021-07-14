"""Visualization supporting tools."""
import matplotlib.pyplot as plt


def visualize_segmentation_mask(time_series_sample, segmentation_mask):
    """Visualize time series and its segementation.

    Args:
        time_series_sample (ndarrays): A time series with shape n_steps, n_features
        segmentation_mask (ndarrays): A generated segmentation of the time series.
    """
    n_steps, n_features = time_series_sample.shape

    _, ax = plt.subplots(n_features, 1)
    for i in range(n_features):
        ax[i].plot(time_series_sample[:, i])
        old_value = segmentation_mask[0, i]
        for j in range(n_steps):
            value = segmentation_mask[j, i]
            if old_value != value:
                ax[i].axvline(x=j, color='red')
                old_value = value
    plt.show()

    _, ax = plt.subplots(n_features, 1)
    for i in range(n_features):
        ax[i].scatter(range(n_steps), time_series_sample[:, i],  c=segmentation_mask[:, i])
    plt.show()


def visualize_perturbation_masks():
    pass
