import matplotlib.pyplot as plt

def visualize_segmentation_mask(time_series_sample, segmentation_mask):
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
        ax[i].scatter(range(n_steps), time_series_sample[:, i], c=segmentation_mask[:, i])
    plt.show()
