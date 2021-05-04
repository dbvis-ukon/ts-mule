import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sampling import MatrixProfileSegmentation

url = "https://zenodo.org/record/4273921/files/STUMPY_Basics_steamgen.csv?download=1"
df = pd.read_csv(url)
n = 32
ts = df["steam flow"][:n]
ts2 = df["drum pressure"][:n]
mts = np.stack([ts, ts2], axis=1)


@pytest.mark.skip("Manual test.")
def test_segment_max():
    mseg = MatrixProfileSegmentation(partitions=10, win_length=4)
    
    seg_m = mseg._segment_with_bins(ts.values.reshape(-1, 1), 4, 10, "max")
    plt.scatter(range(n), ts, c=seg_m)
    plt.show()
    
    seg_m = mseg._segment_with_bins(ts.values.reshape(-1, 1), 4, 10, "min")
    plt.scatter(range(n), ts, c=seg_m)
    plt.show()
    
    seg_mts = mseg._segment_with_bins(mts, 4, 10, "min")
    fig, ax = plt.subplots(2,1)
    ax[0].scatter(range(n), mts[:, 0], c=seg_mts[:,0])
    ax[1].scatter(range(n), mts[:, 1], c=seg_mts[:,1])
    plt.show()