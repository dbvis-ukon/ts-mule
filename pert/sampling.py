import stumpy

import numpy as np

from abc import ABC, abstractmethod

from tslearn.piecewise import PiecewiseAggregateApproximation
from tslearn.piecewise import SymbolicAggregateApproximation


class AbstractSegmentation(ABC):
    """Abstract Segmentation with abstract methods."""

    @abstractmethod
    def __init__(self):
        pass
    

    @abstractmethod
    def segment(self, time_series_sample):
        """ Time series instance segmentation into segments.

        :param time_series_sample: (np.array) time series must be (n_steps, n_features)
        """
        pass


class MatrixProfileSegmentation(AbstractSegmentation):
    """ Matrix Profile Segmentation using a matrix profile on every feature."""

    def __init__(self, partitions, win_length=-1):
        self.partitions = partitions
        self.win_length = win_length


    def _segment_with_slopes(self, time_series_sample):
        """ Time series instance segmentation into segments.
        
        Idea:
         - Take the matrix profile of a time series and sort the distances.
         - Calculate the slope of this new matrix profile and take partition largest ones.

        :param time_series_sample: (np.array) time series must be (n_steps, n_features)
        """
        # create segmentation mask as the time series
        segmentation_mask = np.zeros_like(time_series_sample)
        
        # extract steps and features
        n_steps, n_features = time_series_sample.shape
        
        # set matrix profile window length
        mp_win_len = self.win_length
        if mp_win_len == -1:
            # calculate partitions based matrix profile length
            mp_win_len = int(n_steps / self.partitions)
            
        # set first window index to 0
        win_idx = 0
        
        # create a matrix profile for every feature
        for feature in range(n_features):
            
            # extract matrix profile with the previously set window length
            mp = stumpy.stump(time_series_sample[:, feature], mp_win_len)
            mp_ = mp[:, 0] # just take the matrix profile
            mp_sorted = sorted(mp_) # sort values
            mp_idx_sorted = np.argsort(mp_) # sort indeces with values
        
            # find the largest matrix profile slopes
            # calculate the slopes for every matrix profile step
            slopes = np.array([(mp_sorted[i] - mp_sorted[i + 1]) / (i - (i + 1)) for i in range(len(mp_sorted) - 1)])
            # take amount of partitions of the largest slopes
            slopes_sorted = np.argsort(slopes)[::-1][:self.partitions]
            # retrieve indeces of original time series
            partitions_idx_sorted = sorted([mp_idx_sorted[part] for part in slopes_sorted])
            # add end of time series
            partitions_idx_sorted.append(n_steps)
            
            # create windows segmentation masks
            start = 0
            for idx in partitions_idx_sorted:
                end = idx
                segmentation_mask[start:end, feature] = win_idx
                win_idx += 1
                start = idx
            
        return segmentation_mask

        
    def segment(self, time_series_sample, segmentation_method='slopes'):
        """ Time series instance segmentation into segments.
        
        Currently only with slopes but more is planned.

        :param time_series_sample: (np.array) time series must be (n_steps, n_features)
        """
        
        if segmentation_method == 'slopes':
            return self._segment_with_slopes(time_series_sample)
        
        
class SAXSegmentation(AbstractSegmentation):
    """ SAX Segmentation using a  on every feature."""

    def __init__(self, partitions, win_length=-1):
        self.partitions = partitions
        self.win_length = win_length

    def segment(self, time_series_sample):
        """ Time series instance segmentation into segments.
        
        Idea:
         - Segment data using the SAX transformation.
         - Use SAX to transform data.
         - Use transformed data to identify windows.

        :param time_series_sample: (np.array) time series must be (n_steps, n_features)
        """
        # create segmentation mask as the time series
        segmentation_mask = np.zeros_like(time_series_sample)
        
        # set segements to the partition amount
        # set SAX symbols amount to half of the segments
        n_paa_segments = self.partitions
        n_sax_symbols = int(n_paa_segments / 2)

        # extract steps and features
        _, n_features = time_series_sample.shape
        
        # set first window index to 0
        win_idx = 0

        # create a sax transformation for every feature
        for feature in range(n_features):

            # create SAX symbols for the previously set parameters
            sax = SymbolicAggregateApproximation(n_segments=n_paa_segments, alphabet_size_avg=n_sax_symbols)
            # do the SAX transform for the data
            sax_transformation = sax.fit_transform(time_series_sample[:, feature].reshape(1, -1))
            # inverse the SAX transformation to find the windows
            sax_transformation_inv = sax.inverse_transform(sax_transformation)

            # create the segmentation mask
            old_value = 0
            for i, value in enumerate(sax_transformation_inv.reshape(-1)):
                if old_value != 0 and value != old_value:
                    win_idx += 1
                segmentation_mask[i, feature] = win_idx
                old_value = value

        return segmentation_mask
