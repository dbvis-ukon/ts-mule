import stumpy

import numpy as np

from abc import ABC, abstractmethod

from pyts.approximation import SymbolicAggregateApproximation


class AbstractSegmentation(ABC):
    """ Abstract Segmentation with abstract methods. """

    @abstractmethod
    def __init__(self):
        pass
    

    @abstractmethod
    def segment(self, time_series_sample):
        """ Time series instance segmentation into segments.

        Args:
            time_series_sample (ndarray): Time series data (n_steps, n_features)
            
        Returns:
            segmentation_mask: the segmentation mask of a time series. It has the same shape with time series sample.
        """
        pass


class MatrixProfileSegmentation(AbstractSegmentation):
    """ Matrix Profile Segmentation using a matrix profile on every feature. """

    def __init__(self, partitions, win_length=3):
        self.partitions = partitions
        self.win_length = max(3, win_length)

    def _segment_with_slopes(self, time_series_sample, m=4, k=10, profile='min'):
        """ Time series instance segmentation into segments.
        
        Idea:
         - Take the matrix profile of a time series and sort the distances.
         - Calculate the slope of this new matrix profile and take partition largest ones.
         
        Args:
            time_series_sample (ndarray): Time series data (n_steps, n_features)
            m (int, optional): Windows Size of subsequent to do matrix profile. Defaults to 4.
            k (int, optional): Initial max number of partitions. The final result is possiblily smaller than k paritions. Defaults to 10.
            profile (str, optional): Start the profile either at the minimas or the maximas ('min', 'max'). Defaults to 'min'.
            
        Returns:
            segmentation_mask: the segmentation mask of a time series. It has the same shape with time series sample.
        """
        # create segmentation mask as the time series
        segmentation_mask = np.zeros_like(time_series_sample)
        
        # extract steps and features
        n_steps, n_features = time_series_sample.shape
        
        # set matrix profile window length
        mp_win_len = m
        if mp_win_len == -1:
            # calculate partitions based matrix profile length
            mp_win_len = int(n_steps / k)
            
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
            slopes_sorted = np.argsort(slopes)[::-1][:k]
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
                
            win_idx += 1
            
        return segmentation_mask

    
    @staticmethod
    def _segment_with_bins(time_series_sample, m=4, k=10, distance_method='max'):
        """ The methods divides the matrix profile distance into bins. 
        
        For shared points between two windows, it can minimize or maximize the nearest distance.

        Args:
            time_series_sample (ndarray): Time series data (n_steps, n_features)
            m (int, optional): Windows Size of subsequent to do matrix profile. Defaults to 4.
            k (int, optional): Initial max number of partitions. The final result is possiblily smaller than k paritions. Defaults to 10.
            distance_method (str, optional): Minimize or maximize the shared points between two windows. Options can be `min`, `max`. Defaults to "max".

        Returns:
            segmentation_mask: the segmentation mask of a time series. It has the same shape with time series sample.
        """
        n_steps, n_features = time_series_sample.shape
        segmentation_mask = np.zeros_like(time_series_sample)
        seg_start = 0
        for feature in range(n_features):
            ts = time_series_sample[:, feature]
            
            # Get Matrix Profile Distance
            mp = stumpy.stump(ts, m)
            mp_d = mp[:, 0].astype(float)
            mp_d_min = min(0, mp_d.min())
            mp_d_max = mp_d.max()

            # Create bins of distance from min to max
            # segments number
            #   lower: more similar -> motif classes
            #   highest: more dissimilar -> discord classes
            bins = np.linspace(mp_d_min, mp_d_max, k)
            segments = np.digitize(mp_d, bins) - 1  # -1 to make start from 0
            segments = seg_start + segments 
            
            # unpack segments to time series
            #   Notice: For the shared points between two windows, the segment can be maximized, or minimized   
            if distance_method == 'max':
                init_v = min(segments)
                _fn = np.fmax
            if distance_method == 'min':
                init_v = max(segments)
                _fn = np.fmin

            seg_m = np.full(n_steps, init_v)
            for i, s in enumerate(segments):
                seg_m[i:i+m] = _fn(seg_m[i:i+m], s)
            
            segmentation_mask[:, feature] = seg_m
            seg_start = max(seg_m) + 1
        return segmentation_mask

    def segment(self, time_series_sample, segmentation_method='slopes'):
        """ Time series instance segmentation into segments.
        
        Currently only with slopes but more is planned.

        Args:
            time_series_sample (ndarray): Time series data (n_steps, n_features)
            
        Returns:
            segmentation_mask: the segmentation mask of a time series. It has the same shape with time series sample.
        """
        
        if segmentation_method == 'slopes-min':
            return self._segment_with_slopes(time_series_sample,
                                             m=self.win_length, 
                                             k=self.partitions, 
                                             profile='min')
        if segmentation_method == 'slopes-max':
            return self._segment_with_slopes(time_series_sample,
                                             m=self.win_length, 
                                             k=self.partitions, 
                                             profile='max')
        
        if segmentation_method == 'bins-max':
            return self._segment_with_bins(time_series_sample, 
                                           m=self.win_length, 
                                           k=self.partitions, 
                                           distance_method='max')
        if segmentation_method == 'bins-min':
            return self._segment_with_bins(time_series_sample, 
                                           m=self.win_length, 
                                           k=self.partitions, 
                                           distance_method='min')
        
class SAXSegmentation(AbstractSegmentation):
    """ SAX Segmentation using a  on every feature."""


    def __init__(self, partitions):
        self.partitions = partitions


    def segment(self, time_series_sample, **_kwargs):
        """ Time series instance segmentation into segments.
        
        Idea:
         - Segment data using the SAX transformation.
         - Use SAX to transform data.
         - Use transformed data to identify windows.

        Args:
            time_series_sample (ndarray): Time series data (n_steps, n_features)
            
        Returns:
            segmentation_mask: the segmentation mask of a time series. It has the same shape with time series sample.
        """
        # create segmentation mask as the time series
        segmentation_mask = np.zeros_like(time_series_sample)
        
        # set partition amount
        partitions = self.partitions

        # extract steps and features
        _, n_features = time_series_sample.shape
        
        # set first window index to 0
        win_idx = 0

        # create a sax transformation for every feature
        for feature in range(n_features):
            
            n_bins = 3

            internal_win_idx = 0
            while internal_win_idx < partitions * 9 / 10 or (internal_win_idx > partitions * 11 / 10 and internal_win_idx < partitions * 13 / 10):

                sax = SymbolicAggregateApproximation(n_bins=n_bins, strategy='quantile', alphabet='ordinal')
                sax_transformation = sax.fit_transform(time_series_sample[:, feature].reshape(1, -1))

                internal_win_idx = 0
                old_value = None
                for i, value in enumerate(sax_transformation.reshape(-1)):
                    if old_value and value != old_value:
                        win_idx += 1
                        internal_win_idx += 1
                    segmentation_mask[i, feature] = win_idx
                    old_value = value

                n_bins += 1
                
            win_idx += 1

        return segmentation_mask