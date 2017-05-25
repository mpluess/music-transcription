from abc import ABCMeta, abstractmethod


class AbstractOnsetDetector(metaclass=ABCMeta):
    def __init__(self):
        """Base constructor"""

        pass

    @abstractmethod
    def predict_onsets(self, path_to_wav_file):
        """Predict onsets and return a list of onset times in seconds.

        Parameters
        ----------
        path_to_wav_file : str
            Path to wave file for which the onsets should be predicted

        Returns
        -------
        list of float
            List of onset times in seconds
            List is sorted by time.
            Onsets closer than self.config['onset_group_threshold_seconds'] merged to one onset.
        """

        pass
