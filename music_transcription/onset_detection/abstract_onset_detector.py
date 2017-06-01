from abc import ABCMeta, abstractmethod


class AbstractOnsetDetector(metaclass=ABCMeta):
    def __init__(self, onset_group_threshold_seconds):
        """Base constructor

        Parameters
        ----------
        onset_group_threshold_seconds : float
            (Polyphonic) onsets which are closer than or equal to onset_group_threshold_seconds are grouped together
            to one onset (a chord).
        """

        self.config = {
            'onset_group_threshold_seconds': onset_group_threshold_seconds
        }

    @abstractmethod
    def predict_onsets(self, path_to_wav_file):
        """Predicts onsets and returns a list of onset times in seconds.
        If the wave file is corrupt / not readable, returns None.

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
