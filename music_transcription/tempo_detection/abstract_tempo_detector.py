from abc import ABCMeta, abstractmethod


class AbstractTempoDetector(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def predict(self, path_to_wav_file, onset_times_seconds):
        """Predicts tempo of a recording.

        Parameters
        ----------
        path_to_wav_file : str
            Path to wave file of recording
        onset_times_seconds : list of float
            List of onset times in seconds

        Returns
        -------
        int
            Tempo in bpm (beats per minute)
        """

        pass
