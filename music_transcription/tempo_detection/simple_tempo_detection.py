import numpy as np

from music_transcription.tempo_detection.abstract_tempo_detector import AbstractTempoDetector


class SimpleTempoDetector(AbstractTempoDetector):
    def __init__(self):
        super().__init__()

    def predict(self, path_to_wav_file, onset_times_seconds):
        """Takes the median of the onset times"""

        return np.median(60. / np.diff(onset_times_seconds)).round()
