import aubio
import numpy as np

from music_transcription.tempo_detection.abstract_tempo_detector import AbstractTempoDetector


class AubioTempoDetector(AbstractTempoDetector):
    """ Estimates the tempo of a piece using the aubio library (https://aubio.org/) """

    def __init__(self):
        super().__init__()

    def predict(self, path_to_wav_file, onset_times_seconds):
        src = aubio.source(path_to_wav_file)  # source(path, sample_rate=44100, hop_size=512)
        o = aubio.tempo("specdiff", 1024, src.hop_size, src.samplerate)

        beats = []  # list of beats in samples
        while True:
            samples, read = src()
            is_beat = o(samples)
            if is_beat:
                this_beat = o.get_last_s()
                beats.append(this_beat)
            if read < src.hop_size:
                break

        # Convert to periods and to bpm
        if len(beats) > 1:
            bpm = 60. / np.diff(beats)
            tempo = np.median(bpm).round()
        else:
            tempo = 120  # default fallback

        return tempo
