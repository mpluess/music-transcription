import random


class RandomPitchDetector:
    def __init__(self, tuning, n_frets):
        self.tuning = tuning
        self.n_frets = n_frets

    def predict_pitches_monophonic(self, path_to_wav_file, onset_times_seconds):
        return [random.randint(self.tuning[5], self.tuning[0] + self.n_frets)
                for _ in range(len(onset_times_seconds))]
