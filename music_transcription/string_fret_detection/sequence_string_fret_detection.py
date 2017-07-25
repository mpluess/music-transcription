import numpy as np

from music_transcription.string_fret_detection.abstract_string_fret_detector import AbstractStringFretDetector
from music_transcription.string_fret_detection.plausibility import get_all_fret_possibilities


class SequenceStringFretDetection(AbstractStringFretDetector):
    def __init__(self, tuning, n_frets):
        super().__init__(tuning, n_frets)

    def predict_strings_and_frets(self, path_to_wav_file, onset_times_seconds, list_of_pitch_sets):
        def d(chord_a, chord_b):
            non_empty_frets_a = [fret for fret in chord_a if fret > 0]
            non_empty_frets_b = [fret for fret in chord_b if fret > 0]
            if len(non_empty_frets_a) == 0 or len(non_empty_frets_b) == 0:
                return 0
            else:
                return abs(min(non_empty_frets_a) - min(non_empty_frets_b))

        chords_per_pitch_set = [get_all_fret_possibilities(pitch_set, tuning=self.tuning, n_frets=self.n_frets)
                                for pitch_set in list_of_pitch_sets]
        p_tab = [[1.0 for _ in chords] for chords in chords_per_pitch_set]
        d_tab = []
        for i in range(len(onset_times_seconds)-1):
            chords_a = chords_per_pitch_set[i]
            chords_b = chords_per_pitch_set[i+1]
            d_matrix = []
            for chord_a in chords_a:
                d_matrix_row = []
                for chord_b in chords_b:
                    d_matrix_row.append(d(chord_a, chord_b))
                d_matrix.append(d_matrix_row)
            d_tab.append(np.array(d_matrix))
        p_d_tab = [1 - d_matrix/self.n_frets for d_matrix in d_tab]

        print('')
