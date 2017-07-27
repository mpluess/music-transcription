import numpy as np
from pyswarm import pso

from music_transcription.string_fret_detection.abstract_string_fret_detector import AbstractStringFretDetector
from music_transcription.string_fret_detection.plausibility import get_all_fret_possibilities


class SequenceStringFretDetection(AbstractStringFretDetector):
    def __init__(self, tuning, n_frets):
        super().__init__(tuning, n_frets)

    def predict_strings_and_frets(self, path_to_wav_file, onset_times_seconds, list_of_pitch_sets):
        # TODO More sophisticated distance function
        def d(chord_a_, chord_b_):
            """Distance function of two chords"""

            non_empty_frets_a = [fret for fret in chord_a_ if fret > 0]
            non_empty_frets_b = [fret for fret in chord_b_ if fret > 0]
            if len(non_empty_frets_a) == 0 or len(non_empty_frets_b) == 0:
                return 0
            else:
                return abs(min(non_empty_frets_a) - min(non_empty_frets_b))

        def f(x, *args):
            """Objective function for PSO

            Parameters
            ----------
            x : list of float
            args

            Returns
            -------
            float
                1 - p because pyswarm always minimizes the objective function
            """

            p_tab_ = args[0]
            p_d_tab_ = args[1]

            p = 1.0
            # Multiply probabilities for all chords in the current sequence defined by x
            for i_onset, i_chord in enumerate(x):
                # PSO works with floats --> convert to int index
                i_chord = to_discrete(i_chord)
                p *= p_tab_[i_onset][i_chord]
            # Multiply probabilities of distances between chords
            for i_onset in range(len(x) - 1):
                i_chord_a = to_discrete(x[i_onset])
                i_chord_b = to_discrete(x[i_onset + 1])
                # Example: i_onset 0 = numpy distance matrix between chord 0 and chord 1
                p *= p_d_tab_[i_onset][i_chord_a, i_chord_b]

            # Invert p because pyswarm minimizes the function value while we want to maximize p
            return 1.0 - p

        def to_discrete(f_):
            return int(f_)

        # TODO PSO might work better if possibilities are sorted, e.g. by probability or by position on the guitar.
        # TODO We might want to filter possibilities with low probability so there are fewer possible sequences.
        # TODO Handle edge case where there is no possibility at all for a pitch set.
        chords_per_pitch_set = [get_all_fret_possibilities(pitch_set, tuning=self.tuning, n_frets=self.n_frets)
                                for pitch_set in list_of_pitch_sets]
        # TODO Get proper chord probabilities from get_all_fret_possibilities
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
        # Probability of a distance = 1.0 - d/n_frets (1.0 for d = 0)
        p_d_tab = [1.0 - d_matrix/self.n_frets for d_matrix in d_tab]

        # Lower bounds of the parameters (= chord indices) to be optimized
        lb = [0.0] * len(chords_per_pitch_set)
        # Upper bounds of the parameters.
        # Example: 7 possibilites for pitch set 0 -> upper bound = 6.99 -> to_discrete(6.99) = max index = 6
        ub = [len(chords) - 0.01 for chords in chords_per_pitch_set]
        # Get the optimal sequence maximizing the sequence probability defined by f.
        # Since pyswarm always minimizes the function, f returns the inverse of the probability (1-p).
        # noinspection PyTypeChecker
        xopt, fopt = pso(f, lb, ub, args=(p_tab, p_d_tab), debug=True)
        print('p={}'.format(1.0 - fopt))

        optimal_chords = [chords_per_pitch_set[i_onset][to_discrete(i_chord)] for i_onset, i_chord in enumerate(xopt)]
        list_of_string_lists = []
        list_of_fret_lists = []
        for chords in optimal_chords:
            strings_per_onset = []
            frets_per_onset = []
            for i, fret in enumerate(chords):
                if fret > -1:
                    strings_per_onset.append(i)
                    frets_per_onset.append(fret)
            list_of_string_lists.append(strings_per_onset)
            list_of_fret_lists.append(frets_per_onset)

        return list_of_string_lists, list_of_fret_lists
