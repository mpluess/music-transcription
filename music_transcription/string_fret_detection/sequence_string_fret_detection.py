import numpy as np
from copy import deepcopy
from heapq import heappush, heappop

from music_transcription.string_fret_detection.abstract_string_fret_detector import AbstractStringFretDetector
from music_transcription.string_fret_detection.plausibility import get_all_fret_possibilities, get_chord_probability


class SequenceStringFretDetection(AbstractStringFretDetector):
    def __init__(self, tuning, n_frets, complexity=50):
        """ String and fret detection based on the whole sequence to be written
        optimizes the resulting sequence so there are as few hand position changes as possible
        
        Parameters
        ----------
        tuning : tuple of :int:
            Tuple with length n_strings describing the pitches of the empty strings, in descending order.
        n_frets : int
            Number of frets of the guitar.
        complexity: int, optional
            maximum number of possible sequences to keep per iteration. Default: 50.
            The number of possible sequences that are tested grows linear with complexity and number of onsets.
        """
        super().__init__(tuning, n_frets)
        self.complexity = max(complexity, len(tuning))  # number of possibilities to keep

    def get_fret_distance_with_update_fret_pos(self, chord, fret_pos, duration):
        """ calculates the fret distance from one chord to the next and updates finger/fret positions
        
        Parameters
        ----------
        chord: list of :int:
            note / chord that is played
        fret_pos: (int, int)
            min and max fret positions that are possible to be within the current finger position
        duration: int
            duration in 100ms

        Returns
        -------
        (float, (int, int))
            fret distance and updated finger position
        """

        min_fret_a, max_fret_a = fret_pos
        non_empty_frets = [fret for fret in chord if fret > 0]

        if len(non_empty_frets) > 0:
            min_fret_b = min(non_empty_frets)
            max_fret_b = max(non_empty_frets)
            padding = max(0, 3 - (max_fret_b - min_fret_b))
            if min_fret_b > 14:  # on high frets one fret can easily be skipped -> extend range!
                padding += 1

            if max_fret_b > max_fret_a:  # slide-up
                fret_pos_ = (max(0, min_fret_b - padding), max_fret_b)
                return max_fret_b - max_fret_a, fret_pos_
            if min_fret_b < min_fret_a:  # slide-down
                fret_pos_ = (min_fret_b, min(self.n_frets, max_fret_b + padding))
                return min_fret_a - min_fret_b, fret_pos_

            fret_pos_ = (max(min_fret_a, min_fret_b - padding), min(max_fret_a, max_fret_b + padding))
            return 0, fret_pos_

        else:
            # widen finger possibilities for every 0 stroke -- widen more for long pauses (+1 every 100ms)
            fret_pos_ = (max(0, min_fret_a - duration), min(self.n_frets, max_fret_a + duration))
            return min_fret_a * 0.2, fret_pos_  # penalty for empty strings (higher penalty for high chords)

    def get_string_distance(self, chord_a, chord_b):
        """ calculate string distance: 1.0 for maximum change, 0.0 for no change """
        active_strings_a = [j for j in range(len(chord_a)) if chord_a[j] >= 0]
        active_strings_b = [j for j in range(len(chord_b)) if chord_b[j] >= 0]
        return np.abs(np.median(active_strings_a) - np.median(active_strings_b)) / (len(self.tuning) - 1)

    def get_next_chord_distance(self, chord_a, chord_b, fret_pos, duration_b):
        """ Distance function of two chords from 0.0 (no distance) to 1.0 (max distance)
        
        Parameters
        ----------
        chord_a: list of :int:
            previous chord
        chord_b: list of :int:
            current chord
        fret_pos: (int, int)
            min and max fret positions that are possible to be within the current finger position
        duration_b: int
            duration of current note/chord in 100ms

        Returns
        -------
        (float, (int, int))
            Distance of two chords from 0.0 (no distance) to 1.0 (max distance) and updated finger position
        """

        fret_distance, fret_pos_new = self.get_fret_distance_with_update_fret_pos(chord_b, fret_pos, duration_b)
        string_distance = self.get_string_distance(chord_a, chord_b)

        # reward similar chords
        non_empty_frets_a = [fret for fret in chord_a if fret > 0]
        non_empty_frets_b = [fret for fret in chord_b if fret > 0]
        if len(non_empty_frets_a) == len(non_empty_frets_b):
            reward = True
            for j in range(1, len(non_empty_frets_a)):
                reward = reward and non_empty_frets_a[j - 1] - non_empty_frets_a[j] == \
                                    non_empty_frets_b[j - 1] - non_empty_frets_b[j]
            if reward:
                fret_distance /= 2
                string_distance /= 2

        string_distance = min(1.0, (string_distance * 1.2)**1.5)  # lower impact on just 1 string change
        # fret_distance = min(1.0, (fret_distance / self.n_frets * 1.4)**1.2)  # lower impact on just few fret change
        fret_distance = min(1.0, fret_distance * 2 / 3 / self.n_frets)
        return fret_distance * 0.8 + string_distance * 0.2, fret_pos_new

    def predict_strings_and_frets(self, path_to_wav_file, onset_times_s, list_of_pitch_sets):
        # init priority queue
        pq = []
        for chord in get_all_fret_possibilities(list_of_pitch_sets[0], tuning=self.tuning, n_frets=self.n_frets):
            p = get_chord_probability(chord, self.n_frets)
            seq = [chord]
            heappush(pq, (-p, seq, (0, self.n_frets)))  # use negative p for max heap

        for i in range(1, len(onset_times_s)):
            duration_a = max(1, int((onset_times_s[i] - onset_times_s[i+1]) * 10)) if i < len(onset_times_s)-1 else 1
            pq_new = []
            chords = [(chord, get_chord_probability(chord, self.n_frets))
                      for chord in get_all_fret_possibilities(list_of_pitch_sets[i], self.tuning, self.n_frets)]

            n = self.complexity
            while n > 0 and pq:  # pq is "True" as long as there are elements in it
                p, seq, fret_pos = heappop(pq)
                p *= -1  # invert back (for max heap)
                last_chord = seq[len(seq) - 1]
                for chord, p_chord in chords:
                    p_c = p * p_chord
                    d, fret_pos_c = self.get_next_chord_distance(last_chord, chord, fret_pos, duration_a)
                    p_c *= 1.0 - d
                    heappush(pq_new, (-p_c, deepcopy(seq) + [chord], fret_pos_c))
                n -= 1
            pq = pq_new

        p_opt, optimal_chords, _ = heappop(pq)
        p_opt *= -1  # invert back (for max heap)

        # assemble string and fret lists
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
