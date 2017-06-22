class SimpleStringFretDetection:
    def __init__(self, tuning, n_frets):
        self.tuning = tuning
        self.n_frets = n_frets
        self.min_pitch = min(tuning)
        self.max_pitch = max(tuning) + n_frets

    def predict_strings_and_frets(self, path_to_wav_file, onset_times_seconds, list_of_pitch_sets):
        """Assumes guitar with 6 strings.
        The returned string and fret lists have exactly the same shapes.
        String and fret information at the same indices belong together.

        Returns
        -------
        list_of_string_lists : list of list
            List (len = len(onset_times_seconds) = len(list_of_pitch_sets))
            of lists (0 < len <= 6) of string numbers,
            with 0 being the string with the highest pitch
            and 5 the one with the lowest pitch.
        list_of_fret_lists : list of list
            List (len = len(onset_times_seconds) = len(list_of_pitch_sets))
            of lists (0 < len <= 6) of fret numbers,
            with 0 zero being the empty string.
        """

        list_of_string_lists = []
        list_of_fret_lists = []
        for pitch_set in list_of_pitch_sets:
            strings_per_onset = []
            frets_per_onset = []
            for pitch in sorted(pitch_set, reverse=True):
                if pitch > self.max_pitch:
                    raise ValueError(
                        'Impossible pitch: {} is above {} which is impossible in the tuning {}.'.format(
                            pitch, self.max_pitch, self.tuning
                        )
                    )
                elif pitch >= self.tuning[0]:
                    strings_per_onset.append(0)
                    frets_per_onset.append(pitch - self.tuning[0])
                elif pitch >= self.tuning[1]:
                    strings_per_onset.append(1)
                    frets_per_onset.append(pitch - self.tuning[1])
                elif pitch >= self.tuning[2]:
                    strings_per_onset.append(2)
                    frets_per_onset.append(pitch - self.tuning[2])
                elif pitch >= self.tuning[3]:
                    strings_per_onset.append(3)
                    frets_per_onset.append(pitch - self.tuning[3])
                elif pitch >= self.tuning[4]:
                    strings_per_onset.append(4)
                    frets_per_onset.append(pitch - self.tuning[4])
                elif pitch >= self.tuning[5]:
                    strings_per_onset.append(5)
                    frets_per_onset.append(pitch - self.tuning[5])
                else:
                    raise ValueError(
                        'Impossible pitch: {} is below {} which is impossible in the tuning {}.'.format(
                            pitch, self.min_pitch, self.tuning
                        )
                    )

            list_of_string_lists.append(strings_per_onset)
            list_of_fret_lists.append(frets_per_onset)

        return list_of_string_lists, list_of_fret_lists
