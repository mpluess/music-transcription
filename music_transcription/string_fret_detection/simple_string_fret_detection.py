from music_transcription.string_fret_detection.abstract_string_fret_detector import AbstractStringFretDetector


class SimpleStringFretDetection(AbstractStringFretDetector):
    def __init__(self, tuning, n_frets):
        super().__init__(tuning, n_frets)

    def predict_strings_and_frets(self, path_to_wav_file, onset_times_seconds, list_of_pitch_sets):
        """Assumes guitar with 6 strings."""

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
