class SimpleStringFretDetection:
    def __init__(self, tuning, n_frets):
        self.tuning = tuning
        self.n_frets = n_frets

    def predict_strings_and_frets(self, path_to_wav_file, onset_times_seconds, pitches):
        """Assumes guitar with 6 strings.

        Output: (strings, frets)
        strings: list of string numbers, with 0 being the string with the highest pitch and 5 the one with the lowest pitch
        frets: list of fret numbers, with 0 zero being the empty string
        """

        max_pitch = self.tuning[0] + self.n_frets
        strings = []
        frets = []
        for pitch in pitches:
            if pitch > max_pitch:
                raise ValueError(
                    'Impossible pitch: {} is above {} which is impossible in the tuning {}.'.format(
                        pitch, max_pitch, self.tuning
                    )
                )
            elif pitch >= self.tuning[0]:
                strings.append(0)
                frets.append(pitch - self.tuning[0])
            elif pitch >= self.tuning[1]:
                strings.append(1)
                frets.append(pitch - self.tuning[1])
            elif pitch >= self.tuning[2]:
                strings.append(2)
                frets.append(pitch - self.tuning[2])
            elif pitch >= self.tuning[3]:
                strings.append(3)
                frets.append(pitch - self.tuning[3])
            elif pitch >= self.tuning[4]:
                strings.append(4)
                frets.append(pitch - self.tuning[4])
            elif pitch >= self.tuning[5]:
                strings.append(5)
                frets.append(pitch - self.tuning[5])
            else:
                raise ValueError(
                    'Impossible pitch: {} is below {} which is impossible in the tuning {}.'.format(
                        pitch, self.tuning[5], self.tuning
                    )
                )

        return strings, frets
