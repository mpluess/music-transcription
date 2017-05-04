from music_transcription.fileformat.guitar_pro.utils import beat, note


class SimpleBeatConverter:
    def __init__(self):
        self.tempo = None

    def fit(self, path_to_wav_file):
        self.tempo = 100

    def transform(self, path_to_wav_file, onset_times_seconds, pitches, strings, frets):
        """Assumes 4/4 measure."""

        # beats = [  # measures
        #     [  # measure, tracks
        #         (  # track, 2 voices
        #             [  # voice 1, beats / onsets
        #
        #             ],
        #             []
        #         ),
        #     ],
        # ]
        beats = []
        current_measure = None
        for i, (onset, pitch, string, fret) in enumerate(zip(onset_times_seconds, pitches, strings, frets)):
            if i % 4 == 0:
                if current_measure is not None:
                    beats.append(current_measure)
                current_measure = self._create_measure()
            notes = [None, None, None, None, None, None, None]
            notes[string] = note(fret)
            current_measure[0][0].append(beat(notes))
        beats.append(current_measure)

        return beats

    @staticmethod
    def _create_measure():
        return [([], [])]
