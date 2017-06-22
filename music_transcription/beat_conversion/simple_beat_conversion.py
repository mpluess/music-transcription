import numpy as np
from math import log2, pow
from aubio import source, tempo
from music_transcription.fileformat.guitar_pro.utils import beat, note


# Simple Onset-To-Beat Converter
# Tempo can be estimated using aubio library or set manually
# Smallest note which should be considered can be set, default is 16th
# Beats per measure can be set, default is 4
# - no offset detection
# - only "full" notes are considered, no dotted, held (e.g. 3.5 quarters) nor triplets / n-tuples
# - last note is 1 beat or until the end of measure
class SimpleBeatConverter:
    def __init__(self):
        self.tempo = 0
        self.smallest_note = -2  # default: 16th
        self.beats_per_measure = 4  # default: assume 4/4

    def set_smallest_note(self, n):
        self.smallest_note = {1: 2, 2: 1, 4: 0, 8: -1, 16: -2, 32: -3, 64: -4}[n]

    def set_beats_per_measure(self, bpm):
        self.beats_per_measure = bpm

    def set_tempo(self, bpm):
        self.tempo = bpm

    # takes the median of the onset times
    def determine_tempo_from_onsets(self, onset_times_seconds):
        self.tempo = np.median(60. / np.diff(onset_times_seconds)).round()
        return self.tempo

    def determine_tempo_aubio(self, path_to_wav_file):
        src = source(path_to_wav_file)  # source(path, sample_rate=44100, hop_size=512)
        o = tempo("specdiff", 1024, src.hop_size, src.samplerate)

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
            self.tempo = np.median(bpm).round()
        return self.tempo

    def transform(self, path_to_wav_file,
                  onset_times_seconds, list_of_pitch_sets, list_of_string_lists, list_of_fret_lists):
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

        # default tempo detection method
        if self.tempo == 0:
            self.determine_tempo_aubio(path_to_wav_file)

        bps = self.tempo / 60  # beats per second
        beats = []
        current_measure = self._create_measure()
        m_len = 0  # length of current measure in beats (quarter notes)
        idx = 1
        for (onset, strings, frets) in zip(onset_times_seconds, list_of_string_lists, list_of_fret_lists):
            diff = 1  # default diff = 1 beat (e.g. for last note)
            if idx < len(onset_times_seconds):
                diff = onset_times_seconds[idx] - onset
            idx += 1
            beat_diff = min(diff*bps, self.beats_per_measure - m_len)  # difference in beats (quarter notes)
            gp5_duration = round(log2(beat_diff))  # e.g. if difference is 0.25 beats, log2(0.25) = -2
            if gp5_duration < self.smallest_note:  # ignore beats shorter than the smallest note
                continue
            m_len += pow(2, gp5_duration)
            gp5_duration *= -1  # flip sign as -2 is whole note and 2 is 16th, which is

            notes = [None, None, None, None, None, None, None]
            for string, fret in zip(strings, frets):
                notes[string] = note(fret)
            current_measure[0][0].append(beat(notes, duration=gp5_duration))
            if m_len >= self.beats_per_measure:
                m_len = 0
                beats.append(current_measure)
                current_measure = self._create_measure()
        beats.append(current_measure)

        return beats

    @staticmethod
    def _create_measure():
        return [([], [])]
