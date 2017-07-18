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
    def __init__(self, tempo_bpm=-1, shortest_note=0.25, beats_per_measure=4.0):
        self.tempo = tempo_bpm  # negative = try to determine from file
        self.shortest_note = shortest_note  # default: 16th
        self.beats_per_measure = beats_per_measure  # default: 4/4

    def split_beat_to_gp5_durations(self, beats):
        gp5_durations = []
        cur_duration = 4.0
        while beats > 0 and cur_duration >= self.shortest_note:
            if cur_duration <= self.beats_per_measure and cur_duration <= beats:
                gp5_durations.append(self.convert_duration_beats_to_gp5(cur_duration))
                beats -= cur_duration
            cur_duration /= 2
        return gp5_durations

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
        else:
            self.tempo = 120  # default fallback
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
        if self.tempo < 0:
            self.tempo = self.determine_tempo_aubio(path_to_wav_file)

        bps = self.tempo / 60  # beats per second
        beats = []
        current_measure = self._create_measure()
        m_len = 0.0  # length of current measure in beats (quarter notes)
        idx = 1
        for (onset, strings, frets) in zip(onset_times_seconds, list_of_string_lists, list_of_fret_lists):
            swap = 0  # notes ringing into next measure
            m_remaining = self.beats_per_measure - m_len

            beat_diff = m_remaining  # default duration = rest of measure (e.g. for last note)
            if idx < len(onset_times_seconds):
                # get time difference to next onset, calc difference in beats (quarter notes)
                beat_diff = (onset_times_seconds[idx] - onset) * bps
            idx += 1

            beat_diff = round(beat_diff / self.shortest_note) * self.shortest_note  # round duration
            if beat_diff == 0:
                continue

            if m_remaining < beat_diff:  # cannot be longer than until the end of current measure
                swap = beat_diff - m_remaining
                beat_diff = m_remaining

            # get played strings and frets
            notes = [None] * 7
            notes_tied = [None] * 7
            for string, fret in zip(strings, frets):
                notes[string] = note(fret)
                notes_tied[string] = note(fret, tied=True)

            gp5_durations = self.split_beat_to_gp5_durations(beat_diff)
            while swap > 0:
                swap_beats = min(swap, self.beats_per_measure)
                swap -= swap_beats
                gp5_durations += self.split_beat_to_gp5_durations(swap_beats)

            tied = False
            for gp5_duration in gp5_durations:
                m_len += self.convert_duration_gp5_to_beats(gp5_duration)

                current_measure[0][0].append(beat(notes_tied if tied else notes, duration=gp5_duration))
                tied = True
                if m_len >= self.beats_per_measure:
                    m_len = 0
                    beats.append(current_measure)
                    current_measure = self._create_measure()
        beats.append(current_measure)

        return beats

    @staticmethod
    def convert_duration_beats_to_gp5(beats):
        return round(log2(beats)) * (-1)  # e.g. if beat duration is 0.25 beats -> log2(0.25) = -2 -> flip signs = 2

    @staticmethod
    def convert_duration_gp5_to_beats(gp5_duration):
        return pow(2, gp5_duration*(-1))

    @staticmethod
    def _create_measure():
        return [([], [])]
