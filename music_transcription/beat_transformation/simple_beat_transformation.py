from math import log2, pow

from music_transcription.beat_transformation.abstract_beat_transformer import AbstractBeatTransformer
from music_transcription.fileformat.guitar_pro.utils import Beat, Note


class SimpleBeatTransformer(AbstractBeatTransformer):
    """Simple onset-to-beat transformer

    Tempo can be estimated using aubio library or set manually
    Smallest note which should be considered can be set, default is 16th
    Beats per measure can be set, default is 4
    - no offset detection
    - only "full" notes are considered, no dotted, held (e.g. 3.5 quarters) nor triplets / n-tuples
    - last note is until the end of measure
    """

    def __init__(self, shortest_note=0.25, beats_per_measure=4.0):
        super().__init__()
        self.shortest_note = shortest_note  # default: 16th
        self.beats_per_measure = beats_per_measure  # default: 4/4

    def transform(self, path_to_wav_file, onset_times_seconds, list_of_string_lists, list_of_fret_lists, tempo):
        bps = tempo / 60  # beats per second
        beats = []
        current_measure = self._create_measure()
        m_len = 0.0  # length of current measure in beats (quarter notes)
        idx = 1
        for (onset, strings, frets) in zip(onset_times_seconds, list_of_string_lists, list_of_fret_lists):
            swap = 0  # notes ringing into next measure
            m_remaining = self.beats_per_measure - m_len  # remaining beats that fit in current measure

            beat_diff = m_remaining  # default duration = rest of measure (e.g. for last note)
            if idx < len(onset_times_seconds):
                # get time difference to next onset, calc difference in beats (quarter notes)
                beat_diff = (onset_times_seconds[idx] - onset) * bps
            idx += 1

            beat_diff = round(beat_diff / self.shortest_note) * self.shortest_note  # round duration
            if beat_diff == 0:  # don't do anything if the note was too short to be of self.shortest_note
                continue

            if m_remaining < beat_diff:  # if a note is longer than until the end of current measure
                swap = beat_diff - m_remaining
                beat_diff = m_remaining

            # get played strings and frets
            notes = [None] * 7
            notes_tied = [None] * 7
            for string, fret in zip(strings, frets):
                notes[string] = Note(fret)
                notes_tied[string] = Note(fret, tied=True)

            # build array of notes to write (e.g. a note length of 1.5 beats is written as quarter plus eighth)
            gp5_durations = self._split_beat_to_gp5_durations(beat_diff)
            while swap > 0:
                swap_beats = min(swap, self.beats_per_measure)
                swap -= swap_beats
                gp5_durations += self._split_beat_to_gp5_durations(swap_beats)

            # append actual notes. All except the first are written as tied notes
            tied = False
            for gp5_duration in gp5_durations:
                m_len += self._convert_duration_gp5_to_beats(gp5_duration)

                current_measure[0][0].append(Beat(notes_tied if tied else notes, duration=gp5_duration))
                tied = True
                if m_len >= self.beats_per_measure:
                    m_len = 0
                    beats.append(current_measure)
                    current_measure = self._create_measure()
        beats.append(current_measure)

        return beats

    @staticmethod
    def _convert_duration_beats_to_gp5(beats):
        return round(log2(beats)) * (-1)  # e.g. if beat duration is 0.25 beats -> log2(0.25) = -2 -> flip signs = 2

    @staticmethod
    def _convert_duration_gp5_to_beats(gp5_duration):
        return pow(2, gp5_duration*(-1))

    @staticmethod
    def _create_measure():
        return [([], [])]  # return new empty measure

    def _split_beat_to_gp5_durations(self, beats):
        gp5_durations = []
        cur_duration = 4.0
        # TODO epsilon
        while beats > 0 and cur_duration >= self.shortest_note:
            # TODO epsilon
            if cur_duration <= self.beats_per_measure and cur_duration <= beats:
                gp5_durations.append(self._convert_duration_beats_to_gp5(cur_duration))
                beats -= cur_duration
            cur_duration /= 2
        return gp5_durations
