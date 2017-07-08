"""Open output file in TuxGuitar - Export - Export Audio File

File Encoding: PCM_SIGNED
Custom soundbank: $INSTALL_DIR\TuxGuitar\share\soundfont\MagicSFver2.sf2
"""

import os
import re

from music_transcription.fileformat.guitar_pro.gp5_writer import write_gp5
from music_transcription.fileformat.guitar_pro.utils import beat, Measure, note, Track
from music_transcription.fileformat.truth import write_truth_file
from music_transcription.string_fret_detection import plausibility

TUNING = (64, 59, 55, 50, 45, 40)
N_FRETS = 24
MIN_PITCH = min(TUNING)
MAX_PITCH = max(TUNING) + N_FRETS

START_OFFSET_SECONDS = 0.03

GP5_ENDING_REGEX = re.compile(r'\.gp5$')


def get_string_fret_possibilities(pitch):
    for string, string_pitch in enumerate(TUNING):
        if pitch >= string_pitch and pitch <= string_pitch + N_FRETS:
            yield string, pitch - string_pitch


def generate_mono(filename='generated_mono.gp5', tempo=120, pitches=None):
    if pitches is None:
        pitches = set(range(MIN_PITCH, MAX_PITCH + 1))

    tracks = [
        Track(
            "Electric Guitar",
            len(TUNING), TUNING + (-1,),
            1, 1, 2, N_FRETS, 0, (200, 55, 55, 0), 27
        ),
    ]
    measures = []
    beats = []
    onset_times = []
    list_of_pitches = []
    onset_time = START_OFFSET_SECONDS
    quarter_note_seconds = 60 / tempo
    for pitch in range(MIN_PITCH, MAX_PITCH + 1):
        if pitch in pitches:
            for string, fret in get_string_fret_possibilities(pitch):
                print('pitch={}, string={}, fret={}'.format(pitch, string, fret))

                if len(measures) == 0:
                    measures.append(Measure(4, 4, False, 0, 0, "", (0, 0, 0, 0), 0, 0, False, (2, 2, 2, 2), 0))
                else:
                    measures.append(Measure(0, 0, False, 0, 0, "", (0, 0, 0, 0), 0, 0, False, (0, 0, 0, 0), 0))

                beats_measure = create_measure()
                beats_measure[0][0].append(beat([None] * 7, pause=True))
                onset_time += quarter_note_seconds

                notes = [None] * 7
                notes[string] = note(fret)
                beats_measure[0][0].append(beat(notes))
                onset_times.append(onset_time)
                list_of_pitches.append([pitch])
                onset_time += quarter_note_seconds

                beats_measure[0][0].append(beat([None] * 7, pause=True))
                onset_time += quarter_note_seconds
                beats_measure[0][0].append(beat([None] * 7, pause=True))
                onset_time += quarter_note_seconds
                beats.append(beats_measure)

    path_to_gp5_file = os.path.join(r'..\tmp', filename)
    write_gp5(measures, tracks, beats, tempo=tempo, outfile=path_to_gp5_file)
    path_to_truth_file = os.path.join(r'..\tmp', GP5_ENDING_REGEX.sub('', filename) + '.xml')
    write_truth_file([path_to_truth_file], [onset_times], [list_of_pitches])


def generate_poly(chord_type, filename='generated_poly.gp5', n_measures_per_file=10000, tempo=120, pitches=None):
    if pitches is None:
        pitches = set(range(MIN_PITCH, MAX_PITCH + 1))

    files = []
    tracks, measures, beats, onset_times, list_of_pitches, onset_time = init_file()

    # chords with 2 pitches
    if chord_type == 2:
        for pitch_1 in range(MIN_PITCH, MAX_PITCH + 1):
            if pitch_1 in pitches:
                for pitch_2 in range(pitch_1 + 1, MAX_PITCH + 1):
                    if pitch_1 != pitch_2:
                        possibilities = plausibility.get_all_fret_possibilities((pitch_1, pitch_2), tuning=TUNING, n_frets=N_FRETS)
                        if len(possibilities) > 0:
                            measure, beats_measure, onset_after_seconds, measure_duration_seconds, printable_notes = create_measure_beats_measure_printable_notes(
                                len(measures), possibilities[0], tempo
                            )
                            measures.append(measure)
                            beats.append(beats_measure)
                            onset_times.append(onset_time + onset_after_seconds)
                            onset_time += measure_duration_seconds
                            list_of_pitches.append([pitch_1, pitch_2])
                            print('pitch_1={}, pitch_2={}, notes={}'.format(
                                pitch_1, pitch_2, printable_notes
                            ))

                            if len(measures) >= n_measures_per_file:
                                if len(files) == 0:
                                    current_filename = filename
                                else:
                                    current_filename = GP5_ENDING_REGEX.sub('', filename) + '.' + str(len(files)) + '.gp5'
                                files.append((current_filename, measures, tracks, beats, onset_times, list_of_pitches))
                                tracks, measures, beats, onset_times, list_of_pitches, onset_time = init_file()

    # chords with 3 pitches
    elif chord_type == 3:
        for pitch_1 in range(MIN_PITCH, MAX_PITCH + 1):
            if pitch_1 in pitches:
                for pitch_2 in range(pitch_1 + 1, MAX_PITCH + 1):
                    for pitch_3 in range(pitch_2 + 1, MAX_PITCH + 1):
                        if pitch_1 != pitch_2 and pitch_1 != pitch_3 and pitch_2 != pitch_3:
                            possibilities = plausibility.get_all_fret_possibilities((pitch_1, pitch_2, pitch_3),
                                                                                    tuning=TUNING, n_frets=N_FRETS)
                            if len(possibilities) > 0:
                                measure, beats_measure, onset_after_seconds, measure_duration_seconds, printable_notes = create_measure_beats_measure_printable_notes(
                                    len(measures), possibilities[0], tempo
                                )
                                measures.append(measure)
                                beats.append(beats_measure)
                                onset_times.append(onset_time + onset_after_seconds)
                                onset_time += measure_duration_seconds
                                list_of_pitches.append([pitch_1, pitch_2, pitch_3])
                                print('pitch_1={}, pitch_2={}, pitch_3={}, notes={}'.format(
                                    pitch_1, pitch_2, pitch_3, printable_notes
                                ))

                                if len(measures) >= n_measures_per_file:
                                    if len(files) == 0:
                                        current_filename = filename
                                    else:
                                        current_filename = GP5_ENDING_REGEX.sub('', filename) + '.' + str(len(files)) + '.gp5'
                                    files.append((current_filename, measures, tracks, beats, onset_times, list_of_pitches))
                                    tracks, measures, beats, onset_times, list_of_pitches, onset_time = init_file()

    # chords with 4 pitches
    elif chord_type == 4:
        for pitch_1 in range(MIN_PITCH, MAX_PITCH + 1):
            if pitch_1 in pitches:
                for pitch_2 in range(pitch_1 + 1, MAX_PITCH + 1):
                    for pitch_3 in range(pitch_2 + 1, MAX_PITCH + 1):
                        for pitch_4 in range(pitch_3 + 1, MAX_PITCH + 1):
                            if (pitch_1 != pitch_2 and pitch_1 != pitch_3 and pitch_1 != pitch_4
                                and pitch_2 != pitch_3 and pitch_2 != pitch_4
                                and pitch_3 != pitch_4):
                                possibilities = plausibility.get_all_fret_possibilities((pitch_1, pitch_2, pitch_3, pitch_4),
                                                                                        tuning=TUNING, n_frets=N_FRETS)
                                if len(possibilities) > 0:
                                    measure, beats_measure, onset_after_seconds, measure_duration_seconds, printable_notes = create_measure_beats_measure_printable_notes(
                                        len(measures), possibilities[0], tempo
                                    )
                                    measures.append(measure)
                                    beats.append(beats_measure)
                                    onset_times.append(onset_time + onset_after_seconds)
                                    onset_time += measure_duration_seconds
                                    list_of_pitches.append([pitch_1, pitch_2, pitch_3, pitch_4])
                                    print('pitch_1={}, pitch_2={}, pitch_3={}, pitch_4={}, notes={}'.format(
                                        pitch_1, pitch_2, pitch_3, pitch_4, printable_notes
                                    ))

                                    if len(measures) >= n_measures_per_file:
                                        if len(files) == 0:
                                            current_filename = filename
                                        else:
                                            current_filename = GP5_ENDING_REGEX.sub('', filename) + '.' + str(len(files)) + '.gp5'
                                        files.append((current_filename, measures, tracks, beats, onset_times, list_of_pitches))
                                        tracks, measures, beats, onset_times, list_of_pitches, onset_time = init_file()

    else:
        raise ValueError('Unsupported chord type {}'.format(chord_type))

    if len(measures) > 0:
        if len(files) == 0:
            current_filename = filename
        else:
            current_filename = GP5_ENDING_REGEX.sub('', filename) + '.' + str(len(files)) + '.gp5'
        files.append((current_filename, measures, tracks, beats, onset_times, list_of_pitches))

    for filename, measures, tracks, beats, onset_times, list_of_pitches in files:
        path_to_gp5_file = os.path.join(r'..\tmp', filename)
        write_gp5(measures, tracks, beats, tempo=tempo, outfile=path_to_gp5_file)
        path_to_truth_file = os.path.join(r'..\tmp', GP5_ENDING_REGEX.sub('', filename) + '.xml')
        write_truth_file([path_to_truth_file], [onset_times], [list_of_pitches])


def init_file():
    return [
        Track(
            "Electric Guitar",
            len(TUNING), TUNING + (-1,),
            1, 1, 2, N_FRETS, 0, (200, 55, 55, 0), 27
        ),
    ], [], [], [], [], START_OFFSET_SECONDS


def create_measure_beats_measure_printable_notes(len_measures, notes, tempo):
    quarter_note_seconds = 60 / tempo

    if len_measures == 0:
        measure = Measure(4, 4, False, 0, 0, "", (0, 0, 0, 0), 0, 0, False, (2, 2, 2, 2), 0)
    else:
        measure = Measure(0, 0, False, 0, 0, "", (0, 0, 0, 0), 0, 0, False, (0, 0, 0, 0), 0)

    beats_measure = create_measure()
    beats_measure[0][0].append(beat([None] * 7, pause=True))

    # [0, 3, 2, -1, -1, -1] -> [note(0), note(3), note(2), None, None, None, None]
    notes = [None if fret == -1 else note(fret) for fret in notes]
    notes.append(None)

    beats_measure[0][0].append(beat(notes))

    beats_measure[0][0].append(beat([None] * 7, pause=True))
    beats_measure[0][0].append(beat([None] * 7, pause=True))

    onset_after_seconds = quarter_note_seconds
    measure_duration_seconds = 4*quarter_note_seconds

    return (measure, beats_measure, onset_after_seconds, measure_duration_seconds,
            [None if my_note is None else my_note.fret for my_note in notes])


def create_measure():
    return [([], [])]

pitches = set(range(77, 89))
generate_mono(filename='generated_mono_filtered.gp5', pitches=pitches)
generate_poly(2, filename='generated_poly_2_filtered.gp5', pitches=pitches)
generate_poly(3, filename='generated_poly_3_filtered.gp5', pitches=pitches)
# generate_poly(4, filename='generated_poly_4.gp5')
# generate_poly(5, filename='generated_poly_5.gp5')
# generate_poly(6, filename='generated_poly_6.gp5')
