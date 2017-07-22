"""Run the current guitar recording transcription pipeline.

Input: guitar recording (wave file)
Output: transcribed notes and tabs (gp5 file)

Parts:
- onset detection
- pitch detection
- string and fret detection
- tempo detection
- beat transformation (mapping of onsets and strings/frets to discrete notes in measures)
- gp5 export
"""

import argparse
import os

from music_transcription.beat_transformation.simple_beat_transformation import SimpleBeatTransformer
from music_transcription.fileformat.guitar_pro.utils import Header, Measure, Track
from music_transcription.fileformat.guitar_pro.gp5_writer import write_gp5
from music_transcription.onset_detection.cnn_onset_detection import CnnOnsetDetector
from music_transcription.pitch_detection.cnn_cqt_pitch_detection import CnnCqtPitchDetector
from music_transcription.pitch_detection.aubio_pitch_detection import AubioPitchDetector
from music_transcription.string_fret_detection.simple_string_fret_detection import SimpleStringFretDetection
from music_transcription.tempo_detection.aubio_tempo_detection import AubioTempoDetector

# CONFIG
# Tuning and number of frets are currently not really configurable since we only have
# models for the standard tuning with 24 frets.
# Standard tuning:
# string/fret
# 0/0 = 64
# 1/0 = 59
# 2/0 = 55
# 3/0 = 50
# 4/0 = 45
# 5/0 = 40
TUNING = (64, 59, 55, 50, 45, 40)
N_FRETS = 24

TEMPO_DEFAULT = -1

SHORTEST_NOTES = {
    '1/1': 4.0, '1/2': 2.0, '1/4': 1.0, '1/8': 0.5, '1/16': 0.25, '1/32': 0.125, '1/64': 0.0625
}

parser = argparse.ArgumentParser(description='Transcribe guitar recording (WAV) to notes and tabs (GP5)')
parser.add_argument('path_to_wav', help='Path to guitar recording in WAV format')
parser.add_argument('--model_dir', default=os.path.join('..', 'models'), help='Path to models directory')
parser.add_argument('--musical_texture', default='poly', choices=['mono', 'poly'],
                    help='Is your recording strictly monophonic or does it also contain polyphonic chords?')
parser.add_argument('--tempo', type=int, default=TEMPO_DEFAULT,
                    help="Tempo of the recording in BPM. We'll try to determine this automatically if not set.")
parser.add_argument('--beats_per_measure', type=int, default=4,
                    help='Time signature / number of quarter notes per measure')
parser.add_argument('--shortest_note', default='1/16', choices=SHORTEST_NOTES.keys(),
                    help='Shortest possible note')
parser.add_argument('--instrument_id', type=int, default=27, help='Instrument id for GP5 file')
parser.add_argument('--track_title', help='Track title for GP5 file')
parser.add_argument('--path_to_gp5', help='Output path of GP5 file')
parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output')

args = parser.parse_args()
assert os.path.isfile(args.path_to_wav), 'Recording file not found'
assert args.tempo == -1 or args.tempo > 0, 'Tempo is invalid, should be -1 or > 0'
assert args.beats_per_measure > 0, 'Beats per measure is invalid, should be > 0'
assert args.instrument_id > 0, 'Instrument ID is invalid, should be > 0'

# PIPELINE
onset_detector = CnnOnsetDetector.from_zip(
    os.path.join(args.model_dir,
                 'onset_detection',
                 '20170627-3-channels_ds1-4_80-perc_adjusted-labels_with_config_thresh-0.05.zip')
)
onset_times_seconds = onset_detector.predict_onsets(args.path_to_wav)

if args.musical_texture == 'mono':
    pitch_detector = AubioPitchDetector()
else:
    pitch_detector = CnnCqtPitchDetector.from_zip(
        os.path.join(args.model_dir,
                     'pitch_detection',
                     '20170718_1224_cqt_ds12391011_100-perc_optimized-params_proba-thresh-0.3.zip')
    )
list_of_pitch_sets = pitch_detector.predict_pitches(args.path_to_wav, onset_times_seconds)

string_fret_detector = SimpleStringFretDetection(TUNING, N_FRETS)
list_of_string_lists, list_of_fret_lists = string_fret_detector.predict_strings_and_frets(args.path_to_wav,
                                                                                          onset_times_seconds,
                                                                                          list_of_pitch_sets)

if args.verbose:
    for onset, pitch, string, fret in zip(onset_times_seconds, list_of_pitch_sets,
                                          list_of_string_lists, list_of_fret_lists):
        print('onset={}, pitch={}, string={}, fret={}'.format(onset, sorted(pitch, reverse=True), string, fret))

if args.tempo == TEMPO_DEFAULT:
    tempo_detector = AubioTempoDetector()
    tempo = tempo_detector.predict(args.path_to_wav, onset_times_seconds)
else:
    tempo = args.tempo
beat_transformer = SimpleBeatTransformer(shortest_note=SHORTEST_NOTES[args.shortest_note],
                                         beats_per_measure=float(args.beats_per_measure))
beats = beat_transformer.transform(args.path_to_wav, onset_times_seconds,
                                   list_of_string_lists, list_of_fret_lists, tempo)

measures = []
for i, measure in enumerate(beats):
    if i == 0:
        measures.append(Measure(args.beats_per_measure, 4, False, 0, 0, "", (0, 0, 0, 0), 0, 0, False, (2, 2, 2, 2), 0))
    else:
        measures.append(Measure(0, 0, False, 0, 0, "", (0, 0, 0, 0), 0, 0, False, (0, 0, 0, 0), 0))

tracks = [
    Track(
        "Electric Guitar",
        len(TUNING), TUNING + (-1,),
        1, 1, 2, N_FRETS, 0, (200, 55, 55, 0), args.instrument_id
    ),
]

recording_name = os.path.basename(args.path_to_wav).rstrip('.wav')
if args.track_title is None:
    track_title = recording_name
else:
    track_title = args.track_title

if args.path_to_gp5 is None:
    path_to_gp5 = recording_name + '.gp5'
else:
    path_to_gp5 = args.path_to_gp5

write_gp5(
    measures, tracks, beats, tempo=tempo, outfile=path_to_gp5, header=Header(
        track_title, '', '', '', '', '', '', '', '', ''
    )
)
