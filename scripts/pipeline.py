"""Run the current music transcription pipeline.

Input: guitar recording (wave file)
Output: transcribed notes and tabs (gp5 file)

Parts:
- onset detection
- pitch detection
- string and fret detection
- mapping of onsets and pitches to measures and notes / chords
- gp5 export
"""

import os

from music_transcription.beat_conversion.simple_beat_conversion import SimpleBeatConverter
from music_transcription.fileformat.guitar_pro.utils import Header, Measure, Track
from music_transcription.fileformat.guitar_pro.gp5_writer import write_gp5
from music_transcription.onset_detection.cnn_onset_detection import CnnOnsetDetector
from music_transcription.pitch_detection.aubio_pitch_detection import AubioPitchDetector
from music_transcription.string_fret_detection.simple_string_fret_detection import SimpleStringFretDetection

# CONFIG
DATA_DIR = r'..\data'

path_to_wav_file = os.path.join(DATA_DIR, r'IDMT-SMT-GUITAR_V2\dataset3\audio\pathetique_mono.wav')
# path_to_wav_file = os.path.join(DATA_DIR, r'recordings\mim-riff1-short-slow.wav')

# Standard tuning:
# string / fret
# 0/0 = 64
# 1/0 = 59
# 2/0 = 55
# 3/0 = 50
# 4/0 = 45
# 5/0 = 40
tuning = (64, 59, 55, 50, 45, 40)
# tuning = (63, 58, 54, 49, 44, 39)
n_frets = 24

# PIPELINE
onset_detector = CnnOnsetDetector.from_zip('../models/onset_detection/20170511-3-channels_ds1-4_80-perc_adjusted-labels.zip')
onset_times_seconds = onset_detector.predict_onset_times_seconds(path_to_wav_file)

pitch_detector = AubioPitchDetector(tuning, n_frets)
pitches = pitch_detector.predict(path_to_wav_file, onset_times_seconds)

string_fret_detector = SimpleStringFretDetection(tuning, n_frets)
strings, frets = string_fret_detector.predict_strings_and_frets(path_to_wav_file, onset_times_seconds, pitches)

for onset, pitch, string, fret in zip(onset_times_seconds, pitches, strings, frets):
    print('onset={}, pitch={}, string={}, fret={}'.format(onset, pitch, string, fret))

beat_converter = SimpleBeatConverter()
# beat_converter.set_tempo(49)
beats = beat_converter.transform(path_to_wav_file, onset_times_seconds, pitches, strings, frets)

measures = []
for i, measure in enumerate(beats):
    if i == 0:
        measures.append(Measure(4, 4, False, 0, 0, "", (0, 0, 0, 0), 0, 0, False, (2, 2, 2, 2), 0))
    else:
        measures.append(Measure(0, 0, False, 0, 0, "", (0, 0, 0, 0), 0, 0, False, (0, 0, 0, 0), 0))

tracks = [
    Track(
        "Electric Guitar",
        6, tuning + (-1,),
        1, 1, 2, n_frets, 0, (200, 55, 55, 0), 30
    ),
]

track_title = os.path.basename(path_to_wav_file).rstrip('.wav')
path_to_gp5_file = os.path.join(r'..\tmp', track_title + '.gp5')
write_gp5(
    measures, tracks, beats, tempo=beat_converter.tempo, outfile=path_to_gp5_file, header=Header(
        track_title, '', '', '', '', '', '', '', '', ''
    )
)


# onset_detector.predict_print_metrics([path_to_wav_file], [(path_to_wav_file.replace(".wav", ".xml"), '', 'xml')])