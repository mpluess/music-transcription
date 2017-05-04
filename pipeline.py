import os

from fileformat.gp_utils import Header, Measure, Track
from fileformat.gp5_writer import write_gp5
from beat_conversion.simple_beat_conversion import SimpleBeatConverter
from onset_detection.cnn_onset_detection import CnnOnsetDetector
from pitch_detection.random_pitch_detection import RandomPitchDetector
from string_fret_detection.simple_string_fret_detection import SimpleStringFretDetection

# Files in train set
# path_to_wav_file = r'data\IDMT-SMT-GUITAR_V2\dataset2\audio\AR_Lick5_FN.wav'

# Other files
# path_to_wav_file = r'data\IDMT-SMT-GUITAR_V2\dataset2\audio\LP_Lick5_FN.wav'
path_to_wav_file = r'data\IDMT-SMT-GUITAR_V2\dataset3\audio\pathetique_mono.wav'

# Standard tuning:
# string / fret
# 0/0 = 64
# 1/0 = 59
# 2/0 = 55
# 3/0 = 50
# 4/0 = 45
# 5/0 = 40
tuning = (64, 59, 55, 50, 45, 40)
n_frets = 24

onset_detector = CnnOnsetDetector.from_zip('models/20170504-1-channel_ds2_adjusted-labels_10-epochs.zip')
onset_times_seconds = onset_detector.predict_onset_times_seconds(path_to_wav_file)

# print(onset_times_seconds)
# classes, probas = onset_detector.predict_classes_proba(path_to_wav_file)
# frame_rate_hz = onset_detector.feature_extractor.frame_rate_hz
# for i, label, proba in zip(range(len(classes)), classes, probas):
#     print('time={}, label={}, proba={}'.format(i / frame_rate_hz, label, proba))

pitch_detector = RandomPitchDetector(tuning, n_frets)
pitches = pitch_detector.predict_pitches_monophonic(path_to_wav_file, onset_times_seconds)

# for onset_time_seconds, pitch in zip(onset_times_seconds, pitches):
#     print('onset={}, pitch={}'.format(onset_time_seconds, pitch))

string_fret_detector = SimpleStringFretDetection(tuning, n_frets)
strings, frets = string_fret_detector.predict_strings_and_frets(path_to_wav_file, onset_times_seconds, pitches)

for onset, pitch, string, fret in zip(onset_times_seconds, pitches, strings, frets):
    print('onset={}, pitch={}, string={}, fret={}'.format(onset, pitch, string, fret))

beat_converter = SimpleBeatConverter()
beat_converter.fit(path_to_wav_file)
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
path_to_gp5_file = os.path.join('tmp', track_title + '.gp5')
write_gp5(
    measures, tracks, beats, tempo=beat_converter.tempo, outfile=path_to_gp5_file, header=Header(
        track_title, '', '', '', '', '', '', '', '', ''
    )
)
