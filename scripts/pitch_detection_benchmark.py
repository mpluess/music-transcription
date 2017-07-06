import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split

from music_transcription.pitch_detection.read_data import get_wav_and_truth_files, read_data_y

active_datasets = {1, 2, 3}
# active_datasets = {1, 2}
# active_datasets = {6}
# active_datasets = {1, 2, 3, 6}
# active_datasets = {7}
wav_file_paths, truth_dataset_format_tuples = get_wav_and_truth_files(active_datasets)
(
    wav_file_paths_train, wav_file_paths_test,
    truth_dataset_format_tuples_train, truth_dataset_format_tuples_test
) = train_test_split(
    wav_file_paths, truth_dataset_format_tuples, test_size=0.2, random_state=42
)

# active_datasets = {5}
# wav_file_paths_test, truth_dataset_format_tuples_test = get_wav_and_truth_files(active_datasets)

# wav_file_paths_test = [r'..\data\IDMT-SMT-GUITAR_V2\dataset3\audio\nocturneNr2.wav']
# truth_dataset_format_tuples_test = [(r'..\data\IDMT-SMT-GUITAR_V2\dataset3\annotation\nocturneNr2.xml', 3, 'xml')]

sample_rate = 44100
subsampling_step = 1
min_pitch = 40
max_pitch = 88
onset_group_threshold_seconds = 0.05

data_test, y_test, wav_file_paths_test, truth_dataset_format_tuples_test = read_data_y(
    wav_file_paths_test, truth_dataset_format_tuples_test,
    sample_rate, subsampling_step, min_pitch, max_pitch,
    onset_group_threshold_seconds=onset_group_threshold_seconds
)
assert len(wav_file_paths_test) == len(data_test[1])

# Load your pitch detector here
# from music_transcription.pitch_detection.cnn_pitch_detection import CnnPitchDetector
# pitch_detector = CnnPitchDetector.from_zip('../models/pitch_detection/20170627_1500_ds1-3_80-perc_onset-group-thresh-0.05.zip')
# assert sample_rate == pitch_detector.feature_extractor.sample_rate
# assert subsampling_step == pitch_detector.feature_extractor.subsampling_step
# assert min_pitch == pitch_detector.config['min_pitch']
# assert max_pitch == pitch_detector.config['max_pitch']
# assert onset_group_threshold_seconds == pitch_detector.config['onset_group_threshold_seconds']

from music_transcription.pitch_detection.cnn_cqt_pitch_detection import CnnCqtPitchDetector
pitch_detector = CnnCqtPitchDetector.from_zip('../models/pitch_detection/20170706_1033_cqt_generated_mono_100-perc.zip')
assert sample_rate == pitch_detector.feature_extractor.sample_rate
assert subsampling_step == pitch_detector.config['subsampling_step']
assert min_pitch == pitch_detector.config['min_pitch']
assert max_pitch == pitch_detector.config['max_pitch']
assert onset_group_threshold_seconds == pitch_detector.config['onset_group_threshold_seconds']

# from music_transcription.pitch_detection.aubio_pitch_detection import AubioPitchDetector
# pitch_detector = AubioPitchDetector()

# Predict
y_test_predicted_parts = []
for path_to_wav, onset_times_seconds in zip(wav_file_paths_test, data_test[1]):
    y_test_predicted_parts.append(pitch_detector.predict(path_to_wav, onset_times_seconds))
y_test_predicted = np.concatenate(y_test_predicted_parts)
assert y_test.shape == y_test_predicted.shape

# Print metrics
print('Accuracy: {}'.format(sklearn.metrics.accuracy_score(y_test, y_test_predicted)))
print(sklearn.metrics.classification_report(y_test, y_test_predicted,
                                            target_names=[str(pitch) for pitch in range(min_pitch, max_pitch + 1)]))


# Verbose / confusion
# y_test_offset = 0
# for path_to_wav, onset_times_seconds in zip(wav_file_paths_test, data_test[1]):
#     y_test_part = y_test[y_test_offset:y_test_offset + len(onset_times_seconds), :]
#     y_test_predicted_part = pitch_detector.predict(path_to_wav, onset_times_seconds)
#     assert y_test_part.shape == y_test_predicted_part.shape
#
#     print(path_to_wav)
#     print('Accuracy: {}'.format(sklearn.metrics.accuracy_score(y_test_part, y_test_predicted_part)))
#     print(sklearn.metrics.classification_report(y_test_part, y_test_predicted_part,
#                                                 target_names=[str(pitch) for pitch in range(min_pitch, max_pitch + 1)]))
#
#     for y_test_row, y_test_predicted_row in zip(y_test_part, y_test_predicted_part):
#         for pitch, label, label_predicted in zip(range(min_pitch, max_pitch + 1), y_test_row, y_test_predicted_row):
#             if label != 0 or label_predicted != 0:
#                 print('{} {} {}'.format(pitch, label, label_predicted))
#         print('')
#
#     y_test_offset += len(onset_times_seconds)

# for y_test_row, y_test_predicted_row in zip(y_test, y_test_predicted):
#     for pitch, label, label_predicted in zip(range(min_pitch, max_pitch + 1), y_test_row, y_test_predicted_row):
#         if label != 0 or label_predicted != 0:
#             print('{} {} {}'.format(pitch, label, label_predicted))
#     print('')
