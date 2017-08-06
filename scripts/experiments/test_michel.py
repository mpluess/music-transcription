# import numpy as np
#
# n_frames_after_cutoff_per_file = [10, 15, 5]
# print(n_frames_after_cutoff_per_file)
# # [10 15 5] -> [5 10 15]
# start_index_per_file = np.roll(n_frames_after_cutoff_per_file, 1)
# print(start_index_per_file)
# # [5 10 15] -> [0 10 15]
# start_index_per_file[0] = 0
# print(start_index_per_file)
# # [0 10 15] -> [0 10 25]
# start_index_per_file = np.cumsum(start_index_per_file)
# print(start_index_per_file)


# print(float(0.0000001))


# from sklearn.preprocessing import MultiLabelBinarizer
#
# label_binarizer = MultiLabelBinarizer(classes=range(40, 89))
# label_binarizer.fit(None)
# print(label_binarizer.transform([{40, 42, 43}, {81, 83, 88}, {50}]))


# import numpy as np
# a = np.array([[0.1, 0.6], [0.54, 0.44]])
# print(a)
# print(a > 0.5)

# import os
# import pickle
# from zipfile import ZipFile
#
# config = {
#     'onset_group_threshold_seconds': 0.05
# }
# config_file = r'D:\Users\Michel\Documents\FH\module\8_IP6\git\models\onset_detection\config.pickle'
# with open(config_file, 'wb') as f:
#     pickle.dump(config, f)
#
# to_zip = [
#     config_file,
#     r'D:\Users\Michel\Documents\FH\module\8_IP6\git\models\onset_detection\feature_extractor.pickle',
#     r'D:\Users\Michel\Documents\FH\module\8_IP6\git\models\onset_detection\model.json',
#     r'D:\Users\Michel\Documents\FH\module\8_IP6\git\models\onset_detection\weights.hdf5',
# ]
# with ZipFile(r'D:\Users\Michel\Documents\FH\module\8_IP6\git\models\onset_detection\20170627-3-channels_ds1-4_80-perc_adjusted-labels_with_config_thresh-0.05.zip', 'w') as zip_file:
#     for path_to_file in to_zip:
#         zip_file.write(path_to_file, arcname=os.path.basename(path_to_file))

# a = 22.239999999999998
# b = 22.27
# c = 22.289999999999999
# epsilon=1e-6
# onset_group_threshold_seconds = 0.03
#
# print(b - a)
# print(c - b)
# print(onset_group_threshold_seconds - epsilon)
# print(onset_group_threshold_seconds + epsilon)

# print(5 / 3)
# print(round(5 / 3, 3))

# a = [1, 2, 4]
# b = [1, 2]
# assert a == b

# from music_transcription.pitch_detection.cnn_cqt_pitch_detection import CnnCqtPitchDetector
# print(CnnCqtPitchDetector._get_sample_weights(
#     [0, 1, 1, 2, 2, 2, 0, 0, 2],
#     [(0, 1, 0), (0, 2, 0), (0, 4, 0)]
# ))

# import re
# p = re.compile(r'\.gp5$')
# print(p.sub('', 'gp.gp5'))

# class_distribution = {
#     0: 64,
#     1: 28,
#     2: 20,
#     3: 85,
#     4: 45,
#     5: 232,
#     6: 69,
#     7: 174,
#     8: 389,
#     9: 157,
#     10: 349,
#     11: 109,
#     12: 404,
#     13: 188,
#     14: 331,
#     15: 339,
#     16: 263,
#     17: 372,
#     18: 140,
#     19: 450,
#     20: 251,
#     21: 363,
#     22: 359,
#     23: 104,
#     24: 373,
#     25: 120,
#     26: 141,
#     27: 210,
#     28: 66,
#     29: 249,
#     30: 56,
#     31: 65,
#     32: 71,
#     33: 45,
#     34: 79,
#     35: 21,
#     36: 21,
#     37: 66,
#     38: 79,
#     39: 67,
#     40: 43,
#     41: 43,
#     42: 43,
#     43: 49,
#     44: 26,
#     45: 23,
#     46: 23,
#     47: 23,
#     48: 26,
# }
# min_value = min(class_distribution.values())
# class_distribution_normalized = {k: v / min_value for k, v in class_distribution.items()}
# print('{')
# for k, v in sorted(class_distribution_normalized.items(), key=lambda t: t[0]):
#     print('    {}: {},'.format(k, v))
# print('}')

# with open(r'D:\Users\Michel\Dropbox\FH\module\8_IP6\doc\results\pitch_detection\20170711_notebook\dropout_proba_sweep.txt') as f:
#     for line in f:
#         line = line.rstrip()
#         if line.startswith('dropout_conv=') or line.startswith('Accuracy: '):
#             print(line)

# import os
# from sklearn.model_selection import KFold
#
# os.chdir('..')
#
# from music_transcription.pitch_detection.read_data import get_wav_and_truth_files
#
# DATASETS_CV = {1, 2}
# DATASETS_ADDITIONAL = {3, 9, 10, 11}
#
# wav_file_paths_cv, truth_dataset_format_tuples_cv = get_wav_and_truth_files(DATASETS_CV)
# wav_file_paths_additional, truth_dataset_format_tuples_additional = get_wav_and_truth_files(DATASETS_ADDITIONAL)
#
# k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
# for k, (train_indices, test_indices) in enumerate(k_fold.split(wav_file_paths_cv)):
#     print(k)
#     for path in sorted([wav_file_paths_cv[i] for i in test_indices]):
#         print(path)

# for f in [1/2, 1/4, 1/8, 1/16, 1/32, 1/64]:
#     print(f)

# print(float(4))


# print('')
# print('n_tolerance_seconds_plus_minus=0.02')
# for file, metrics in sorted(file_metrics_2_tuples, key=lambda t: t[1].f1(), reverse=True):
#     print(file)
#     print_metrics(metrics)


# proba_threshold
# for proba_threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
#     pitch_detector.config['proba_threshold'] = proba_threshold
#     print(proba_threshold)
#     predict_print_metrics(pitch_detector, wav_file_paths_test, data_test[1], y_test, min_pitch, max_pitch)

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

# from music_transcription.string_fret_detection.sequence_string_fret_detection import SequenceStringFretDetection
#
# string_fret_detector = SequenceStringFretDetection(tuning=(64, 59, 55, 50, 45, 40), n_frets=24)
# string_fret_detector.predict_strings_and_frets(None, [0.5, 1.0, 1.5], [{50, 60}, {45, 55}, {60, 70}])

from music_transcription.string_fret_detection.sequence_string_fret_detection import SequenceStringFretDetection

onset_times_seconds = [0.5, 1.0, 1.5]
list_of_pitch_sets = [{50, 60}, {45, 55}, {65, 75}]

string_fret_detector = SequenceStringFretDetection((64, 59, 55, 50, 45, 40), 24)
list_of_string_lists, list_of_fret_lists = string_fret_detector.predict_strings_and_frets(
    None, onset_times_seconds, list_of_pitch_sets
)
print(list_of_string_lists)
print(list_of_fret_lists)
