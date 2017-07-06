"""Do a 5-fold cross validation on a set of datasets using additional datasets for training.

The primary purpose is to have a fair comparison with the paper:
AUTOMATIC TABLATURE TRANSCRIPTION OF ELECTRIC GUITAR RECORDINGS BY ESTIMATION OF SCORE- AND INSTRUMENT-RELATED PARAMETERS
Kehling et al., 2014

This paper reports precision, recall and F-measure for the datasets 1 and 2.
The cross validation approach makes sure we don't predict any samples we've already seen while making sure
we can still train on a substantial percentage of the datasets 1 and 2.
"""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import KFold

from music_transcription.pitch_detection.cnn_pitch_detection import CnnPitchDetector
from music_transcription.pitch_detection.cnn_cqt_pitch_detection import CnnCqtPitchDetector
from music_transcription.pitch_detection.read_data import get_wav_and_truth_files, read_data_y


def predict(pitch_detector, wav_file_paths, truth_dataset_format_tuples,
            sample_rate, subsampling_step, min_pitch, max_pitch, onset_group_threshold_seconds):
    print('Predicting labels')
    data, y, wav_file_paths, truth_dataset_format_tuples = read_data_y(
        wav_file_paths, truth_dataset_format_tuples,
        sample_rate, subsampling_step, min_pitch, max_pitch,
        onset_group_threshold_seconds=onset_group_threshold_seconds
    )
    y_predicted_parts = []
    for path_to_wav, onset_times_seconds in zip(wav_file_paths, data[1]):
        y_predicted_parts.append(pitch_detector.predict(path_to_wav, onset_times_seconds))
    y_predicted = np.concatenate(y_predicted_parts)

    print('Accuracy: {}'.format(accuracy_score(y, y_predicted)))
    print(classification_report(y, y_predicted,
                                target_names=[str(pitch) for pitch in range(min_pitch, max_pitch + 1)]))

DATASETS_CV = {1, 2}
# DATASETS_ADDITIONAL = {3}
# DATASETS_ADDITIONAL = {3, 6}
# DATASETS_ADDITIONAL = {3, 6, 7}
DATASETS_ADDITIONAL = {3, 9, 10, 11}

sample_rate = 44100
subsampling_step = 1
min_pitch = 40
max_pitch = 88
onset_group_threshold_seconds = 0.05

wav_file_paths_cv, truth_dataset_format_tuples_cv = get_wav_and_truth_files(DATASETS_CV)
wav_file_paths_additional, truth_dataset_format_tuples_additional = get_wav_and_truth_files(DATASETS_ADDITIONAL)

k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
for k, (train_indices, test_indices) in enumerate(k_fold.split(wav_file_paths_cv)):
    if k > 0:
        print('Skipping split {}'.format(k))
        continue

    wav_file_paths_train = [wav_file_paths_cv[i] for i in train_indices] + wav_file_paths_additional
    truth_dataset_format_tuples_train = [truth_dataset_format_tuples_cv[i] for i in train_indices] + truth_dataset_format_tuples_additional
    wav_file_paths_test = [wav_file_paths_cv[i] for i in test_indices]
    truth_dataset_format_tuples_test = [truth_dataset_format_tuples_cv[i] for i in test_indices]

    # pitch_detector = CnnPitchDetector()
    pitch_detector = CnnCqtPitchDetector()
    print('Fitting pitch detector')
    pitch_detector.fit(wav_file_paths_train, truth_dataset_format_tuples_train,
                       wav_file_paths_test, truth_dataset_format_tuples_test)

    # print('TRAIN')
    # predict(pitch_detector, wav_file_paths_train, truth_dataset_format_tuples_train,
    #         frame_rate_hz, sample_rate, subsampling_step, min_pitch, max_pitch)

    print('TEST')
    predict(pitch_detector, wav_file_paths_test, truth_dataset_format_tuples_test,
            sample_rate, subsampling_step, min_pitch, max_pitch, onset_group_threshold_seconds)

    pitch_detector.save('../models/pitch_detection/20170706_1644_cqt_ds12-cv_ds391011-additional_onset-group-thresh-0.05_20-filters_fold-' + str(k) + '.zip')
