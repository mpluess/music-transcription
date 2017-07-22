"""Do a 5-fold cross validation on a set of datasets using additional datasets for training.

The primary purpose is to have a fair comparison with the paper:
AUTOMATIC TABLATURE TRANSCRIPTION OF ELECTRIC GUITAR RECORDINGS BY ESTIMATION OF SCORE- AND INSTRUMENT-RELATED PARAMETERS
Kehling et al., 2014

This paper reports precision, recall and F-measure for the datasets 1 and 2.
The cross validation approach makes sure we don't predict any samples we've already seen while making sure
we can still train on a substantial percentage of the datasets 1 and 2.
"""

from sklearn.model_selection import KFold

from music_transcription.onset_detection.cnn_onset_detection import CnnOnsetDetector
from music_transcription.read_data import get_wav_and_truth_files

DATASETS_CV = {1, 2}
DATASETS_ADDITIONAL = {3, 4}

wav_file_paths_cv, truth_dataset_format_tuples_cv = get_wav_and_truth_files(DATASETS_CV)
wav_file_paths_additional, truth_dataset_format_tuples_additional = get_wav_and_truth_files(DATASETS_ADDITIONAL)

k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
for k, (train_indices, test_indices) in enumerate(k_fold.split(wav_file_paths_cv)):
    wav_file_paths_train = [wav_file_paths_cv[i] for i in train_indices] + wav_file_paths_additional
    truth_dataset_format_tuples_train = [truth_dataset_format_tuples_cv[i] for i in train_indices] + truth_dataset_format_tuples_additional
    wav_file_paths_test = [wav_file_paths_cv[i] for i in test_indices]
    truth_dataset_format_tuples_test = [truth_dataset_format_tuples_cv[i] for i in test_indices]

    onset_detector = CnnOnsetDetector()
    onset_detector.fit(wav_file_paths_train, truth_dataset_format_tuples_train)

    print('TRAIN')
    onset_detector.predict_print_metrics(wav_file_paths_train, truth_dataset_format_tuples_train)
    print('TEST')
    onset_detector.predict_print_metrics(wav_file_paths_test, truth_dataset_format_tuples_test)

    onset_detector.save('../models/onset_detection/20170511-1-channel_ds12-cv_ds34-additional_fold-' + str(k) + '_adjusted-labels.zip')
