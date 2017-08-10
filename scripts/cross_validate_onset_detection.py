"""Do a 5-fold cross validation on a set of datasets using additional datasets for training.

The primary purpose is to have a fair comparison with the paper:
AUTOMATIC TABLATURE TRANSCRIPTION OF ELECTRIC GUITAR RECORDINGS BY ESTIMATION OF SCORE- AND INSTRUMENT-RELATED PARAMETERS
Kehling et al., 2014

This paper reports precision, recall and F-measure for the datasets 1 and 2.
The cross validation approach makes sure we don't predict any samples we've already seen while making sure
we can still train on a substantial percentage of the datasets 1 and 2.
"""

from sklearn.model_selection import KFold
from warnings import warn

from music_transcription.onset_detection.cnn_onset_detection import CnnOnsetDetector
from music_transcription.onset_detection.metrics import Metrics, onset_metric_times
from music_transcription.onset_detection.read_data import read_onset_times
from music_transcription.read_data import get_wav_and_truth_files


def predict_print_metrics(wav_file_paths, truth_dataset_format_tuples):
    metrics_2 = Metrics(0, 0, 0)
    metrics_5 = Metrics(0, 0, 0)
    for path_to_wav, (path_to_truth, dataset, truth_format) in zip(wav_file_paths, truth_dataset_format_tuples):
        onset_times = read_onset_times(path_to_truth, dataset, truth_format, 0.05)
        onset_times_predicted = onset_detector.predict_onsets(path_to_wav)
        if onset_times_predicted is None:
            warn('Failed to predict {}'.format(path_to_wav))
        else:
            metrics_2.add(onset_metric_times(onset_times, onset_times_predicted, n_tolerance_seconds_plus_minus=0.02))
            metrics_5.add(onset_metric_times(onset_times, onset_times_predicted, n_tolerance_seconds_plus_minus=0.05))

    print('n_tolerance_seconds_plus_minus=0.02')
    print_metrics(metrics_2)
    print('n_tolerance_seconds_plus_minus=0.05')
    print_metrics(metrics_5)


def print_metrics(metrics):
    print('TP=' + str(metrics.tp) + ', FN=' + str(metrics.fn) + ', FP=' + str(metrics.fp))
    print('precision=' + str(metrics.precision()) + ', recall=' + str(metrics.recall()) + ', F1=' + str(metrics.f1()))

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
    # onset_detector.save('../models/onset_detection/20170725-3-channels_ds12-cv_ds34-additional_fold-' + str(k) + '.zip')

    print('TRAIN')
    predict_print_metrics(wav_file_paths_train, truth_dataset_format_tuples_train)
    print('TEST')
    predict_print_metrics(wav_file_paths_test, truth_dataset_format_tuples_test)
