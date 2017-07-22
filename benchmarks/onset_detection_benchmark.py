from sklearn.model_selection import train_test_split
from warnings import warn

from music_transcription.onset_detection.cnn_onset_detection import CnnOnsetDetector
from music_transcription.onset_detection.metrics import Metrics, onset_metric_times
from music_transcription.onset_detection.read_data import read_onset_times
from music_transcription.read_data import get_wav_and_truth_files


def print_metrics(metrics):
    print('TP=' + str(metrics.tp) + ', FN=' + str(metrics.fn) + ', FP=' + str(metrics.fp))
    print('precision=' + str(metrics.precision()) + ', recall=' + str(metrics.recall()) + ', F1=' + str(metrics.f1()))

onset_group_threshold_seconds = 0.05

# active_datasets = {1, 2, 3, 4}
# wav_file_paths, truth_dataset_format_tuples = get_wav_and_truth_files(active_datasets)
# wav_file_paths_train, wav_file_paths_test, truth_dataset_format_tuples_train, truth_dataset_format_tuples_test = train_test_split(
#     wav_file_paths, truth_dataset_format_tuples, test_size=0.2, random_state=42
# )

active_datasets = {5}
wav_file_paths_test, truth_dataset_format_tuples_test = get_wav_and_truth_files(active_datasets)

# wav_file_paths_test = [r'..\data\recordings\audio\instrumental_lead.wav']
# truth_dataset_format_tuples_test = [(r'..\data\recordings\annotation\instrumental_lead.xml', 5, 'xml')]

onset_detector = CnnOnsetDetector.from_zip('../models/onset_detection/20170627-3-channels_ds1-4_80-perc_adjusted-labels_with_config_thresh-0.05.zip')
assert onset_group_threshold_seconds == onset_detector.config['onset_group_threshold_seconds']

metrics_2_aggr = Metrics(0, 0, 0)
metrics_5_aggr = Metrics(0, 0, 0)
file_metrics_2_tuples = []
file_metrics_5_tuples = []
for path_to_wav, (path_to_truth, dataset, truth_format) in zip(wav_file_paths_test, truth_dataset_format_tuples_test):
    onset_times = read_onset_times(path_to_truth, dataset, truth_format, onset_group_threshold_seconds)
    onset_times_predicted = onset_detector.predict_onsets(path_to_wav)
    if onset_times_predicted is None:
        warn('Failed to predict {}'.format(path_to_wav))
    else:
        metrics_2 = onset_metric_times(onset_times, onset_times_predicted, n_tolerance_seconds_plus_minus=0.02)
        metrics_2_aggr.add(metrics_2)
        file_metrics_2_tuples.append((path_to_wav, metrics_2))

        metrics_5 = onset_metric_times(onset_times, onset_times_predicted, n_tolerance_seconds_plus_minus=0.05)
        metrics_5_aggr.add(metrics_5)
        file_metrics_5_tuples.append((path_to_wav, metrics_5))

print('n_tolerance_seconds_plus_minus=0.02')
print_metrics(metrics_2_aggr)
print('n_tolerance_seconds_plus_minus=0.05')
print_metrics(metrics_5_aggr)
