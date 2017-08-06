import numpy as np
import sklearn.metrics
from sklearn.model_selection import train_test_split

from music_transcription.pitch_detection.read_data import read_data_y
from music_transcription.read_data import get_wav_and_truth_files


def predict_print_metrics(pitch_detector_, wav_file_paths_, list_of_onset_times_, y_, min_pitch_, max_pitch_):
    y_predicted_parts = []
    for path_to_wav, onset_times_seconds in zip(wav_file_paths_, list_of_onset_times_):
        y_predicted_parts.append(pitch_detector_.predict(path_to_wav, onset_times_seconds))
    y_predicted = np.concatenate(y_predicted_parts)
    assert y_.shape == y_predicted.shape

    # Print metrics
    print('Accuracy: {}'.format(round(sklearn.metrics.accuracy_score(y_, y_predicted), 4)))
    print(sklearn.metrics.classification_report(y_, y_predicted, digits=3,
                                                target_names=[str(pitch) for pitch in range(min_pitch_, max_pitch_ + 1)]))

# active_datasets = {1, 2, 3, 9, 10, 11}
# wav_file_paths, truth_dataset_format_tuples = get_wav_and_truth_files(active_datasets)
# (
#     wav_file_paths_train, wav_file_paths_test,
#     truth_dataset_format_tuples_train, truth_dataset_format_tuples_test
# ) = train_test_split(
#     wav_file_paths, truth_dataset_format_tuples, test_size=0.2, random_state=42
# )

# active_datasets = {5}
active_datasets = {1, 2}
wav_file_paths_test, truth_dataset_format_tuples_test = get_wav_and_truth_files(active_datasets)

# wav_file_paths_test = [r'..\data\recordings\audio\instrumental_lead.wav']
# truth_dataset_format_tuples_test = [(r'..\data\recordings\annotation\instrumental_lead.xml', 5, 'xml')]

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
# from music_transcription.pitch_detection.cnn_cqt_pitch_detection import CnnCqtPitchDetector
# pitch_detector = CnnCqtPitchDetector.from_zip('../models/pitch_detection/20170725_cqt_ds12391011_80-perc.zip')
# assert sample_rate == pitch_detector.feature_extractor.sample_rate
# assert subsampling_step == pitch_detector.config['subsampling_step']
# assert min_pitch == pitch_detector.config['min_pitch']
# assert max_pitch == pitch_detector.config['max_pitch']
# assert onset_group_threshold_seconds == pitch_detector.config['onset_group_threshold_seconds']

from music_transcription.pitch_detection.aubio_pitch_detection import AubioPitchDetector
pitch_detector = AubioPitchDetector()

# Predict
predict_print_metrics(pitch_detector, wav_file_paths_test, data_test[1], y_test, min_pitch, max_pitch)
