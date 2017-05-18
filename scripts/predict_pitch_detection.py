import sklearn.metrics
from sklearn.model_selection import train_test_split

from music_transcription.pitch_detection.cnn_pitch_detection import CnnPitchDetector
from music_transcription.pitch_detection.read_data import get_wav_and_truth_files, read_X_y

active_datasets = {1, 2, 3}

wav_file_paths, truth_dataset_format_tuples = get_wav_and_truth_files(active_datasets)

(
    wav_file_paths_train, wav_file_paths_test,
    truth_dataset_format_tuples_train, truth_dataset_format_tuples_test
) = train_test_split(
    wav_file_paths, truth_dataset_format_tuples, test_size=0.2, random_state=42
)

pitch_detector = CnnPitchDetector.from_zip('../models/pitch_detection/20170518_1718.zip')

data_train, list_of_pitches_train = read_X_y(wav_file_paths_train, truth_dataset_format_tuples_train,
                                             pitch_detector.feature_extractor.frame_rate_hz,
                                             pitch_detector.feature_extractor.sample_rate,
                                             pitch_detector.feature_extractor.subsampling_step,
                                             pitch_detector.feature_extractor.min_pitch,
                                             pitch_detector.feature_extractor.max_pitch)
X_train, y_train = pitch_detector.feature_extractor.transform(data_train, list_of_pitches_train)

data_test, list_of_pitches_test = read_X_y(wav_file_paths_test, truth_dataset_format_tuples_test,
                                           pitch_detector.feature_extractor.frame_rate_hz,
                                           pitch_detector.feature_extractor.sample_rate,
                                           pitch_detector.feature_extractor.subsampling_step,
                                           pitch_detector.feature_extractor.min_pitch,
                                           pitch_detector.feature_extractor.max_pitch)
X_test, y_test = pitch_detector.feature_extractor.transform(data_test, list_of_pitches_test)

y_train_probas = pitch_detector.model.predict(X_train)
y_test_probas = pitch_detector.model.predict(X_test)

print('TRAIN')
print(y_train_probas.shape)
for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    print(threshold)
    print(sklearn.metrics.accuracy_score(y_train, y_train_probas > threshold))
    print(sklearn.metrics.classification_report(y_train, y_train_probas > threshold))
print('TEST')
print(y_test_probas.shape)
for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    print(threshold)
    print(sklearn.metrics.accuracy_score(y_test, y_test_probas > threshold))
    print(sklearn.metrics.classification_report(y_test, y_test_probas > threshold))
