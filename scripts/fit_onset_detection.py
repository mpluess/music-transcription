from music_transcription.onset_detection.cnn_onset_detection import CnnOnsetDetector
from sklearn.model_selection import train_test_split

from music_transcription.onset_detection.read_data import get_wav_and_truth_files

active_datasets = {1, 2, 3, 4}

wav_file_paths, truth_dataset_format_tuples = get_wav_and_truth_files(active_datasets)
wav_file_paths_train, wav_file_paths_test, truth_dataset_format_tuples_train, truth_dataset_format_tuples_test = train_test_split(
    wav_file_paths, truth_dataset_format_tuples, test_size=0.2, random_state=42
)

onset_detector = CnnOnsetDetector()
onset_detector.fit(wav_file_paths_train, truth_dataset_format_tuples_train,
                   wav_file_paths_test, truth_dataset_format_tuples_test)
onset_detector.save('../models/20170511-1-channel_ds1-4_80-perc_adjusted-labels_10-epochs.zip')

print('TRAIN')
onset_detector.predict_print_metrics(wav_file_paths_train, truth_dataset_format_tuples_train)
print('TEST')
onset_detector.predict_print_metrics(wav_file_paths_test, truth_dataset_format_tuples_test)
