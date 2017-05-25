from sklearn.model_selection import train_test_split

from music_transcription.pitch_detection.cnn_pitch_detection import CnnPitchDetector
from music_transcription.pitch_detection.read_data import get_wav_and_truth_files

active_datasets = {1, 2, 3}

wav_file_paths, truth_dataset_format_tuples = get_wav_and_truth_files(active_datasets)

(
    wav_file_paths_train, wav_file_paths_test,
    truth_dataset_format_tuples_train, truth_dataset_format_tuples_test
) = train_test_split(
    wav_file_paths, truth_dataset_format_tuples, test_size=0.2, random_state=42
)

pitch_detector = CnnPitchDetector()
pitch_detector.fit(
    wav_file_paths_train, truth_dataset_format_tuples_train,
    wav_file_paths_test, truth_dataset_format_tuples_test,
)
pitch_detector.save('../models/pitch_detection/20170525_1345.zip')
