from sklearn.model_selection import train_test_split

from music_transcription.read_data import get_wav_and_truth_files
from music_transcription.string_fret_detection.cnn_string_detection import CnnStringDetector


# active_datasets = {1, 2, 3}
active_datasets = {1}
# X_parts, y_parts, y_start_only_parts, ds_labels
wav_file_paths, truth_dataset_format_tuples = get_wav_and_truth_files(active_datasets)
wav_file_paths_train, wav_file_paths_test, truth_dataset_format_tuples_train, truth_dataset_format_tuples_test = train_test_split(
    wav_file_paths, truth_dataset_format_tuples, test_size=0.2, random_state=42
)

cnn_string_detector = CnnStringDetector()
cnn_string_detector.fit(wav_file_paths_train, truth_dataset_format_tuples_train,
                        wav_file_paths_test, truth_dataset_format_tuples_test)
cnn_string_detector.save('../models/string_detection/20170706_1713_ds1.zip')
