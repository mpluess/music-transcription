from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Activation, Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.wrappers.scikit_learn import KerasClassifier

import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from music_transcription.onset_detection.metrics import onset_metric
from music_transcription.onset_detection.read_data import get_wav_and_truth_files
from music_transcription.string_fret_detection.read_data import read_data_y
from music_transcription.string_fret_detection.cnn_string_detector import CnnStringDetector, CnnStringFeatureExtractor


active_datasets = {1, 2, 3, 4}
active_datasets = {1}
# X_parts, y_parts, y_start_only_parts, ds_labels
wav_file_paths, truth_dataset_format_tuples = get_wav_and_truth_files(active_datasets)
wav_file_paths_train, wav_file_paths_test, truth_dataset_format_tuples_train, truth_dataset_format_tuples_test = train_test_split(
    wav_file_paths, truth_dataset_format_tuples, test_size=0.2, random_state=42
)



"""feature_extractor = CnnStringFeatureExtractor()
data_train, y_train, _, _ = read_data_y(wav_file_paths_train, truth_dataset_format_tuples_train, 44100, 1, 6, 0.05)
X_train = feature_extractor.fit_transform(data_train)"""

cnn_string_detector = CnnStringDetector()
cnn_string_detector.fit(wav_file_paths_train, truth_dataset_format_tuples_train,
                        wav_file_paths_test, truth_dataset_format_tuples_test)
pitch_detector.save('../models/string_detection/20170706_1640_ds1.zip')
