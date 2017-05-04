import datetime
from keras.callbacks import EarlyStopping
from keras.models import Sequential, model_from_json
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
import numpy as np
import os
import pickle
from python_speech_features import fbank, logfbank
import shutil
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from zipfile import ZipFile

from onset_detection.metrics import onset_metric
from onset_detection.read_data import read_X, read_y


class CnnFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, frame_rate_hz, sample_rate, subsampling_step, image_data_format):
        self.frame_rate_hz = frame_rate_hz
        self.sample_rate = sample_rate
        self.subsampling_step = subsampling_step
        self.image_data_format = image_data_format

        self.standard_scalers_per_channel = None
        self.file_lengths_seconds = None
        self.n_frames_after_cutoff_per_file = None

    def fit(self, wav_file_paths, y=None):
        # TODO fit_transform so spectrograms are only done once
        X_channels, file_lengths_seconds, n_frames_after_cutoff_per_file = self._read_and_extract(wav_file_paths)
        self.file_lengths_seconds = file_lengths_seconds
        self.n_frames_after_cutoff_per_file = n_frames_after_cutoff_per_file

        print('Fitting standard scalers for each channel and band')
        self.standard_scalers_per_channel = []
        for X in X_channels:
            standard_scalers = []
            for j in range(X.shape[1]):
                standard_scaler = StandardScaler()
                standard_scaler.fit(X[:, j:j + 1])
                standard_scalers.append(standard_scaler)
            self.standard_scalers_per_channel.append(standard_scalers)

        return self

    def transform(self, wav_file_paths):
        X_channels, file_lengths_seconds, n_frames_after_cutoff_per_file = self._read_and_extract(wav_file_paths)
        for X in X_channels:
            print(X.shape)
            print(X.mean())
            print(X.std())

        print('Standardizing for each channel and band')
        for X, standard_scalers in zip(X_channels, self.standard_scalers_per_channel):
            for j, ss in enumerate(standard_scalers):
                X[:, j:j + 1] = ss.transform(X[:, j:j + 1])
        for X in X_channels:
            print(X.mean())
            print(X.std())

        for i in range(len(X_channels)):
            X_channels[i] = self._get_X_with_context_frames(X_channels[i])
            print(X_channels[i].shape)

        print('Reshaping data')
        img_rows, img_cols = (X_channels[0].shape[1], X_channels[0].shape[2])
        for i in range(len(X_channels)):
            # Theano is 3 times faster with channels_first vs. channels_last on MNIST, so this setting matters.
            # "image_data_format": "channels_first" @ %USERPROFILE%/.keras/keras.json
            if self.image_data_format == 'channels_first':
                X_channels[i] = X_channels[i].reshape(X_channels[i].shape[0], 1, img_rows, img_cols)
            else:
                X_channels[i] = X_channels[i].reshape(X_channels[i].shape[0], img_rows, img_cols, 1)
            print(X_channels[i].shape)

        print('Concatenating channels')
        X = np.concatenate(X_channels, axis=1)
        print(X.shape)

        return X, file_lengths_seconds, n_frames_after_cutoff_per_file

    def _read_and_extract(self, wav_file_paths):
        print('Reading wave files')
        X_parts = []
        file_lengths_seconds = []
        for path_to_wav in wav_file_paths:
            X_part, file_length_seconds = read_X(path_to_wav, self.frame_rate_hz, self.sample_rate, self.subsampling_step)
            if X_part is not None:
                X_parts.append(X_part)
                file_lengths_seconds.append(file_length_seconds)

        print('Creating spectrograms')
        X_channels, n_frames_after_cutoff_per_file = self._extract_spectrogram_features(X_parts)

        return X_channels, file_lengths_seconds, n_frames_after_cutoff_per_file

    def _extract_spectrogram_features(self, X_parts):
        n_frames_after_cutoff_per_file = [None] * len(X_parts)
        X_channels = []
        # Create 3 channels with different window length.
        # Make sure to run the largest window first which cuts off the most at the end of the file.
        # Return and reuse the number of frames for each part = each file for the other nfft values.
        for winlen, nfft in sorted(
                # [(0.023, 1024), (0.046, 2048), (0.092, 4096)],
                [(0.046, 2048)],

                key=lambda t: t[1], reverse=True
        ):
            transformed = [self._extract_spectrogram_features_X(X_part, n_frames, winlen=winlen, nfft=nfft)
                           for X_part, n_frames
                           in zip(X_parts, n_frames_after_cutoff_per_file)]
            X = np.concatenate([t[0] for t in transformed])
            n_frames_after_cutoff_per_file = [t[1] for t in transformed]
            X_channels.append(X)

        return X_channels, n_frames_after_cutoff_per_file

    def _extract_spectrogram_features_X(self, X_part, n_frames, log_transform_magnitudes=True,
                                        winlen=0.046, nfilt=80, nfft=2048,
                                        lowfreq=27.5, highfreq=16000, preemph=0):
        """Last (winlen - winstep) seconds will be cut off"""

        winstep = 1 / self.frame_rate_hz
        samples = X_part.ravel()
        if log_transform_magnitudes:
            filterbank = logfbank(samples, self.sample_rate, winlen=winlen, winstep=winstep, nfilt=nfilt,
                                  nfft=nfft, lowfreq=lowfreq, highfreq=highfreq, preemph=preemph)
        else:
            filterbank, _ = fbank(samples, self.sample_rate, winlen=winlen, winstep=winstep, nfilt=nfilt,
                                  nfft=nfft, lowfreq=lowfreq, highfreq=highfreq, preemph=preemph)

        if n_frames is None:
            n_frames = filterbank.shape[0]
        return filterbank[:n_frames, :], n_frames

    def _get_X_with_context_frames(self, X, c=7, border_value=0.0):
        """Return new X with new dimensions (X.shape[0] = n_samples, 2*c + 1, X.shape[1] = filterbank_size)

        One entry of X_new consists of c frames of context before the current frame,
        the current frame and another c frames of context after the current frame.
        """

        n_samples = X.shape[0]
        filterbank_size = X.shape[1]
        X_new = np.empty((n_samples, 2 * c + 1, filterbank_size))
        for i in range(n_samples):
            for offset in range(-c, c + 1):
                if i + offset > -1 and i + offset < n_samples:
                    # X_new 2nd dim: [0, 2*c + 1[
                    # X 1st dim: [i-c, i+c+1[
                    X_new[i, offset + c, :] = X[i + offset, :]
                else:
                    X_new[i, offset + c].fill(border_value)
        return X_new


class CnnOnsetDetector:
    FEATURE_EXTRACTOR_FILE = 'feature_extractor.pickle'
    MODEL_FILE = 'model.json'
    WEIGHTS_FILE = 'weights.hdf5'

    LOSS = 'binary_crossentropy'
    OPTIMIZER = 'adam'
    METRICS = ['accuracy']
    BATCH_SIZE = 1024

    def __init__(self, feature_extractor=None, model=None,
                 frame_rate_hz=100, sample_rate=44100, subsampling_step=1, image_data_format='channels_first'):
        if feature_extractor is None:
            self.feature_extractor = CnnFeatureExtractor(frame_rate_hz=frame_rate_hz,
                                                         sample_rate=sample_rate,
                                                         subsampling_step=subsampling_step,
                                                         image_data_format=image_data_format)
        else:
            self.feature_extractor = feature_extractor

        self.model = model

    @classmethod
    def from_zip(cls, path_to_zip, work_dir='zip_tmp'):
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir)

        with ZipFile(path_to_zip) as zip_file:
            zip_file.extractall(path=work_dir)
            feature_extractor = cls._load_feature_extractor(work_dir)
            model = cls._load_model(work_dir)
        shutil.rmtree(work_dir)

        return cls(feature_extractor, model)

    @classmethod
    def _load_feature_extractor(cls, path_to_model_folder):
        with open(os.path.join(path_to_model_folder, cls.FEATURE_EXTRACTOR_FILE), 'rb') as f:
            feature_extractor = pickle.load(f)
        return feature_extractor

    @classmethod
    def _load_model(cls, path_to_model_folder):
        with open(os.path.join(path_to_model_folder, cls.MODEL_FILE)) as f:
            model = model_from_json(f.read())
        model.load_weights(os.path.join(path_to_model_folder, cls.WEIGHTS_FILE))

        model.compile(loss=cls.LOSS, optimizer=cls.OPTIMIZER, metrics=cls.METRICS)

        return model

    def fit(self, wav_file_paths, truth_dataset_format_tuples):
        X, _, _ = self.feature_extractor.fit_transform(wav_file_paths)
        input_shape = (X.shape[1], X.shape[2], X.shape[3])
        y, y_actual_onset_only = self._get_labels(truth_dataset_format_tuples,
                                self.feature_extractor.file_lengths_seconds,
                                self.feature_extractor.n_frames_after_cutoff_per_file,
                                self.feature_extractor.frame_rate_hz)
        print('y.sum()={}'.format(y.sum()))
        print('y_actual_onset_only.sum()={}'.format(y_actual_onset_only.sum()))

        self.model = self._create_model(input_shape)
        self.model.fit(X, y,
                       # epochs=500,
                       epochs=10,
                       batch_size=self.BATCH_SIZE,
                       callbacks=[EarlyStopping(monitor='loss', patience=5)], verbose=2)

    @classmethod
    def _create_model(cls, input_shape):
        model = Sequential()

        model.add(Conv2D(10, (7, 3), padding='valid', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1, 3)))
        model.add(Conv2D(20, (3, 3), padding='valid'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(1, 3)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))

        model.compile(loss=cls.LOSS, optimizer=cls.OPTIMIZER, metrics=cls.METRICS)

        return model

    @staticmethod
    def _get_labels(truth_dataset_format_tuples, file_lengths_seconds, n_frames_after_cutoff_per_file,
                    frame_rate_hz):
        y_parts = []
        y_actual_onset_only_parts = []
        for truth_dataset_format, file_length_seconds, n_frames_after_cutoff in zip(
                truth_dataset_format_tuples,
                file_lengths_seconds,
                n_frames_after_cutoff_per_file
        ):
            path_to_truth, dataset, format = truth_dataset_format
            y_part, y_actual_onset_only_part = read_y(format, path_to_truth, file_length_seconds, frame_rate_hz, dataset)
            y_parts.append(y_part[:n_frames_after_cutoff])
            y_actual_onset_only_parts.append(y_actual_onset_only_part[:n_frames_after_cutoff])
        y = np.concatenate(y_parts)
        y_actual_onset_only = np.concatenate(y_actual_onset_only_parts)

        return y, y_actual_onset_only

    def predict_classes(self, path_to_wav_file):
        X, _, _ = self.feature_extractor.transform([path_to_wav_file])
        return self.model.predict_classes(X, batch_size=self.BATCH_SIZE).ravel()

    def predict_proba(self, path_to_wav_file):
        X, _, _ = self.feature_extractor.transform([path_to_wav_file])
        return self.model.predict_proba(X, batch_size=self.BATCH_SIZE).ravel()

    def predict_classes_proba(self, path_to_wav_file):
        X, _, _ = self.feature_extractor.transform([path_to_wav_file])
        return (
            self.model.predict_classes(X, batch_size=self.BATCH_SIZE).ravel(),
            self.model.predict_proba(X, batch_size=self.BATCH_SIZE).ravel()
        )

    def predict_onset_times_seconds(self, path_to_wav_file):
        classes, probas = self.predict_classes_proba(path_to_wav_file)
        frame_rate_hz = self.feature_extractor.frame_rate_hz

        # Filter duplicate onsets caused by the labeling of neighbors during training
        onset_indices_unfiltered = classes.nonzero()[0]
        last_index = -2
        onset_group = []
        onset_times_seconds = []
        for index in onset_indices_unfiltered:
            if index - last_index == 1:
                onset_group.append(index)
            else:
                if len(onset_group) > 0:
                    index_with_highest_proba = max(onset_group, key=lambda i: probas[i])
                    onset_times_seconds.append(index_with_highest_proba / frame_rate_hz)
                onset_group = [index]
            last_index = index
        index_with_highest_proba = max(onset_group, key=lambda i: probas[i])
        onset_times_seconds.append(index_with_highest_proba / frame_rate_hz)

        # Return all onsets
        # onset_times_seconds = [index / frame_rate_hz for index in classes.nonzero()]

        return onset_times_seconds

    def predict_print_metrics(self, wav_file_paths, truth_dataset_format_tuples):
        X, file_lengths_seconds, n_frames_after_cutoff_per_file = self.feature_extractor.transform(wav_file_paths)
        y_predicted = self.model.predict_classes(X, batch_size=self.BATCH_SIZE).ravel()

        y, y_actual_onset_only = self._get_labels(truth_dataset_format_tuples,
                                                  file_lengths_seconds,
                                                  n_frames_after_cutoff_per_file,
                                                  self.feature_extractor.frame_rate_hz)

        self._print_metrics(y, y_actual_onset_only, y_predicted)

    @staticmethod
    def _print_metrics(y, y_actual_onset_only, y_predicted):
        print(classification_report(y, y_predicted))
        print(onset_metric(y, y_actual_onset_only, y_predicted, n_tolerance_frames_plus_minus=2))
        print(onset_metric(y, y_actual_onset_only, y_predicted, n_tolerance_frames_plus_minus=5))
        print('')

    def save(self, path_to_zip, work_dir='zip_tmp'):
        if os.path.exists(path_to_zip):
            path_to_zip_orig = path_to_zip
            path_to_zip = 'CnnOnsetDetector_model_' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.zip'
            print('Zip file {} exists, writing to {}'.format(path_to_zip_orig, path_to_zip))

        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir)

        to_zip = []
        path_to_file = os.path.join(work_dir, self.FEATURE_EXTRACTOR_FILE)
        with open(path_to_file, 'wb') as f:
            pickle.dump(self.feature_extractor, f)
        to_zip.append(path_to_file)

        path_to_file = os.path.join(work_dir, self.MODEL_FILE)
        with open(path_to_file, 'w') as f:
            f.write(self.model.to_json())
        to_zip.append(path_to_file)

        path_to_file = os.path.join(work_dir, self.WEIGHTS_FILE)
        self.model.save_weights(path_to_file)
        to_zip.append(path_to_file)

        with ZipFile(path_to_zip, 'w') as zip_file:
            for path_to_file in to_zip:
                zip_file.write(path_to_file, arcname=os.path.basename(path_to_file))
        shutil.rmtree(work_dir)
