import datetime
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, model_from_json
import numpy as np
import os
import pickle
from python_speech_features import fbank, logfbank
import shutil
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from zipfile import ZipFile

from music_transcription.onset_detection.abstract_onset_detector import AbstractOnsetDetector
from music_transcription.onset_detection.metrics import onset_metric
from music_transcription.onset_detection.read_data import group_onsets, read_X, read_X_y


class CnnFeatureExtractor(BaseEstimator, TransformerMixin):
    """Transformer doing feature extraction for the onset detection CNN.

    Method is described in detail here: http://www.ofai.at/~jan.schlueter/pubs/2014_icassp.pdf
    For a given frame rate and sample rate, extracts one or more spectrogram excerpts per frame.
    If more than one spectrogram excerpt per frame is created, they are organized in channels.
    """

    def __init__(self, frame_rate_hz, sample_rate, subsampling_step, image_data_format, winlen_nfft_per_channel):
        self.frame_rate_hz = frame_rate_hz
        self.sample_rate = sample_rate
        self.subsampling_step = subsampling_step
        self.image_data_format = image_data_format
        self.winlen_nfft_per_channel = winlen_nfft_per_channel

        self.standard_scalers_per_channel = None

    def fit(self, wav_file_paths, y=None):
        """Fit standard scalers per channel and band."""

        # TODO fit_transform so spectrograms are only done once
        X_channels = self._read_and_extract(wav_file_paths)

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

    def transform(self, wav_file_paths, truth_dataset_format_tuples=None):
        """Transform wave files to a feature matrix.

        Input:
        wav_file_paths: List of paths to wave files
        truth_dataset_format_tuples: List of tuples (path_to_truth, dataset, truth_format).
        See music_transcription.onset_detection.read_data.get_wav_and_truth_files.
        If this is set, also returns the labels y (with neighbors) and y_actual_onset_only.
        If not, None is returned for y and y_actual_onset_only.

        Output: Tuple (X, y, y_actual_onset_only)
        """

        if truth_dataset_format_tuples is None:
            X_channels = self._read_and_extract(wav_file_paths)
            y = None
            y_actual_onset_only = None
        else:
            X_channels, y, y_actual_onset_only = self._read_and_extract_with_labels(wav_file_paths, truth_dataset_format_tuples)

        if X_channels is None:
            return None, None, None

        # for X in X_channels:
        #     print(X.shape)
        #     print(X.mean())
        #     print(X.std())

        # print('Standardizing for each channel and band')
        for X, standard_scalers in zip(X_channels, self.standard_scalers_per_channel):
            for j, ss in enumerate(standard_scalers):
                X[:, j:j + 1] = ss.transform(X[:, j:j + 1])
        # for X in X_channels:
        #     print(X.mean())
        #     print(X.std())

        for i in range(len(X_channels)):
            X_channels[i] = self._get_X_with_context_frames(X_channels[i])
            # print(X_channels[i].shape)

        # print('Reshaping data')
        img_rows, img_cols = (X_channels[0].shape[1], X_channels[0].shape[2])
        for i in range(len(X_channels)):
            # Theano is 3 times faster with channels_first vs. channels_last on MNIST, so this setting matters.
            # "image_data_format": "channels_first" @ %USERPROFILE%/.keras/keras.json
            if self.image_data_format == 'channels_first':
                X_channels[i] = X_channels[i].reshape(X_channels[i].shape[0], 1, img_rows, img_cols)
            else:
                X_channels[i] = X_channels[i].reshape(X_channels[i].shape[0], img_rows, img_cols, 1)
            # print(X_channels[i].shape)

        # print('Concatenating channels')
        # TODO concatenate and delete one by one to use less memory at once
        # TODO axis should change depending on image_data_format (1 vs. 3)
        X = np.concatenate(X_channels, axis=1)
        # print(X.shape)

        return X, y, y_actual_onset_only

    def _read_and_extract(self, wav_file_paths):
        """Read wave files, extract spectrogram features and return a feature matrix with shape (n_frames_all_files, n_bands) per channel."""

        # print('Reading wave files')
        X_parts = []
        for path_to_wav in wav_file_paths:
            X_part, file_length_seconds = read_X(path_to_wav, self.frame_rate_hz, self.sample_rate, self.subsampling_step)
            if X_part is not None:
                X_parts.append(X_part)

        if len(X_parts) == 0:
            return None

        # print('Creating spectrograms')
        X_channels, n_frames_after_cutoff_per_file = self._extract_spectrogram_features(X_parts)

        return X_channels

    def _read_and_extract_with_labels(self, wav_file_paths, truth_dataset_format_tuples):
        """Return a tuple (X_channels, y, y_actual_onset_only)

        X_channels: list of one feature matrix with shape (n_frames_all_files, n_bands) per channel
        y, y_actual_onset_only: corresponding onset labels
        """

        print('Reading wave files and labels')
        X_parts = []
        y_parts = []
        y_actual_onset_only_parts = []
        for path_to_wav, truth_dataset_format_tuple in zip(wav_file_paths, truth_dataset_format_tuples):
            path_to_truth, dataset, truth_format = truth_dataset_format_tuple
            X_part, y_part, y_actual_onset_only_part = read_X_y(path_to_wav, self.frame_rate_hz,
                                                                self.sample_rate, self.subsampling_step,
                                                                path_to_truth, truth_format, dataset)
            if X_part is not None and y_part is not None and y_actual_onset_only_part is not None:
                X_parts.append(X_part)
                y_parts.append(y_part)
                y_actual_onset_only_parts.append(y_actual_onset_only_part)

        if len(X_parts) == 0:
            return None, None, None

        print('Creating spectrograms')
        X_channels, n_frames_after_cutoff_per_file = self._extract_spectrogram_features(X_parts)
        # Cut labels to the same size as X. Number of frames that are cut off is defined by the largest window size
        # of all channels.
        y = np.concatenate([y_part[:n_frames]
                            for y_part, n_frames
                            in zip(y_parts, n_frames_after_cutoff_per_file)])
        y_actual_onset_only = np.concatenate([y_actual_onset_only_part[:n_frames]
                                              for y_actual_onset_only_part, n_frames
                                              in zip(y_actual_onset_only_parts, n_frames_after_cutoff_per_file)])

        return X_channels, y, y_actual_onset_only

    def _extract_spectrogram_features(self, X_parts):
        """Return a tuple (X_channels, n_frames_after_cutoff_per_file)

        X_channels: list of one feature matrix with shape (n_frames_all_files, n_bands) per channel
        n_frames_after_cutoff_per_file: number of frames that remain after the last (max(winlen of channels) - winstep)
        seconds are cut off by _extract_spectrogram_features_X.
        """

        n_frames_after_cutoff_per_file = [None] * len(X_parts)
        X_channels = []
        # Create 3 channels with different window length.
        # Make sure to run the largest window first which cuts off the most at the end of the file.
        # Return and reuse the number of frames for each part = each file for the other nfft values.
        for winlen, nfft in sorted(
                # 3 channels:
                # ((0.023, 1024), (0.046, 2048), (0.092, 4096)),

                # 1 channel:
                # ((0.046, 2048),)
                self.winlen_nfft_per_channel,

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
        """Extract spectrogram features for one file.

        If n_frames is None: Last (winlen - winstep) seconds will be cut off.
        Else: n_frames will be kept, the rest is cut off.

        Returns a tuple (filterbank, n_frames)
        filterbank: feature matrix with shape (n_frames, n_bands)
        n_frames: if n_frames is already set: return unchanged, else: return filterbank.shape[0]
        """

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


class CnnOnsetDetector(AbstractOnsetDetector):
    CONFIG_FILE = 'config.pickle'
    FEATURE_EXTRACTOR_FILE = 'feature_extractor.pickle'
    MODEL_FILE = 'model.json'
    WEIGHTS_FILE = 'weights.hdf5'

    LOSS = 'binary_crossentropy'
    OPTIMIZER = 'adam'
    METRICS = ['accuracy']
    BATCH_SIZE = 1024

    def __init__(self,
                 # loaded config, feature extractor and model
                 config=None, feature_extractor=None, model=None,

                 # config params
                 onset_group_threshold_seconds=0.05,

                 # feature extractor params
                 frame_rate_hz=100, sample_rate=44100, subsampling_step=1, image_data_format='channels_first',
                 winlen_nfft_per_channel=((0.023, 1024), (0.046, 2048), (0.092, 4096))):
        if config is None:
            super().__init__(onset_group_threshold_seconds)
        else:
            self.config = config

        if feature_extractor is None:
            self.feature_extractor = CnnFeatureExtractor(frame_rate_hz=frame_rate_hz,
                                                         sample_rate=sample_rate,
                                                         subsampling_step=subsampling_step,
                                                         image_data_format=image_data_format,
                                                         winlen_nfft_per_channel=winlen_nfft_per_channel)
        else:
            self.feature_extractor = feature_extractor

        self.model = model

    @classmethod
    def from_zip(cls, path_to_zip, work_dir='zip_tmp_onset'):
        """Load CnnOnsetDetector from a zipfile containing a pickled config dict, a pickled CnnFeatureExtractor,
        a Keras model JSON file and a Keras weights HDF5 file."""

        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir)

        with ZipFile(path_to_zip) as zip_file:
            zip_file.extractall(path=work_dir)
            config = cls._load_pickled_object(work_dir, cls.CONFIG_FILE)
            feature_extractor = cls._load_pickled_object(work_dir, cls.FEATURE_EXTRACTOR_FILE)
            model = cls._load_model(work_dir)
        shutil.rmtree(work_dir)

        return cls(config=config, feature_extractor=feature_extractor, model=model)

    @classmethod
    def _load_pickled_object(cls, path_to_model_folder, filename):
        """Load pickled object"""

        with open(os.path.join(path_to_model_folder, filename), 'rb') as f:
            loaded_object = pickle.load(f)
        return loaded_object

    @classmethod
    def _load_model(cls, path_to_model_folder):
        """Load and compile Keras model"""

        with open(os.path.join(path_to_model_folder, cls.MODEL_FILE)) as f:
            model = model_from_json(f.read())
        model.load_weights(os.path.join(path_to_model_folder, cls.WEIGHTS_FILE))

        model.compile(loss=cls.LOSS, optimizer=cls.OPTIMIZER, metrics=cls.METRICS)

        return model

    def save(self, path_to_zip, work_dir='zip_tmp'):
        """Save this CnnOnsetDetector to a zipfile containing a pickled config, a pickled CnnFeatureExtractor,
        a Keras model JSON file and a Keras weights HDF5 file."""

        if os.path.exists(path_to_zip):
            path_to_zip_orig = path_to_zip
            path_to_zip = 'CnnOnsetDetector_model_' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.zip'
            print('Zip file {} exists, writing to {}'.format(path_to_zip_orig, path_to_zip))

        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir)

        to_zip = []
        path_to_file = os.path.join(work_dir, self.CONFIG_FILE)
        with open(path_to_file, 'wb') as f:
            pickle.dump(self.config, f)
        to_zip.append(path_to_file)

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

    def fit(self, wav_file_paths_train, truth_dataset_format_tuples_train,
            wav_file_paths_val=None, truth_dataset_format_tuples_val=None):
        """Fit onset detector to the supplied wave files and labels.

        truth_dataset_format_tuples: List of tuples (path_to_truth, dataset, truth_format).
        See music_transcription.onset_detection.read_data.get_wav_and_truth_files.

        wav_file_paths_val and truth_dataset_format_tuples_val are used as a validation set during training if set.

        Fit CnnFeatureExtractor and the Keras model created by _create_model.
        """

        self.feature_extractor.fit(wav_file_paths_train)
        X_train, y_train, y_actual_onset_only_train = self.feature_extractor.transform(wav_file_paths_train,
                                                                                       truth_dataset_format_tuples_train)
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
        print('y_train.sum()={}'.format(y_train.sum()))
        print('y_actual_onset_only_train.sum()={}'.format(y_actual_onset_only_train.sum()))

        if wav_file_paths_val is not None and truth_dataset_format_tuples_val is not None:
            X_val, y_val, y_actual_onset_only_val = self.feature_extractor.transform(wav_file_paths_val,
                                                                                     truth_dataset_format_tuples_val)
            print('y_val.sum()={}'.format(y_val.sum()))
            print('y_actual_onset_only_val.sum()={}'.format(y_actual_onset_only_val.sum()))
            validation_data = (X_val, y_val)
        else:
            validation_data = None

        self.model = self._create_model(input_shape)
        self.model.fit(X_train, y_train,
                       epochs=500,
                       batch_size=self.BATCH_SIZE,
                       callbacks=[EarlyStopping(monitor='loss', patience=5)], verbose=2,
                       validation_data=validation_data)

    @classmethod
    def _create_model(cls, input_shape):
        """Keras model description"""

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

    def predict_onsets(self, path_to_wav_file):
        classes_filtered = self._predict_classes_filtered(path_to_wav_file)
        if classes_filtered is None:
            return None

        onset_indices_filtered = classes_filtered.nonzero()[0]

        frame_rate_hz = self.feature_extractor.frame_rate_hz
        onset_times = [index / frame_rate_hz for index in onset_indices_filtered]
        onset_times_grouped = group_onsets(onset_times, self.config['onset_group_threshold_seconds'])

        return onset_times_grouped

    def predict_print_metrics(self, wav_file_paths, truth_dataset_format_tuples):
        """Legacy method, use predict_onsets / see benchmark for how to get metrics."""

        X, y, y_actual_onset_only = self.feature_extractor.transform(wav_file_paths, truth_dataset_format_tuples)

        print('unfiltered:')
        y_predicted = self.model.predict_classes(X, batch_size=self.BATCH_SIZE).ravel()
        self._print_metrics(y, y_actual_onset_only, y_predicted)

        print('filtered:')
        probas = self.model.predict_proba(X, batch_size=self.BATCH_SIZE).ravel()
        y_predicted_filtered = self._filter_classes(y_predicted, probas)
        self._print_metrics(y, y_actual_onset_only, y_predicted_filtered)

    def _predict_classes_filtered(self, path_to_wav_file):
        X, _, _ = self.feature_extractor.transform([path_to_wav_file])
        if X is None:
            return None

        classes = self.model.predict_classes(X, batch_size=self.BATCH_SIZE, verbose=0).ravel()
        probas = self.model.predict_proba(X, batch_size=self.BATCH_SIZE, verbose=0).ravel()

        return self._filter_classes(classes, probas)

    @staticmethod
    def _filter_classes(classes, probas):
        """Filter duplicate onsets caused by the labeling of neighbors during training"""

        onset_indices_unfiltered = classes.nonzero()[0]
        if len(onset_indices_unfiltered) == 0:
            return onset_indices_unfiltered

        onset_indices_filtered = []
        last_index = -2
        onset_group = []
        for index in onset_indices_unfiltered:
            if index - last_index == 1:
                onset_group.append(index)
            else:
                if len(onset_group) > 0:
                    index_with_highest_proba = max(onset_group, key=lambda i: probas[i])
                    onset_indices_filtered.append(index_with_highest_proba)
                onset_group = [index]
            last_index = index
        index_with_highest_proba = max(onset_group, key=lambda i: probas[i])
        onset_indices_filtered.append(index_with_highest_proba)

        classes_filtered = np.zeros(len(classes), dtype=np.int8)
        classes_filtered[onset_indices_filtered] = 1

        return classes_filtered

    @staticmethod
    def _print_metrics(y, y_actual_onset_only, y_predicted):
        print(classification_report(y, y_predicted))
        print(onset_metric(y, y_actual_onset_only, y_predicted, n_tolerance_frames_plus_minus=2))
        print(onset_metric(y, y_actual_onset_only, y_predicted, n_tolerance_frames_plus_minus=5))
        print('')
