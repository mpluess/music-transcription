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
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from warnings import warn
from zipfile import ZipFile

from music_transcription.pitch_detection.read_data import read_X_y


class CnnFeatureExtractor(BaseEstimator, TransformerMixin):
    EPSILON = 1e-7

    def __init__(self, frame_rate_hz, sample_rate, subsampling_step, image_data_format, winlen_nfft_per_channel,
                 onset_group_threshold_seconds, min_pitch, max_pitch):
        self.frame_rate_hz = frame_rate_hz
        self.sample_rate = sample_rate
        self.subsampling_step = subsampling_step
        self.image_data_format = image_data_format
        self.winlen_nfft_per_channel = winlen_nfft_per_channel
        self.onset_group_threshold_seconds = onset_group_threshold_seconds
        self.min_pitch = min_pitch
        self.max_pitch = max_pitch

        self.standard_scalers_per_channel = None
        self.label_binarizer = None

    def fit(self, data, y=None):
        list_of_samples, _ = data

        # TODO fit_transform so spectrograms are only done once
        X_channels, _ = self._extract_spectrogram_features(list_of_samples)

        # TODO standardize only for regions with onset?
        print('Fitting standard scalers for each channel and band')
        self.standard_scalers_per_channel = []
        for X in X_channels:
            standard_scalers = []
            for j in range(X.shape[1]):
                standard_scaler = StandardScaler()
                standard_scaler.fit(X[:, j:j + 1])
                standard_scalers.append(standard_scaler)
            self.standard_scalers_per_channel.append(standard_scalers)

        self.label_binarizer = MultiLabelBinarizer(classes=range(self.min_pitch, self.max_pitch + 1))
        self.label_binarizer.fit(None)

        return self

    def transform(self, data, list_of_pitches=None):
        list_of_samples, list_of_onset_times = data

        X_channels, n_frames_after_cutoff_per_file = self._extract_spectrogram_features(list_of_samples)
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

        list_of_onset_times_grouped, list_of_pitches_grouped = self._group_onsets(list_of_onset_times, list_of_pitches)
        for i in range(len(X_channels)):
            X_channels[i] = self._get_X_after_onset_with_context(X_channels[i], list_of_onset_times_grouped, n_frames_after_cutoff_per_file)
            print(X_channels[i].shape)

        if list_of_pitches_grouped is None:
            y = None
        else:
            pitch_groups_flat = [pitch_group for pitches_grouped in list_of_pitches_grouped for pitch_group in pitches_grouped]
            assert len(pitch_groups_flat) == X_channels[0].shape[0]
            y = self.label_binarizer.transform(pitch_groups_flat)

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
        # TODO concatenate and delete one by one to use less memory at once
        # TODO axis should change depending on image_data_format (1 vs. 3)
        X = np.concatenate(X_channels, axis=1)
        print(X.shape)

        return X, y

    def _extract_spectrogram_features(self, list_of_samples):
        print('Creating spectrograms')
        n_frames_after_cutoff_per_file = [None] * len(list_of_samples)
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
            transformed = [self._extract_spectrogram_features_X(samples, n_frames, winlen=winlen, nfft=nfft)
                           for samples, n_frames
                           in zip(list_of_samples, n_frames_after_cutoff_per_file)]
            X = np.concatenate([t[0] for t in transformed])
            n_frames_after_cutoff_per_file = [t[1] for t in transformed]
            X_channels.append(X)

        return X_channels, n_frames_after_cutoff_per_file

    def _extract_spectrogram_features_X(self, samples, n_frames, log_transform_magnitudes=True,
                                        winlen=0.046, nfilt=80, nfft=2048,
                                        lowfreq=27.5, highfreq=16000, preemph=0):
        winstep = 1 / self.frame_rate_hz
        if log_transform_magnitudes:
            filterbank = logfbank(samples, self.sample_rate, winlen=winlen, winstep=winstep, nfilt=nfilt,
                                  nfft=nfft, lowfreq=lowfreq, highfreq=highfreq, preemph=preemph)
        else:
            filterbank, _ = fbank(samples, self.sample_rate, winlen=winlen, winstep=winstep, nfilt=nfilt,
                                  nfft=nfft, lowfreq=lowfreq, highfreq=highfreq, preemph=preemph)

        if n_frames is None:
            n_frames = filterbank.shape[0]
        return filterbank[:n_frames, :], n_frames

    def _group_onsets(self, list_of_onset_times, list_of_pitches):
        """Assumes onset times per file are sorted (pitch_detection.read_data.read_y does that).

        Group onsets and corresponding pitches in a way that onsets closer than self.onset_group_threshold_seconds
        belong to the same group.
        """

        # TODO list_of_pitches is None
        list_of_onset_times_grouped = []
        list_of_pitches_grouped = []
        return_pitches = True
        if list_of_pitches is None:
            return_pitches = False
            list_of_pitches = [[0] * len(onset_times) for onset_times in list_of_onset_times]
        for onset_times, pitches in zip(list_of_onset_times, list_of_pitches):
            onset_times_grouped = []
            pitches_grouped = []
            last_onset = None
            onset_group_start = None
            onset_group_pitches = set()
            for onset_time, pitch in zip(onset_times, pitches):
                if last_onset is not None and onset_time - last_onset > self.onset_group_threshold_seconds - self.EPSILON:
                    onset_times_grouped.append(onset_group_start)
                    pitches_grouped.append(onset_group_pitches)
                    onset_group_start = None
                    onset_group_pitches = set()
                last_onset = onset_time
                if onset_group_start is None:
                    onset_group_start = onset_time
                onset_group_pitches.add(pitch)
            onset_times_grouped.append(onset_group_start)
            pitches_grouped.append(onset_group_pitches)

            list_of_onset_times_grouped.append(onset_times_grouped)
            list_of_pitches_grouped.append(pitches_grouped)

        if return_pitches:
            return list_of_onset_times_grouped, list_of_pitches_grouped
        else:
            return list_of_onset_times_grouped, None

    def _get_X_after_onset_with_context(self, X, list_of_onset_times_grouped, n_frames_after_cutoff_per_file,
                                        c=10, border_value=0.0):
        n_samples = X.shape[0]
        assert n_samples == sum(n_frames_after_cutoff_per_file)

        # [10 15 5] -> [5 10 15]
        start_index_per_file = np.roll(n_frames_after_cutoff_per_file, 1)
        # [5 10 15] -> [0 10 15]
        start_index_per_file[0] = 0
        # [0 10 15] -> [0 10 25]
        start_index_per_file = np.cumsum(start_index_per_file)
        assert len(start_index_per_file) == len(list_of_onset_times_grouped)

        start_indices = []
        for start_index, n_frames, onset_times_grouped in zip(
            start_index_per_file, n_frames_after_cutoff_per_file, list_of_onset_times_grouped
        ):
            for onset_time in onset_times_grouped:
                index = int(onset_time * self.frame_rate_hz)
                assert index < n_frames
                if n_frames - index - 1 < c:
                    warn('Onset is too close to end of file ({} < {}), '
                         'operation will add context from different file!'.format(n_frames - index - 1, c))
                start_indices.append(start_index + index)

        n_onsets = len(start_indices)
        filterbank_size = X.shape[1]
        X_new = np.empty((n_onsets, c + 1, filterbank_size))
        for i, start_index in enumerate(start_indices):
            for offset in range(0, c + 1):
                if start_index + offset < n_samples:
                    X_new[i, offset, :] = X[start_index + offset, :]
                else:
                    X_new[i, offset].fill(border_value)
        return X_new


class CnnPitchDetector:
    FEATURE_EXTRACTOR_FILE = 'feature_extractor.pickle'
    MODEL_FILE = 'model.json'
    WEIGHTS_FILE = 'weights.hdf5'

    LOSS = 'binary_crossentropy'
    OPTIMIZER = 'adam'
    METRICS = ['accuracy']
    BATCH_SIZE = 1024

    def __init__(self, feature_extractor=None, model=None,
                 frame_rate_hz=100, sample_rate=44100, subsampling_step=1, image_data_format='channels_first',
                 winlen_nfft_per_channel=((0.023, 1024), (0.046, 2048), (0.092, 4096)),
                 onset_group_threshold_seconds=0.03,
                 tuning=(64, 59, 55, 50, 45, 40), n_frets=24):
        if feature_extractor is None:
            min_pitch = min(tuning)
            max_pitch = max(tuning) + n_frets
            self.feature_extractor = CnnFeatureExtractor(frame_rate_hz=frame_rate_hz,
                                                         sample_rate=sample_rate,
                                                         subsampling_step=subsampling_step,
                                                         image_data_format=image_data_format,
                                                         winlen_nfft_per_channel=winlen_nfft_per_channel,
                                                         onset_group_threshold_seconds=onset_group_threshold_seconds,
                                                         min_pitch=min_pitch,
                                                         max_pitch=max_pitch)
        else:
            self.feature_extractor = feature_extractor

        self.model = model

    @classmethod
    def from_zip(cls, path_to_zip, work_dir='zip_tmp'):
        """Load CnnPitchDetector from a zipfile containing a pickled CnnFeatureExtractor, a Keras model JSON file and a Keras weights HDF5 file."""

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
        """Load feature extractor"""

        with open(os.path.join(path_to_model_folder, cls.FEATURE_EXTRACTOR_FILE), 'rb') as f:
            feature_extractor = pickle.load(f)
        return feature_extractor

    @classmethod
    def _load_model(cls, path_to_model_folder):
        """Load and compile Keras model"""

        with open(os.path.join(path_to_model_folder, cls.MODEL_FILE)) as f:
            model = model_from_json(f.read())
        model.load_weights(os.path.join(path_to_model_folder, cls.WEIGHTS_FILE))

        model.compile(loss=cls.LOSS, optimizer=cls.OPTIMIZER, metrics=cls.METRICS)

        return model

    def save(self, path_to_zip, work_dir='zip_tmp_pitch'):
        """Save this CnnPitchDetector to a zipfile containing a pickled CnnFeatureExtractor, a Keras model JSON file and a Keras weights HDF5 file."""

        if os.path.exists(path_to_zip):
            path_to_zip_orig = path_to_zip
            path_to_zip = 'CnnPitchDetector_model_' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.zip'
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

    def fit(self, wav_file_paths_train, truth_dataset_format_tuples_train,
            wav_file_paths_val=None, truth_dataset_format_tuples_val=None):
        data_train, list_of_pitches_train = read_X_y(wav_file_paths_train, truth_dataset_format_tuples_train,
                                                     self.feature_extractor.frame_rate_hz,
                                                     self.feature_extractor.sample_rate,
                                                     self.feature_extractor.subsampling_step,
                                                     self.feature_extractor.min_pitch, self.feature_extractor.max_pitch)
        self.feature_extractor.fit(data_train)
        X_train, y_train = self.feature_extractor.transform(data_train, list_of_pitches_train)
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
        print(y_train)

        if wav_file_paths_val is not None and truth_dataset_format_tuples_val is not None:
            data_val, list_of_pitches_val = read_X_y(wav_file_paths_val, truth_dataset_format_tuples_val,
                                                     self.feature_extractor.frame_rate_hz,
                                                     self.feature_extractor.sample_rate,
                                                     self.feature_extractor.subsampling_step,
                                                     self.feature_extractor.min_pitch,
                                                     self.feature_extractor.max_pitch)
            X_val, y_val = self.feature_extractor.transform(data_val, list_of_pitches_val)
            validation_data = (X_val, y_val)
            print(y_val)
        else:
            validation_data = None

        self.model = self._create_model(input_shape,
                                        self.feature_extractor.max_pitch - self.feature_extractor.min_pitch + 1)
        self.model.fit(X_train, y_train,
                       epochs=500,
                       batch_size=self.BATCH_SIZE,
                       callbacks=[EarlyStopping(monitor='loss', patience=5)], verbose=2,
                       validation_data=validation_data)

    @classmethod
    def _create_model(cls, input_shape, n_output_units):
        """Keras model description

        Multi-label classification with Keras: https://github.com/fchollet/keras/issues/741
        """

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
        model.add(Dense(n_output_units))
        model.add(Activation('sigmoid'))

        model.compile(loss=cls.LOSS, optimizer=cls.OPTIMIZER, metrics=cls.METRICS)

        return model
