import datetime
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential, model_from_json
from librosa import cqt
import numpy as np
import os
import pickle
import shutil
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from warnings import warn
from zipfile import ZipFile

from music_transcription.pitch_detection.abstract_pitch_detector import AbstractPitchDetector
from music_transcription.pitch_detection.read_data import read_samples, read_data_y


class CnnCqtFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, image_data_format,
                 sample_rate, hop_length, fmin, n_bins, bins_per_octave, scale):
        self.image_data_format = image_data_format

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.fmin = fmin
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.scale = scale

        self.frame_rate_hz = sample_rate / hop_length

        self.standard_scalers = None

    def fit(self, data, y=None, save_data=False):
        list_of_samples, list_of_onset_times = data

        X, n_frames_per_file = self._extract_spectrogram_features(list_of_samples)
        if save_data:
            self._X = X
            self._n_frames_per_file = n_frames_per_file
            self._list_of_onset_times = list_of_onset_times

        # TODO standardize only for regions with onset?
        print('Fitting standard scalers for each bin')
        self.standard_scalers = []
        for j in range(X.shape[1]):
            standard_scaler = StandardScaler()
            standard_scaler.fit(X[:, j:j + 1])
            self.standard_scalers.append(standard_scaler)

        return self

    def transform(self, data, load_data=False):
        if load_data:
            X = self._X
            n_frames_per_file = self._n_frames_per_file
            list_of_onset_times = self._list_of_onset_times
            self._X = None
            self._n_frames_per_file = None
            self._list_of_onset_times = None
        else:
            list_of_samples, list_of_onset_times = data
            X, n_frames_per_file = self._extract_spectrogram_features(list_of_samples)
        # print(X.shape)
        # print(X.mean())
        # print(X.std())

        # print('Standardizing for each band')
        for j, ss in enumerate(self.standard_scalers):
            X[:, j:j + 1] = ss.transform(X[:, j:j + 1])
        # print(X.mean())
        # print(X.std())

        X = self._get_X_around_onset_with_context(X, list_of_onset_times, n_frames_per_file)
        # print(X.shape)

        # print('Reshaping data')
        img_rows, img_cols = (X.shape[1], X.shape[2])
        # Theano is 3 times faster with channels_first vs. channels_last on MNIST, so this setting matters.
        # "image_data_format": "channels_first" @ %USERPROFILE%/.keras/keras.json
        if self.image_data_format == 'channels_first':
            X = X.reshape(X.shape[0], 1, img_rows, img_cols)
        else:
            X = X.reshape(X.shape[0], img_rows, img_cols, 1)
        # print(X.shape)

        return X

    def fit_transform(self, data, y=None, **fit_params):
        return self.fit(data, save_data=True).transform(None, load_data=True)

    def _extract_spectrogram_features(self, list_of_samples):
        # print('Creating spectrograms')
        X_parts = [self._extract_spectrogram_features_X(samples)
                   for samples in list_of_samples]
        X = np.concatenate(X_parts)
        n_frames_per_file = [X_part.shape[0] for X_part in X_parts]

        return X, n_frames_per_file

    def _extract_spectrogram_features_X(
            self, samples,
            hop_length=512,

            # A1 (midi 33)
            # gtr E 82.4068892282
            fmin=55.0,

            n_bins=180, bins_per_octave=36, scale=False
    ):
        cqt_spectrogram = cqt(
            samples, sr=self.sample_rate, hop_length=hop_length, fmin=fmin, n_bins=n_bins,
            bins_per_octave=bins_per_octave, scale=scale
        )
        # Convert from complex to real (uses "norm")
        # Transpose (n_bins, n_hops) -> (n_hops, n_bins)
        cqt_spectrogram = np.abs(cqt_spectrogram).T

        return cqt_spectrogram

    def _get_X_around_onset_with_context(self, X, list_of_onset_times, n_frames_per_file,
                                         frames_before=5, frames_after=10, border_value=0.0):
        """X = complete spectrogram"""
        n_samples = X.shape[0]
        assert n_samples == sum(n_frames_per_file)

        # [10 15 5] -> [5 10 15]
        start_index_per_file = np.roll(n_frames_per_file, 1)
        # [5 10 15] -> [0 10 15]
        start_index_per_file[0] = 0
        # [0 10 15] -> [0 10 25]
        start_index_per_file = np.cumsum(start_index_per_file)
        assert len(start_index_per_file) == len(list_of_onset_times)

        start_indices = []
        for start_index, n_frames, onset_times_grouped in zip(
            start_index_per_file, n_frames_per_file, list_of_onset_times
        ):
            for onset_time in onset_times_grouped:
                index = int(onset_time * self.frame_rate_hz)
                assert index < n_frames

                if n_frames - index - 1 < frames_after:
                    warn('Onset is too close to end of file ({} < {}), '
                         'operation will add context from different file!'.format(n_frames - index - 1, frames_after))
                start_indices.append(start_index + index)

        n_onsets = len(start_indices)
        n_bins = X.shape[1]
        X_new = np.empty((n_onsets, frames_before + 1 + frames_after, n_bins))
        for i, start_index in enumerate(start_indices):
            for offset in range(-frames_before, frames_after + 1):
                if start_index + offset > -1 and start_index + offset < n_samples:
                    # X_new 2nd dim: [0, frames_before+1+frames_after[
                    # X 1st dim: [start_index-frames_before, start_index+frames_after+1[
                    X_new[i, offset + frames_before, :] = X[start_index + offset, :]
                else:
                    X_new[i, offset + frames_before].fill(border_value)

        return X_new


# TODO Try alternative using 3 differents CQT spectrograms with 3 parallel CNNs
class CnnCqtPitchDetector(AbstractPitchDetector):
    CONFIG_FILE = 'config.pickle'
    FEATURE_EXTRACTOR_FILE = 'feature_extractor.pickle'
    MODEL_FILE = 'model.json'
    WEIGHTS_FILE = 'weights.hdf5'

    LOSS = 'binary_crossentropy'
    OPTIMIZER = 'adam'
    METRICS = None
    BATCH_SIZE = 256

    def __init__(self,
                 # loaded config, feature extractor and model
                 config=None, feature_extractor=None, model=None,

                 # config params
                 tuning=(64, 59, 55, 50, 45, 40), n_frets=24, proba_threshold=0.5, onset_group_threshold_seconds=0.05,

                 # feature extractor params
                 image_data_format='channels_first',

                 sample_rate=44100,
                 hop_length=512,

                 # A1 (midi 33)
                 # gtr E 82.4068892282
                 fmin=55.0,

                 n_bins=180, bins_per_octave=36, scale=False):
        if config is None:
            super().__init__(tuning, n_frets)
            self.config['proba_threshold'] = proba_threshold
            self.config['onset_group_threshold_seconds'] = onset_group_threshold_seconds
        else:
            self.config = config

        if feature_extractor is None:
            self.feature_extractor = CnnCqtFeatureExtractor(image_data_format=image_data_format,
                                                            sample_rate=sample_rate,
                                                            hop_length=hop_length,
                                                            fmin=fmin,
                                                            n_bins=n_bins,
                                                            bins_per_octave=bins_per_octave,
                                                            scale=scale)
        else:
            self.feature_extractor = feature_extractor

        self.model = model

    @classmethod
    def from_zip(cls, path_to_zip, work_dir='zip_tmp'):
        """Load CnnCqtPitchDetector from a zipfile containing a pickled config dict, a pickled CnnFeatureExtractor, a Keras model JSON file and a Keras weights HDF5 file."""

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

    def save(self, path_to_zip, work_dir='zip_tmp_pitch'):
        """Save this CnnCqtPitchDetector to a zipfile containing a pickled config, a pickled CnnFeatureExtractor,
        a Keras model JSON file and a Keras weights HDF5 file.
        """

        if os.path.exists(path_to_zip):
            path_to_zip_orig = path_to_zip
            path_to_zip = 'CnnCqtPitchDetector_model_' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.zip'
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
        data_train, y_train, _, _ = read_data_y(wav_file_paths_train, truth_dataset_format_tuples_train,
                                                self.feature_extractor.sample_rate, 1,
                                                self.config['min_pitch'], self.config['max_pitch'],
                                                onset_group_threshold_seconds=self.config['onset_group_threshold_seconds'])
        X_train = self.feature_extractor.fit_transform(data_train)
        input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])

        if wav_file_paths_val is not None and truth_dataset_format_tuples_val is not None:
            data_val, y_val, _, _ = read_data_y(wav_file_paths_val, truth_dataset_format_tuples_val,
                                                self.feature_extractor.sample_rate, 1,
                                                self.config['min_pitch'], self.config['max_pitch'],
                                                onset_group_threshold_seconds=self.config['onset_group_threshold_seconds'])
            X_val = self.feature_extractor.transform(data_val)
            validation_data = (X_val, y_val)
        else:
            validation_data = None

        self.model = self._create_model(input_shape,
                                        self.config['max_pitch'] - self.config['min_pitch'] + 1)
        self.model.fit(X_train, y_train,
                       epochs=1000,
                       batch_size=self.BATCH_SIZE,
                       callbacks=[EarlyStopping(monitor='loss', patience=8)], verbose=2,
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
        model.summary()

        return model

    def predict(self, path_to_wav_file, onset_times_seconds, epsilon=1e-7):
        samples = read_samples(path_to_wav_file,
                               self.feature_extractor.sample_rate,
                               1)
        if samples is None:
            return None
        X = self.feature_extractor.transform(([samples], [onset_times_seconds]))

        proba_matrix = self.model.predict(X)
        y = proba_matrix > self.config['proba_threshold']
        y = y.astype(np.int8)

        # Make sure at least one pitch is returned.
        for probas, labels in zip(proba_matrix, y):
            if labels.sum() == 0:
                max_proba = max(probas)
                max_index = np.where(np.logical_and(probas > max_proba - epsilon, probas < max_proba + epsilon))[0][0]
                labels[max_index] = 1

        return y

    def predict_pitches(self, path_to_wav_file, onset_times_seconds):
        return self.multilabel_matrix_to_pitch_sets(self.predict(path_to_wav_file, onset_times_seconds))
