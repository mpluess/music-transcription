import numpy as np
import os
import datetime
import pickle
import shutil
from zipfile import ZipFile
from math import ceil

from keras.callbacks import EarlyStopping
from keras.layers import Activation, Concatenate, Conv1D, Dense, Dropout, Flatten, MaxPooling1D
from keras.models import Input, Model, model_from_json

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

from music_transcription.string_fret_detection.abstract_string_detector import AbstractStringDetector
from music_transcription.string_fret_detection.read_data import read_data_y, read_samples


class CnnStringFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n_strings, n_input_samples):
        self.n_strings = n_strings
        self.n_input_samples = n_input_samples

        self.standard_scaler = None

    def fit(self, X):
        """ fit standard scaler """

        print('Fitting standard scaler')
        self.standard_scaler = StandardScaler()
        self.standard_scaler.fit(np.concatenate(X).reshape((-1,1))) # flatten and reshape to single feature

        return self

    def transform(self, X):
        """ standardize input """

        print('Standardizing samples')
        for i in range(len(X)):
            X[i] = self.standard_scaler.transform(X[i].reshape((-1,1)))

            # add zero padding and reshape to fit input size
            n_zeros_padding = (self.n_input_samples - (len(X[i]) % self.n_input_samples)) % self.n_input_samples
            Z = np.zeros((1, n_zeros_padding)).reshape(-1, 1)
            X[i] = np.vstack((X[i], Z)).reshape(-1, self.n_input_samples, 1)

        return np.concatenate(X)

    def fit_transform(self, X, **fit_params):
        """ fit standard scaler and standardize input """
        return self.fit(X).transform(X)

    def get_labels(self, data):
        samples, onsets, pitches, strings = data
        assert len(samples) == len(onsets)
        assert len(onsets) == len(pitches)
        assert len(pitches) == len(strings)

        label_binarizer = MultiLabelBinarizer(classes=range(1, self.n_strings + 1))
        label_binarizer.fit(None)  # fit needs to be called before transform

        y_list = []
        no_string = np.zeros((1, self.n_strings)).astype('uint8')
        for i in range(len(strings)):
            frame = 0
            cur_y = no_string
            total_frames = ceil(samples[i].shape[0] / 441)
            y = np.empty((total_frames, 6), 'uint8')
            for j in range(len(onsets[i])):
                frame_end_prev = max(0, int(onsets[i][j] * 100) - 3)  # TODO consider offset detection!
                for k in range(frame, frame_end_prev):
                    y[k] = cur_y  # label prev strings
                frame = max(0, int(onsets[i][j] * 100) - 1)  # update frame: label even one frame before onset!
                for k in range(frame_end_prev, frame):
                    y[k] = no_string  # label empty strings between
                cur_y = label_binarizer.transform([strings[i][j]]).astype('uint8')  # update current strings
                # TODO add pitch list?

            # write last note until the end
            for k in range(frame, total_frames):
                y[k] = cur_y  # label prev strings

            y_list.append(y)

        # y = np.array([]).reshape(-1, self.n_strings)
        # no_string = np.zeros((1,self.n_strings)).astype('uint8')
        # for i in range(len(strings)):
        #     frame = 0
        #     cur_y = no_string
        #     for j in range(len(onsets[i])):
        #         frame_end_prev = max(0, int(onsets[i][j]*100)-3)  # TODO consider offset detection!
        #         for k in range(frame, frame_end_prev):
        #             y = np.vstack((y, cur_y))  # label prev strings
        #         frame = max(0, int(onsets[i][j]*100)-1)  # update frame: label even one frame before onset!
        #         for k in range(frame_end_prev, frame):
        #             y = np.vstack((y, no_string))  # label empty strings between
        #         cur_y = label_binarizer.transform([strings[i][j]]).astype('uint8')  # update current strings
        #         # TODO add pitch list?
        #
        #     # write last note until the end
        #     total_frames = ceil(samples[i].shape[0] / 441)
        #     for k in range(frame, total_frames + 1):
        #         y = np.vstack((y, cur_y))  # label prev strings

        return np.concatenate(y_list)


class CnnStringDetector(AbstractStringDetector):
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
                 tuning=(64, 59, 55, 50, 45, 40), proba_threshold=0.5, onset_group_threshold_seconds=0.05,
                 frame_rate_hz=100, sample_rate=44100, subsampling_step=1):
        if config is None:
            super().__init__(tuning)
            self.config['proba_threshold'] = proba_threshold
            self.config['onset_group_threshold_seconds'] = onset_group_threshold_seconds
            self.config['frame_rate_hz'] = frame_rate_hz
            self.config['sample_rate'] = sample_rate
            self.config['subsampling_step'] = subsampling_step
        else:
            self.config = config

        print(config)
        self.n_input_samples = int(self.config['sample_rate'] / self.config['frame_rate_hz'])

        if feature_extractor is None:
            self.feature_extractor = CnnStringFeatureExtractor(self.config['strings'], self.n_input_samples)
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

    def save(self, path_to_zip, work_dir='zip_tmp_pitch'):
        """Save this CnnCqtPitchDetector to a zipfile containing a pickled config, a pickled CnnFeatureExtractor,
        a Keras model JSON file and a Keras weights HDF5 file.
        """

        if os.path.exists(path_to_zip):
            path_to_zip_orig = path_to_zip
            path_to_zip = 'CnnStringDetector_model_' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '.zip'
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
        # _, _ = wav_file_paths_valid, truth_dataset_format_tuples_valid
        data_train, _, _ = read_data_y(wav_file_paths_train, truth_dataset_format_tuples_train,
                                                self.config['sample_rate'], self.config['subsampling_step'],
                                                self.config['strings'], self.config['onset_group_threshold_seconds'])
        y_train = self.feature_extractor.get_labels(data_train)
        X_train = self.feature_extractor.fit_transform(data_train[0])

        if wav_file_paths_val is not None and truth_dataset_format_tuples_val is not None:
            data_val, _, _ = read_data_y(wav_file_paths_val, truth_dataset_format_tuples_val,
                                                self.config['sample_rate'], self.config['subsampling_step'],
                                                self.config['strings'], self.config['onset_group_threshold_seconds'])
            y_val = self.feature_extractor.get_labels(data_val)
            X_val = self.feature_extractor.transform(data_val[0])
            monitor = 'val_loss'
            validation_data = (X_val, y_val)
        else:
            monitor = 'loss'
            validation_data = None

        self.model = self._create_model(self.n_input_samples, self.config['strings'])

        self.model.fit(X_train, y_train, epochs=1000, batch_size=self.BATCH_SIZE,
                       callbacks=[EarlyStopping(monitor=monitor, patience=6)],
                       verbose=2, validation_data=validation_data)

    @classmethod
    def _create_model(cls, n_input_samples, n_output_units):
        """Keras model description

        Multi-label classification with Keras: https://github.com/fchollet/keras/issues/741
        """
        input = Input(shape=(n_input_samples, 1))

        conv = Conv1D(12, 7, padding='valid')(input)
        conv = Activation('relu')(conv)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Conv1D(12, 3, padding='valid')(conv)
        conv = Activation('relu')(conv)
        conv = MaxPooling1D(pool_size=2)(conv)
        conv = Dropout(0.25)(conv)
        conv = Flatten()(conv)

        z = Dense(256)(conv)
        z = Activation('relu')(z)
        z = Dropout(0.5)(z)
        output = Dense(n_output_units, activation='sigmoid')(z)

        model = Model(input, output)
        model.compile(loss=cls.LOSS, optimizer=cls.OPTIMIZER, metrics=cls.METRICS)
        model.summary()

        return model

    def predict(self, path_to_wav_file, onset_times_seconds, pitches, epsilon=1e-7):
        samples = read_samples(path_to_wav_file, self.config['sample_rate'], self.config['subsampling_step'])
        if samples is None:
            return None
        list_of_X = self.feature_extractor.transform([samples])

        proba_matrix = self.model.predict(list_of_X)
        y = proba_matrix > self.config['proba_threshold']
        y = y.astype(np.int8)

        # Make sure at least one pitch is returned.
        # TODO for every pitch exactly one string?
        for probas, labels in zip(proba_matrix, y):
            if labels.sum() == 0:
                max_proba = max(probas)
                max_index = np.where(np.logical_and(probas > max_proba - epsilon, probas < max_proba + epsilon))[0][0]
                labels[max_index] = 1

        return y
