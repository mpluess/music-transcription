from collections import defaultdict
import datetime
from keras.callbacks import EarlyStopping
from keras.layers import Activation, Concatenate, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Input, Model, model_from_json
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
from music_transcription.string_fret_detection.plausibility import get_all_fret_possibilities


class CnnCqtFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract Constant-Q spectrogram excerpts around each onset."""

    def __init__(self, image_data_format, sample_rate, cqt_configs, n_frames_before, n_frames_after):
        self.image_data_format = image_data_format
        self.sample_rate = sample_rate
        self.cqt_configs = cqt_configs
        self.n_frames_before = n_frames_before
        self.n_frames_after = n_frames_after

        self.standard_scalers_per_X = None

    def fit_transform(self, data, y=None, **fit_params):
        """Fit and transform in one go. Feature extraction is only done once.

        Parameters
        ----------
        data : tuple
            (list_of_samples, list_of_onset_times)
        y
            Not used
        fit_params
            Not used
        Returns
        -------
        tuple
            (list_of_X, sample_file_indexes)
            list_of_X
                List of feature matrices X, one per Constant-Q spectrogram configuration -> len(list_of_X) = len(self.cqt_configs)
                channels_first: X.shape = (n_samples, 1, n_frames_before + 1 + n_frames_after, n_bins)
                channels_last: X.shape = (n_samples, n_frames_before + 1 + n_frames_after, n_bins, 1)
            sample_file_indexes : list
                len(sample_file_indexes) = n_samples
                Information about which file each sample comes from, in the form of an index
                to the list_of_samples / list_of_onset_times lists.
        """

        return self.fit(data, save_data=True).transform(load_data=True, verbose=True)

    def fit(self, data, y=None, save_data=False):
        """Fit standard scalers for each spectrogram and bin.

        Parameters
        ----------
        data : tuple
            (list_of_samples, list_of_onset_times)
        y
            Not used
        save_data : bool
            If true, saves list_of_X, list_of_n_frames_per_file, cqt_configs, list_of_onset_times to the object
            so they can be reused in the transform method.

        Returns
        -------
        self : CnnCqtFeatureExtractor
        """

        list_of_samples, list_of_onset_times = data

        print('Creating spectrograms')
        list_of_X, list_of_n_frames_per_file, cqt_configs = self._extract_spectrogram_features(list_of_samples)
        if save_data:
            self._list_of_X = list_of_X
            self._list_of_n_frames_per_file = list_of_n_frames_per_file
            self._cqt_configs = cqt_configs
            self._list_of_onset_times = list_of_onset_times

        # TODO standardize only for regions with onset?
        print('Fitting standard scalers for each spectrogram and bin')
        self.standard_scalers_per_X = []
        for X in list_of_X:
            standard_scalers = []
            for j in range(X.shape[1]):
                standard_scaler = StandardScaler()
                standard_scaler.fit(X[:, j:j + 1])
                standard_scalers.append(standard_scaler)
            self.standard_scalers_per_X.append(standard_scalers)

        return self

    def transform(self, data=None, load_data=False, verbose=False):
        """

        Parameters
        ----------
        data : tuple
            (list_of_samples, list_of_onset_times)
        load_data : bool
            If true, retrieve saved versions of list_of_X, list_of_n_frames_per_file, cqt_configs, list_of_onset_times
            rather than extracting them again.
        verbose : bool
            If true, print debug information, mostly about the shapes of tensors.

        Returns
        -------
        tuple
            (list_of_X, sample_file_indexes)
            list_of_X : list
                List of feature tensors X, one per Constant-Q spectrogram configuration -> len(list_of_X) = len(self.cqt_configs)
                channels_first: X.shape = (n_samples, 1, n_frames_before + 1 + n_frames_after, n_bins)
                channels_last: X.shape = (n_samples, n_frames_before + 1 + n_frames_after, n_bins, 1)
            sample_file_indexes : list
                len(sample_file_indexes) = n_samples
                Information about which file each sample comes from, in the form of an index
                to the list_of_samples / list_of_onset_times lists.
        """

        assert data is not None or load_data is True
        if load_data:
            list_of_X = self._list_of_X
            list_of_n_frames_per_file = self._list_of_n_frames_per_file
            cqt_configs = self._cqt_configs
            list_of_onset_times = self._list_of_onset_times
            self._list_of_X = None
            self._list_of_n_frames_per_file = None
            self._cqt_configs = None
            self._list_of_onset_times = None
        else:
            list_of_samples, list_of_onset_times = data
            if verbose:
                print('Creating spectrograms')
            list_of_X, list_of_n_frames_per_file, cqt_configs = self._extract_spectrogram_features(list_of_samples)
        if verbose:
            for X in list_of_X:
                print(X.shape)
                print(X.mean())
                print(X.std())

        if verbose:
            print('Standardizing for each spectrogram and bin')
        for X, standard_scalers in zip(list_of_X, self.standard_scalers_per_X):
            for j, ss in enumerate(standard_scalers):
                X[:, j:j + 1] = ss.transform(X[:, j:j + 1])
        if verbose:
            for X in list_of_X:
                print(X.mean())
                print(X.std())

        sample_file_indexes = None
        for i, (n_frames_per_file, cqt_config) in enumerate(zip(list_of_n_frames_per_file, cqt_configs)):
            list_of_X[i], sample_file_indexes_i = self._get_X_around_onset_with_context(
                list_of_X[i], list_of_onset_times, n_frames_per_file, self.sample_rate / cqt_config['hop_length']
            )
            assert len(list_of_X[i]) == len(sample_file_indexes_i)
            if sample_file_indexes is None:
                sample_file_indexes = sample_file_indexes_i
            else:
                assert sample_file_indexes == sample_file_indexes_i
            if verbose:
                print(list_of_X[i].shape)

        if verbose:
            print('Reshaping data')
        for i in range(len(list_of_X)):
            # Theano is 3 times faster with channels_first vs. channels_last on MNIST, so this setting matters.
            # "image_data_format": "channels_first" @ %USERPROFILE%/.keras/keras.json
            if self.image_data_format == 'channels_first':
                list_of_X[i] = list_of_X[i].reshape(
                    list_of_X[i].shape[0], 1, list_of_X[i].shape[1], list_of_X[i].shape[2]
                )
            else:
                list_of_X[i] = list_of_X[i].reshape(
                    list_of_X[i].shape[0], list_of_X[i].shape[1], list_of_X[i].shape[2], 1
                )
            if verbose:
                print(list_of_X[i].shape)

        return list_of_X, sample_file_indexes

    def _extract_spectrogram_features(self, list_of_samples):
        """Extract Constant-Q spectrograms.

        Parameters
        ----------
        list_of_samples : list of ndarray
            List of 1D ndarrays of samples

        Returns
        -------
        tuple
            (list_of_spectrograms, list_of_n_frames_per_file, cqt_configs)
            list_of_spectrograms : list
                List of spectrograms, one per Constant-Q spectrogram configuration -> len(list_of_spectrograms) = len(self.cqt_configs)
                spectrogram.shape = (n_frames_all_files, n_bins)
            list_of_n_frames_per_file : list of list
                List of lists of number of frames per WAV file, one per configuration
            cqt_configs : list of dict
                Constant-Q configurations used for the spectrograms
        """

        list_of_spectrograms = []
        list_of_n_frames_per_file = []
        for cqt_config in self.cqt_configs:
            X_parts = [self._extract_spectrogram_features_X(samples,
                                                            hop_length=cqt_config['hop_length'],
                                                            fmin=cqt_config['fmin'],
                                                            n_bins=cqt_config['n_bins'],
                                                            bins_per_octave=cqt_config['bins_per_octave'],
                                                            scale=cqt_config['scale'])
                       for samples in list_of_samples]
            list_of_spectrograms.append(np.concatenate(X_parts))
            list_of_n_frames_per_file.append([X_part.shape[0] for X_part in X_parts])

        return list_of_spectrograms, list_of_n_frames_per_file, self.cqt_configs

    def _extract_spectrogram_features_X(
            self, samples,
            hop_length, fmin, n_bins, bins_per_octave, scale
    ):
        """Extract Constant-Q spectrogram.

        Parameters
        ----------
        samples : ndarray
            1D ndarray of samples
        hop_length : int
            Hop length for Constant-Q transformation
        fmin : float
            Minimum frequency
        n_bins : int
            Total number of bins
        bins_per_octave : int
            Bins per octave. Determines the maximum frequency together with n_bins.
        scale : bool
            Whether or not to scale / normalize the spectrogram.

        See http://librosa.github.io/librosa/generated/librosa.core.cqt.html
        for more details on the CQT-specific parameters.

        Returns
        -------
        cqt_spectrogram : ndarray
            Constant-Q spectrogram with shape (n_frames, n_bins)
        """

        cqt_spectrogram = cqt(
            samples, sr=self.sample_rate, hop_length=hop_length, fmin=fmin, n_bins=n_bins,
            bins_per_octave=bins_per_octave, scale=scale
        )
        # Convert from complex to real (uses "norm")
        # Transpose (n_bins, n_frames) -> (n_frames, n_bins)
        cqt_spectrogram = np.abs(cqt_spectrogram).T

        return cqt_spectrogram

    def _get_X_around_onset_with_context(self, X, list_of_onset_times, n_frames_per_file, frame_rate_hz,
                                         border_value=0.0):
        """Limit spectrogram to the frames around each onset.

        Parameters
        ----------
        X : ndarray
            X.shape = (n_frames_all_files, n_bins)
        list_of_onset_times : list of list
            List of onset times per file.
        n_frames_per_file : list
            List of number of frames per WAV file
        frame_rate_hz : float
            Frame rate in Hz
        border_value
            Fill matrix with this value in edge cases (start / end of spectrogram).

        Returns
        -------
        tuple
            (X_new, sample_file_indexes)
            X_new : ndarray
                X_new.shape = (n_onsets, n_frames_before + 1 + n_frames_after, n_bins)
            sample_file_indexes : list
                len(sample_file_indexes) = n_samples
                Information about which file each sample comes from, in the form of an index
                to the list_of_onset_times list.
        """

        n_frames_all_files = X.shape[0]
        assert n_frames_all_files == sum(n_frames_per_file)

        # [10 15 5] -> [5 10 15]
        start_index_per_file = np.roll(n_frames_per_file, 1)
        # [5 10 15] -> [0 10 15]
        start_index_per_file[0] = 0
        # [0 10 15] -> [0 10 25]
        start_index_per_file = np.cumsum(start_index_per_file)
        assert len(start_index_per_file) == len(list_of_onset_times)

        start_indices = []
        sample_file_indexes = []
        for i, (start_index, n_frames, onset_times_grouped) in enumerate(zip(
            start_index_per_file, n_frames_per_file, list_of_onset_times
        )):
            for onset_time in onset_times_grouped:
                index = int(onset_time * frame_rate_hz)
                assert index < n_frames

                if n_frames - index - 1 < self.n_frames_after:
                    warn('Onset is too close to end of file ({} < {}), '
                         'operation will add context from different file!'.format(n_frames - index - 1, self.n_frames_after))
                start_indices.append(start_index + index)
                sample_file_indexes.append(i)

        n_onsets = len(start_indices)
        n_bins = X.shape[1]
        X_new = np.empty((n_onsets, self.n_frames_before + 1 + self.n_frames_after, n_bins))
        for i, start_index in enumerate(start_indices):
            for offset in range(-self.n_frames_before, self.n_frames_after + 1):
                if start_index + offset > -1 and start_index + offset < n_frames_all_files:
                    # X_new 2nd dim: [0, frames_before+1+frames_after[
                    # X 1st dim: [start_index-frames_before, start_index+frames_after+1[
                    X_new[i, offset + self.n_frames_before, :] = X[start_index + offset, :]
                else:
                    X_new[i, offset + self.n_frames_before].fill(border_value)

        return X_new, sample_file_indexes


class CnnCqtPitchDetector(AbstractPitchDetector):
    CONFIG_FILE = 'config.pickle'
    FEATURE_EXTRACTOR_FILE = 'feature_extractor.pickle'
    MODEL_FILE = 'model.json'
    WEIGHTS_FILE = 'weights.hdf5'

    LOSS = 'binary_crossentropy'
    OPTIMIZER = 'adam'
    METRICS = None
    BATCH_SIZE = 256

    # A1 (midi 33)
    # gtr E 82.4068892282
    # fmin = 55.0
    DEFAULT_CQT_CONFIGS = [
        {
            'hop_length': 512,
            'fmin': 55.0,
            'n_bins': 180,
            'bins_per_octave': 36,
            'scale': False,
        },
    ]
    # CQT_CONFIGS_3_SPECTROGRAMS = [
    #     {
    #         'hop_length': 256,
    #         'fmin': 55.0,
    #         'n_bins': 60,
    #         'bins_per_octave': 12,
    #         'scale': False,
    #     },
    #     {
    #         'hop_length': 512,
    #         'fmin': 55.0,
    #         'n_bins': 180,
    #         'bins_per_octave': 36,
    #         'scale': False,
    #     },
    #     {
    #         'hop_length': 1024,
    #         'fmin': 55.0,
    #         'n_bins': 300,
    #         'bins_per_octave': 60,
    #         'scale': False,
    #     },
    # ]

    # Class distribution taken from analyze_pitches.py, overall for ds 1, 2, 3, 9, 10, 11, 8.7.17 16:05
    # Multiplications: adjustments for underrepresented pitches.
    # Didn't improve CV score though, no effect on the "bad" fold 2.
    # class_distribution_for_weights={
    #     0: 3.2,
    #     1: 1.4 * 3,
    #     2: 1.0 * 4,
    #     3: 4.25,
    #     4: 2.25 * 2,
    #     5: 11.6,
    #     6: 3.45 * 3,
    #     7: 8.7,
    #     8: 19.45,
    #     9: 7.85 * 2,
    #     10: 17.45,
    #     11: 5.45 * 3,
    #     12: 20.2,
    #     13: 9.4 * 2,
    #     14: 16.55,
    #     15: 16.95,
    #     16: 13.15 * 1.5,
    #     17: 18.6,
    #     18: 7.0 * 2,
    #     19: 22.5,
    #     20: 12.55 * 1.5,
    #     21: 18.15,
    #     22: 17.95,
    #     23: 5.2 * 3,
    #     24: 18.65,
    #     25: 6.0 * 3,
    #     26: 7.05 * 2,
    #     27: 10.5,
    #     28: 3.3 * 3,
    #     29: 12.45,
    #     30: 2.8 * 2,
    #     31: 3.25 * 2,
    #     32: 3.55 * 2,
    #     33: 2.25 * 2,
    #     34: 3.95,
    #     35: 1.05 * 3,
    #     36: 1.05 * 3,
    #     37: 3.3,
    #     38: 3.95,
    #     39: 3.35,
    #     40: 2.15,
    #     41: 2.15,
    #     42: 2.15,
    #     43: 2.45,
    #     44: 1.3,
    #     45: 1.15,
    #     46: 1.15,
    #     47: 1.15,
    #     48: 1.3,
    # }

    def __init__(self,
                 # loaded config, feature extractor and model
                 config=None, feature_extractor=None, model=None,

                 # config params
                 tuning=(64, 59, 55, 50, 45, 40), n_frets=24, proba_threshold=0.5, onset_group_threshold_seconds=0.05,
                 subsampling_step=1, sample_weights=None, class_distribution_for_weights=None,

                 # feature extractor params
                 image_data_format='channels_first', sample_rate=44100, cqt_configs=None,
                 n_frames_before=15, n_frames_after=20):
        """

        Parameters
        ----------
        config : dict
            CnnOnsetDetector configuration (use this when loading an existing model)
        feature_extractor : CnnFeatureExtractor
            Feature extractor object (use this when loading an existing model)
        model
            Keras model (use this when loading an existing model)
        tuning : tuple
            Tuple with len(tuning) = number of strings, with each entry being a midi pitch, in descending order from left to right.
        n_frets : int
            Number of frets
        proba_threshold : float
            Probability threshold: all probabilities above this will be classified as onsets.
            Can be used to fine-tune the model towards precision or recall.
        onset_group_threshold_seconds : float
            Consecutive onsets less than onset_group_threshold_seconds apart will be grouped together.
        subsampling_step : int
            If > 1: only take every nth sample.
        sample_weights : str
            If 'balanced': Assign sample weights inversely proportional to the number of samples in a dataset.
            Rationale: boost the samples in the small datasets so all datasets have the same impact.
        class_distribution_for_weights : dict
            Desired class distribution
            key = class index, value = how frequent is a class compared to the rarest class which has value = 1.0
            If set, class weights will be adjusted to mimic this distribution.
        image_data_format : str
            One of 'channels_first' (for Theano backend), 'channels_last' (Tensorflow backend).
        sample_rate : int
            Sample rate in Hz
        cqt_configs : list of dict
            Constant-Q configurations used for the spectrograms
            See DEFAULT_CQT_CONFIGS, CQT_CONFIGS_3_SPECTROGRAMS for an example.
        n_frames_before : int
            This amount of frames before the onset will be included in an onset's spectrogram as context.
        n_frames_after : int
            This amount of frames after the onset will be included in an onset's spectrogram as context.
        """

        if config is None:
            super().__init__(tuning, n_frets)
            self.config['proba_threshold'] = proba_threshold
            self.config['onset_group_threshold_seconds'] = onset_group_threshold_seconds
            self.config['subsampling_step'] = subsampling_step

            assert sample_weights is None or sample_weights == 'balanced'
            self.config['sample_weights'] = sample_weights
            self.config['class_distribution_for_weights'] = class_distribution_for_weights
        else:
            self.config = config

        if feature_extractor is None:
            if cqt_configs is None:
                cqt_configs = CnnCqtPitchDetector.DEFAULT_CQT_CONFIGS
            self.feature_extractor = CnnCqtFeatureExtractor(image_data_format, sample_rate, cqt_configs,
                                                            n_frames_before, n_frames_after)
        else:
            self.feature_extractor = feature_extractor

        self.model = model

    @classmethod
    def from_zip(cls, path_to_zip, work_dir='zip_tmp'):
        """Load CnnCqtPitchDetector from a zipfile containing a pickled config dict, a pickled CnnCqtFeatureExtractor,
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
        """Save this CnnCqtPitchDetector to a zipfile containing a pickled config, a pickled CnnCqtFeatureExtractor,
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
        """Fit Keras model using spectrograms from wav_file_paths_train and labels from truth_dataset_format_tuples_train.

        Parameters
        ----------
        wav_file_paths_train : list
            List of WAV file paths to train the model with.
        truth_dataset_format_tuples_train : list of tuple
            List of tuples (path_to_truth_file, dataset, format) for the labels.
        wav_file_paths_val : list
            List of WAV file paths for the validation set. Only used to monitor the validation loss during training.
        truth_dataset_format_tuples_val : list of tuple
            List of tuples (path_to_truth_file, dataset, format) for the validation set labels.
        """

        data_train, y_train, wav_file_paths_train_valid, truth_dataset_format_tuples_train_valid = read_data_y(
            wav_file_paths_train, truth_dataset_format_tuples_train,
            self.feature_extractor.sample_rate, self.config['subsampling_step'],
            self.config['min_pitch'], self.config['max_pitch'],
            onset_group_threshold_seconds=self.config['onset_group_threshold_seconds']
        )
        list_of_X_train, sample_file_indexes_train = self.feature_extractor.fit_transform(data_train)

        if wav_file_paths_val is not None and truth_dataset_format_tuples_val is not None:
            data_val, y_val, wav_file_paths_val_valid, truth_dataset_format_tuples_val_valid = read_data_y(
                wav_file_paths_val, truth_dataset_format_tuples_val,
                self.feature_extractor.sample_rate, self.config['subsampling_step'],
                self.config['min_pitch'], self.config['max_pitch'],
                onset_group_threshold_seconds=self.config['onset_group_threshold_seconds']
            )
            list_of_X_val, sample_file_indexes_val = self.feature_extractor.transform(data_val, verbose=True)
            monitor = 'val_loss'
            if self.config['sample_weights'] == 'balanced':
                validation_data = (list_of_X_val, y_val, self._get_sample_weights(sample_file_indexes_val,
                                                                                  truth_dataset_format_tuples_val_valid))
            else:
                validation_data = (list_of_X_val, y_val)
        else:
            monitor = 'loss'
            validation_data = None

        self.model = self._create_model(list_of_X_train,
                                        self.config['max_pitch'] - self.config['min_pitch'] + 1)

        if self.config['sample_weights'] == 'balanced':
            sample_weights = self._get_sample_weights(sample_file_indexes_train,
                                                      truth_dataset_format_tuples_train_valid)
        else:
            sample_weights = None

        if self.config['class_distribution_for_weights'] is not None:
            print(y_train.shape)
            class_distribution = {j: y_train[:, j].sum() for j in range(y_train.shape[1])}
            print(class_distribution)
            assert class_distribution.keys() == self.config['class_distribution_for_weights'].keys()

            min_value = min(class_distribution.values())
            class_distribution_normalized = {k: v / min_value for k, v in class_distribution.items()}
            print(class_distribution_normalized)

            # Boost classes which are underrepresented and vice versa
            class_weights = {
                k: value_should / class_distribution_normalized[k]
                for k, value_should in self.config['class_distribution_for_weights'].items()
            }
            print(class_weights)
        else:
            class_weights = None
        self.model.fit(list_of_X_train, y_train,
                       epochs=1000,
                       batch_size=self.BATCH_SIZE,
                       sample_weight=sample_weights,
                       class_weight=class_weights,
                       # Optimize weights till the loss has converged
                       callbacks=[EarlyStopping(monitor='loss', patience=6)], verbose=2,
                       validation_data=validation_data)

    @staticmethod
    def _get_sample_weights(sample_file_indexes, truth_dataset_format_tuples_valid):
        """Assign sample weights inversely proportional to the number of samples in a dataset.

        Rationale: boost the samples in the small datasets so all datasets have the same impact.
        """

        def transform_dataset(dataset):
            """Merge datasets 1-3"""

            if dataset in {1, 2, 3}:
                return 123
            else:
                return dataset

        assert max(sample_file_indexes) == len(truth_dataset_format_tuples_valid) - 1

        dataset_counts = defaultdict(int)
        for i in sample_file_indexes:
            _, dataset, _ = truth_dataset_format_tuples_valid[i]
            dataset_counts[transform_dataset(dataset)] += 1
        print('dataset_counts={}'.format(dataset_counts))

        # Weights = inversely proportional to number of samples in dataset.
        # Multiply by the min count so the smallest dataset has weight 1.
        dataset_weights = {dataset: 1/count*min(dataset_counts.values()) for dataset, count in dataset_counts.items()}
        print('dataset_weights={}'.format(dataset_weights))
        sample_weights = np.array([dataset_weights[transform_dataset(truth_dataset_format_tuples_valid[i][1])]
                                   for i in sample_file_indexes])

        return sample_weights

    @classmethod
    def _create_model(cls, list_of_X, n_output_units):
        """Keras model description

        Multi-label classification with Keras: https://github.com/fchollet/keras/issues/741
        """

        inputs = []
        conv_blocks = []
        # One input with two convolutional blocks per spectrogram
        for X in list_of_X:
            spectrogram = Input(shape=X.shape[1:])
            inputs.append(spectrogram)

            # 10 small filters finding local patterns applicable to different pitches
            conv = Conv2D(10, (10, 3), padding='valid')(spectrogram)
            conv = Activation('relu')(conv)
            conv = MaxPooling2D(pool_size=(6, 3))(conv)
            conv = Dropout(0.15)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)

            # 512 filters spanning the whole pitch bandwidth of the guitar
            conv = Conv2D(512, (10, 180), strides=(5, 1), padding='valid')(spectrogram)
            conv = Activation('relu')(conv)
            conv = MaxPooling2D(pool_size=(2, 1))(conv)
            conv = Dropout(0.25)(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)

        # Concatenate convolutional blocks and feed them to a feed forward NN with one hidden layer.
        z = Concatenate()(conv_blocks)
        z = Dense(256)(z)
        z = Activation('relu')(z)
        z = Dropout(0.3)(z)
        output = Dense(n_output_units, activation='sigmoid')(z)

        model = Model(inputs, output)
        model.compile(loss=cls.LOSS, optimizer=cls.OPTIMIZER, metrics=cls.METRICS)
        model.summary()

        return model

    def predict(self, path_to_wav_file, onset_times_seconds, epsilon=1e-7):
        """Predict pitches at all onsets of this WAV file. Return them as binary multilabel classification matrix."""

        samples = read_samples(path_to_wav_file,
                               self.feature_extractor.sample_rate,
                               self.config['subsampling_step'])
        if samples is None:
            return None
        list_of_X, _ = self.feature_extractor.transform(([samples], [onset_times_seconds]))

        proba_matrix = self.model.predict(list_of_X)
        y = proba_matrix > self.config['proba_threshold']
        y = y.astype(np.int8)

        # Make sure at least one pitch is returned.
        for probas, labels in zip(proba_matrix, y):
            if labels.sum() == 0:
                max_proba = max(probas)
                max_index = np.where(np.logical_and(probas > max_proba - epsilon, probas < max_proba + epsilon))[0][0]
                labels[max_index] = 1

        # Make sure all chords are actually playable. If not, remove pitches till they are.
        for probas, labels in zip(proba_matrix, y):
            pitch_set = self.multilabel_matrix_to_pitch_sets(labels.reshape(1, -1))
            while len(get_all_fret_possibilities(pitch_set[0], tuning=self.config['tuning'], n_frets=self.config['n_frets'])) == 0:
                # Create (proba, index) tuples of all pitches with label 1.
                # Get the index of the one with minimum proba.
                # Remove this pitch from the chord.
                min_proba, min_index = min(zip(np.extract(labels, probas), labels.nonzero()[0]), key=lambda t: t[0])
                labels[min_index] = 0
                warn('no plausible transcription for pitches [{}], discarding pitch: {}'.format(
                    pitch_set, min_index + self.config['min_pitch']
                ))

        return y

    def predict_pitches(self, path_to_wav_file, onset_times_seconds):
        """Predict pitches at all onsets of this WAV file. Return them as a list of pitch sets / chords."""

        return self.multilabel_matrix_to_pitch_sets(self.predict(path_to_wav_file, onset_times_seconds))
