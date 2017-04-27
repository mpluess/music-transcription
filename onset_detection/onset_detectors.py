from keras.models import model_from_json
import os
import numpy as np
import pickle
from python_speech_features import fbank, logfbank
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

from onset_detection.read_data import read_X


class CnnFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, frame_rate_hz=100, sample_rate=44100, subsampling_step=1, image_data_format='channels_first'):
        self.frame_rate_hz = frame_rate_hz
        self.sample_rate = sample_rate
        self.subsampling_step = subsampling_step
        self.image_data_format = image_data_format

        self.standard_scalers_per_channel = None

    def fit(self, wav_file_paths, y=None):
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

    def transform(self, wav_file_paths):
        X_channels = self._read_and_extract(wav_file_paths)
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

        return X

    def _read_and_extract(self, wav_file_paths):
        print('Reading wave files')
        X_parts = []
        for path_to_wav in wav_file_paths:
            X_part, length_seconds = read_X(path_to_wav, self.frame_rate_hz, self.sample_rate, self.subsampling_step)
            if X_part is not None:
                X_parts.append(X_part)

        print('Creating spectrograms')
        return self._extract_spectrogram_features(X_parts)

    def _extract_spectrogram_features(self, X_parts):
        n_frames_list = [None] * len(X_parts)
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
                           in zip(X_parts, n_frames_list)]
            X = np.concatenate([t[0] for t in transformed])
            n_frames_list = [t[1] for t in transformed]
            X_channels.append(X)

        return X_channels

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

    def __init__(self, feature_extractor=None, model=None):
        if feature_extractor is None:
            self.feature_extractor = CnnFeatureExtractor()
        else:
            self.feature_extractor = feature_extractor

        self.model = model

    @classmethod
    def from_model_folder(cls, path_to_model_folder):
        feature_extractor = cls._load_feature_extractor(path_to_model_folder)
        model = cls._load_model(path_to_model_folder)
        return cls(feature_extractor, model)

    @classmethod
    def _load_feature_extractor(cls, path_to_model_folder):
        return pickle.load(open(os.path.join(path_to_model_folder, cls.FEATURE_EXTRACTOR_FILE), 'rb'))

    @classmethod
    def _load_model(cls, path_to_model_folder):
        with open(os.path.join(path_to_model_folder, cls.MODEL_FILE)) as f:
            model = model_from_json(f.read())
        model.load_weights(os.path.join(path_to_model_folder, cls.WEIGHTS_FILE))

        # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(None, None)

        return model

    def fit(self, wav_file_paths, truth_dataset_format_tuples):
        pass

    def predict(self, wav_file_paths):
        X = self.feature_extractor.transform(wav_file_paths)
        y_predicted = self.model.predict_classes(X, batch_size=1024).ravel()

    def save(self, path_to_model_folder):
        pickle.dump(self.feature_extractor, open(os.path.join(path_to_model_folder, self.FEATURE_EXTRACTOR_FILE), 'wb'))
        with open(os.path.join(path_to_model_folder, self.MODEL_FILE), 'w') as f:
            f.write(self.model.to_json())
        self.model.save_weights(os.path.join(path_to_model_folder, self.WEIGHTS_FILE))
