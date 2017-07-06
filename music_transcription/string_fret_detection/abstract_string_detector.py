import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from abc import ABCMeta, abstractmethod


class AbstractStringDetector(metaclass=ABCMeta):
    def __init__(self, tuning):
        """Base constructor

        Parameters
        ----------
        tuning : tuple
            Tuple describing the pitches of the empty strings, in descending order. -1 or 0 may indicate unused string
        """

        self.config = {
            'tuning': tuning,
            'strings': (np.array(tuning) > 0).astype('int').sum(),
        }

    @abstractmethod
    def predict(self, path_to_wav_file, onset_times_seconds, pitches):
        """Predict a string per onset and pitch, return strings as multilabel matrix.

        Parameters
        ----------
        path_to_wav_file : str
            Path to wave file for which the pitches should be predicted.
        onset_times_seconds : list of float
            List of onsets as floats, describing the time in seconds when each onset happens, sorted by time.
        pitches
            List of pitches per onset.

        Returns
        -------
        ndarray
            ndarray of shape (len(onset_times_seconds), n_strings).
            Each row corresponds to one onset (same order as onset_times_seconds).
            The columns correspond to all possible pitches in the given guitar configuration (tuning, n_frets),
            sorted ascending by midi number.
            Every pitch that is associated with the onset is marked as 1, other pitches with 0.
            At least one pitch per onset has to be picked.
        """

        pass

    def string_sets_to_multilabel_matrix(self, list_of_string_sets):
        label_binarizer = MultiLabelBinarizer(classes=range(1, self.config['strings'] + 1))
        label_binarizer.fit(None)

        return label_binarizer.transform(list_of_string_sets)

    def multilabel_matrix_to_string_sets(self, y):
        strings = list(range(1, self.config['strings'] + 1))
        assert len(strings) == y.shape[1]

        list_of_string_sets = []
        for labels in y:
            string_set = set()
            for string, label in zip(strings, labels):
                if label == 1:
                    string_set.add(string)
            list_of_string_sets.append(string_set)

        return list_of_string_sets
