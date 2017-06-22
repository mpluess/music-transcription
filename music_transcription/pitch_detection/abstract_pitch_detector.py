from abc import ABCMeta, abstractmethod
from sklearn.preprocessing import MultiLabelBinarizer


class AbstractPitchDetector(metaclass=ABCMeta):
    def __init__(self, tuning, n_frets):
        """Base constructor

        Parameters
        ----------
        tuning : tuple
            Tuple with length n_strings describing the pitches of the empty strings, in descending order.
        n_frets : int
            Number of frets of the guitar.
        """

        self.config = {
            'tuning': tuning,
            'n_frets': n_frets,
            'min_pitch': min(tuning),
            'max_pitch': max(tuning) + n_frets,
        }

    @abstractmethod
    def predict(self, path_to_wav_file, onset_times_seconds):
        """Predict at least one pitch per onset, return pitches as multilabel matrix.

        Parameters
        ----------
        path_to_wav_file : str
            Path to wave file for which the pitches should be predicted.
        onset_times_seconds : list of float
            List of onsets as floats, describing the time in seconds when each onset happens, sorted by time.

        Returns
        -------
        ndarray
            ndarray of shape (len(onset_times_seconds), n_possible_pitches).
            Each row corresponds to one onset (same order as onset_times_seconds).
            The columns correspond to all possible pitches in the given guitar configuration (tuning, n_frets),
            sorted ascending by midi number.
            Every pitch that is associated with the onset is marked as 1, other pitches with 0.
            At least one pitch per onset has to be picked.
        """

        pass

    @abstractmethod
    def predict_pitches(self, path_to_wav_file, onset_times_seconds):
        """Predict at least one pitch per onset, return pitches as list of sets of pitches.

        Parameters
        ----------
        path_to_wav_file : str
            Path to wave file for which the pitches should be predicted.
        onset_times_seconds : list of float
            List of onsets as floats, describing the time in seconds when each onset happens, sorted by time.

        Returns
        -------
        list of set
            list with len = len(onset_times_seconds) of sets of pitches with 0 < len <= n_strings.
            Each set corresponds to one onset (same order as onset_times_seconds).
        """

        pass

    def pitch_sets_to_multilabel_matrix(self, list_of_pitch_sets):
        label_binarizer = MultiLabelBinarizer(classes=range(self.config['min_pitch'], self.config['max_pitch'] + 1))
        label_binarizer.fit(None)

        return label_binarizer.transform(list_of_pitch_sets)

    def multilabel_matrix_to_pitch_sets(self, y):
        pitches = list(range(self.config['min_pitch'], self.config['max_pitch'] + 1))
        assert len(pitches) == y.shape[1]

        list_of_pitch_sets = []
        for labels in y:
            pitch_set = set()
            for pitch, label in zip(pitches, labels):
                if label == 1:
                    pitch_set.add(pitch)
            list_of_pitch_sets.append(pitch_set)

        return list_of_pitch_sets
