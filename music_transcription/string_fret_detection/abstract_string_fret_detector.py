from abc import ABCMeta, abstractmethod


class AbstractStringFretDetector(metaclass=ABCMeta):
    def __init__(self, tuning, n_frets):
        """Base constructor

        Parameters
        ----------
        tuning : tuple
            Tuple with length n_strings describing the pitches of the empty strings, in descending order.
        n_frets : int
            Number of frets of the guitar.
        """

        self.tuning = tuning
        self.n_frets = n_frets
        self.min_pitch = min(tuning)
        self.max_pitch = max(tuning) + n_frets

    @abstractmethod
    def predict_strings_and_frets(self, path_to_wav_file, onset_times_seconds, list_of_pitch_sets):
        """Predict string and fret per pitch.

        The returned string and fret lists have exactly the same shapes.
        String and fret information at the same indices belong together.
        
        The onset list and pitch list also have the same length and belong together at the same indices.

        Parameters
        ----------
        path_to_wav_file : str
            Path to wave file for which the strings and frets should be predicted.
        onset_times_seconds : list of float
            List of onsets as floats, describing the time in seconds when each onset happens, sorted by time.
        list_of_pitch_sets : list of list of int
            List (len = len(onset_times_seconds) = len(list_of_pitch_sets))
            of lists (len = length of the corresponding pitch set) of string numbers,
            with 0 being the string with the highest pitch
            and 5 the one with the lowest pitch.

        Returns
        -------
        list_of_string_lists : list of list of int
            List (len = len(onset_times_seconds) = len(list_of_pitch_sets))
            of lists (len = length of the corresponding pitch set) of string numbers,
            with 0 being the string with the highest pitch
            and 5 the one with the lowest pitch.
        list_of_fret_lists : list of list of int
            List (len = len(onset_times_seconds) = len(list_of_pitch_sets))
            of lists (len = length of the corresponding pitch set) of fret numbers,
            with 0 zero being the empty string.
        """

        pass
