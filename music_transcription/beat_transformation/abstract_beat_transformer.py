from abc import ABCMeta, abstractmethod


class AbstractBeatTransformer(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def transform(self, path_to_wav_file, onset_times_seconds, list_of_string_lists, list_of_fret_lists, tempo):
        """ Discretize onsets and transform them together with the tabbed chords to notes and measures.

        Parameters
        ----------
        path_to_wav_file : str
            Path to wave file of recording
        onset_times_seconds : list of float
            List of onset times in seconds
        list_of_string_lists : list of list of int
            List of lists of string numbers,
            with 0 being the string with the highest pitch
            and 5 the one with the lowest pitch.
        list_of_fret_lists : list of list of int
            List of lists of fret numbers,
            with 0 zero being the empty string.
        tempo : int
            Tempo of recording in BPM

        Returns
        -------
        list
            List of measures containing the onsets and tabbed chords as notes.
            The format is as expected by the parameter beats of the function
            music_transcription.fileformat.guitar_pro.gp5_writer.write_gp5:
            [  # measures
                [  # measure, tracks
                    (  # track, 2 voices
                        [  # voice 1, beats (onsets with corresponding notes) go here

                        ],
                        [] # voice 2 is empty
                    ),
                ],
            ]

            See also scripts.fileformat.test_write_gp5.py.
        """

        pass
