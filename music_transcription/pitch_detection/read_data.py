from sklearn.preprocessing import MultiLabelBinarizer
import soundfile
from warnings import warn
from xml.etree import ElementTree

from music_transcription.read_data import DATASET_CORRECTIONS


def read_data_y(wav_file_paths, truth_dataset_format_tuples,
                sample_rate, subsampling_step,
                min_pitch, max_pitch, onset_group_threshold_seconds=0.05, frame_rate_hz=None):
    """Read samples, onset times and labels.

    Parameters
    ----------
    wav_file_paths : list
        List of WAV file paths.
    truth_dataset_format_tuples : list
        List of tuples (path_to_truth_file, dataset, format)
    sample_rate : int
        Expected sample rate of files
    subsampling_step : int
        If > 1: only take every nth sample.
    min_pitch : int
        Minimum expected pitch
    max_pitch : int
        Maximum expected pitch
    onset_group_threshold_seconds : float
        Onsets <= this threshold will be grouped together
    frame_rate_hz
        Expected frame rate. If set, condition sample_rate % frame_rate_hz != 0 will be asserted.

    Returns
    -------
    tuple
        ((list_of_samples, list_of_onset_times), y, wav_file_paths_valid, truth_dataset_format_tuples_valid)
        list_of_samples: list of samples per valid WAV file with valid truth
        list_of_onset_times: list of list of onset times per valid WAV file with valid truth
        y: label matrix for multilabel classification
        wav_file_paths_valid: list of valid WAV files with valid truth

        All iterables in this tuple have the same length, except y.
        len(y) = sum([len(onset_times) for onset_times in list_of_onset_times])
    """

    list_of_samples = []
    list_of_onset_times = []
    list_of_pitches = []
    wav_file_paths_valid = []
    truth_dataset_format_tuples_valid = []
    for path_to_wav, truth_dataset_format_tuple in zip(wav_file_paths, truth_dataset_format_tuples):
        path_to_xml, dataset, truth_format = truth_dataset_format_tuple
        if truth_format != 'xml':
            raise ValueError('Unsupported format {}'.format(truth_format))
        samples = read_samples(path_to_wav, sample_rate, subsampling_step, frame_rate_hz=frame_rate_hz)
        onset_times_grouped, pitches_grouped = _read_onset_times_pitches(path_to_xml, min_pitch, max_pitch, dataset,
                                                                         onset_group_threshold_seconds)
        # Skip invalid WAV files and files with missing or invalid truth.
        if samples is not None and onset_times_grouped is not None and pitches_grouped is not None:
            list_of_samples.append(samples)
            list_of_onset_times.append(onset_times_grouped)
            list_of_pitches.append(pitches_grouped)
            wav_file_paths_valid.append(path_to_wav)
            truth_dataset_format_tuples_valid.append(truth_dataset_format_tuple)

    label_binarizer = MultiLabelBinarizer(classes=range(min_pitch, max_pitch + 1))
    # No-op since classes are already passed in the constructor
    label_binarizer.fit(None)

    pitch_groups_flat = [pitch_group for pitches_grouped in list_of_pitches for pitch_group in pitches_grouped]
    assert len(pitch_groups_flat) == sum([len(onset_times) for onset_times in list_of_onset_times])
    # List of sets to binary label matrix for multilabel classification
    y = label_binarizer.transform(pitch_groups_flat)

    return (list_of_samples, list_of_onset_times), y, wav_file_paths_valid, truth_dataset_format_tuples_valid


def read_samples(path_to_wav, expected_sample_rate, subsampling_step, frame_rate_hz=None):
    """Read samples of a WAV file. Apply subsampling if subsampling_step > 1.

    Parameters
    ----------
    path_to_wav : str
        Path to WAV file
    expected_sample_rate : int
        Expected sample rate of files
    subsampling_step : int
        If > 1: only take every nth sample.
    frame_rate_hz : int
        Expected frame rate. If set, condition sample_rate % frame_rate_hz != 0 will be asserted.

    Returns
    -------
    samples : ndarray
        1D ndarray containing the samples

    WARNING: Only use this function if you don't have labels for a file. Otherwise always use read_data_y, even
    if you don't need the labels at this moment. read_data_y makes sure files without proper labels are filtered out.
    When using read_samples directly, there's a chance samples and labels will be out of sync.
    """

    samples, sample_rate = soundfile.read(path_to_wav)
    if len(samples.shape) > 1:
        warn('Skipping ' + path_to_wav + ', cannot handle stereo signal.')
        return None
    elif sample_rate != expected_sample_rate:
        warn('Skipping ' + path_to_wav +
             ', sample rate ' + str(sample_rate) + ' != expected sample rate ' + str(expected_sample_rate) + '.')
        return None
    elif frame_rate_hz is not None and sample_rate % frame_rate_hz != 0:
        raise ValueError('Sample rate ' + str(sample_rate) + ' % frame rate ' + str(frame_rate_hz) + ' != 0')

    return samples[::subsampling_step]


def _read_onset_times_pitches(path_to_xml, min_pitch, max_pitch, dataset, onset_group_threshold_seconds):
    """Read truth (onset times and pitches).

    Parameters
    ----------
    path_to_xml : str
        Path to truth XML file
    min_pitch : int
        Minimum expected pitch
    max_pitch : int
        Maximum expected pitch
    dataset : int
        Dataset label of this file.
        Used to adjust the onset time using music_transcription.read_data.DATASET_CORRECTIONS.
    onset_group_threshold_seconds : float
        Consecutive onsets less than onset_group_threshold_seconds apart will be grouped together

    Returns
    -------
    tuple
        (onset_times_grouped, pitches_grouped)
        onset_times_grouped: List of onset times, with consecutive onsets less than onset_group_threshold_seconds apart grouped together.
        pitches_grouped: List of pitch groups = chords.

        All iterables in this tuple have the same length
    """

    tree = ElementTree.parse(path_to_xml)
    root = tree.getroot()
    onset_times = []
    pitches = []
    for root_child in root:
        if root_child.tag == 'transcription':
            for event in root_child:
                if event.tag != 'event':
                    raise ValueError('Unexpected XML element, expected event, got ' + event.tag)
                onset_time = None
                pitch = None
                for event_child in event:
                    if event_child.tag == 'onsetSec':
                        onset_time = float(event_child.text) + DATASET_CORRECTIONS[dataset]
                    elif event_child.tag == 'pitch':
                        pitch = int(event_child.text)
                if onset_time is not None and pitch is not None:
                    if pitch >= min_pitch and pitch <= max_pitch:
                        onset_times.append(onset_time)
                        pitches.append(pitch)
                    else:
                        warn('Skipping {}, pitch {} is out of range.'.format(path_to_xml, pitch))
                        return None, None
                else:
                    raise ValueError('File {} does not contain both onset and pitch information: onset_time={}, pitch={}'.format(path_to_xml, onset_time, pitch))
            break

    # Sort by onset time since this is not always the case in the files.
    onset_pitch_tuples_sorted = sorted(zip(onset_times, pitches), key=lambda t: t[0])
    onset_times = [t[0] for t in onset_pitch_tuples_sorted]
    pitches = [t[1] for t in onset_pitch_tuples_sorted]

    onset_times_grouped, pitches_grouped = _group_onsets(onset_times, pitches, onset_group_threshold_seconds)

    return onset_times_grouped, pitches_grouped


def _group_onsets(onset_times, pitches, onset_group_threshold_seconds, epsilon=1e-6):
    """Group onsets and corresponding pitches in a way that onsets closer than onset_group_threshold_seconds belong to the same group.

    Assumes onset times are sorted (pitch_detection.read_data._read_onset_times_pitches does that).
    """

    onset_times_grouped = []
    pitches_grouped = []
    last_onset = None
    onset_group_start = None
    onset_group_pitches = set()
    for onset_time, pitch in zip(onset_times, pitches):
        if last_onset is not None and onset_time - last_onset > onset_group_threshold_seconds + epsilon:
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

    return onset_times_grouped, pitches_grouped
