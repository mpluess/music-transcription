import numpy as np
import soundfile
from xml.etree import ElementTree
from warnings import warn

from music_transcription.read_data import DATASET_CORRECTIONS


def read_X_y(path_to_wav, frame_rate_hz, expected_sample_rate, subsampling_step,
             path_to_truth, truth_format, dataset):
    """Read samples and labels of a wave file.

    Returns a tuple (samples, y_part, y_actual_onset_only_part)
    """

    samples, length_seconds = read_X(path_to_wav, frame_rate_hz, expected_sample_rate, subsampling_step)
    if samples is not None:
        y_part, y_actual_onset_only_part = read_y(truth_format, path_to_truth, length_seconds, frame_rate_hz, dataset)
        samples_per_frame = expected_sample_rate // frame_rate_hz // subsampling_step
        if samples.shape[0] // samples_per_frame != y_part.shape[0]:
            raise ValueError('samples vs. y_part shape mismatch: ' + str(samples.shape[0] // samples_per_frame) + ' != ' + str(y_part.shape[0]))
        return samples, y_part, y_actual_onset_only_part
    else:
        return None, None, None


def read_X(path_to_wav, frame_rate_hz, expected_sample_rate, subsampling_step):
    """Read samples of a wave file. Returns a tuple (sample_np_array, length_seconds).

    sample_np_array.shape = (n_samples,)
    """

    # scipy.io.wavfile is not able to read 24-bit data, hence the need to use this alternative library
    samples, sample_rate = soundfile.read(path_to_wav)
    if len(samples.shape) > 1:
        warn('Skipping ' + path_to_wav + ', cannot handle stereo signal.')
        return None, -1
    elif sample_rate != expected_sample_rate:
        warn('Skipping ' + path_to_wav +
             ', sample rate ' + str(sample_rate) + ' != expected sample rate ' + str(expected_sample_rate) + '.')
        return None, -1

    if sample_rate % frame_rate_hz != 0:
        raise ValueError('Sample rate ' + str(sample_rate) + ' % frame rate ' + str(frame_rate_hz) + ' != 0')
    samples_per_frame = int(sample_rate / frame_rate_hz)
    offset = 0
    samples_subsampled_cut = []
    # Cut off last samples
    while offset <= len(samples) - samples_per_frame:
        samples_subsampled_cut.extend(samples[offset:offset + samples_per_frame:subsampling_step])
        offset += samples_per_frame

    samples_subsampled_cut = np.array(samples_subsampled_cut)
    return samples_subsampled_cut, offset / sample_rate


def read_y(truth_format, path_to_truth, length_seconds, frame_rate_hz, dataset):
    """Read labels of a wave file. Returns a tuple (y_part, y_actual_onset_only_part)."""

    if truth_format == 'xml':
        y_part, y_actual_onset_only_part = read_y_xml(path_to_truth, length_seconds, frame_rate_hz, dataset)
    elif truth_format == 'csv':
        y_part, y_actual_onset_only_part = read_y_csv(path_to_truth, length_seconds, frame_rate_hz, dataset)
    else:
        raise ValueError('Unknown truth format')

    return y_part, y_actual_onset_only_part


def read_y_xml(path_to_xml, length_seconds, frame_rate_hz, dataset):
    """Read labels of a wave file (xml format, datasets 1, 2, 3). Returns a tuple (y_part, y_actual_onset_only_part)."""

    tree = ElementTree.parse(path_to_xml)
    root = tree.getroot()
    y = _init_y(length_seconds, frame_rate_hz)
    y_actual_onset_only = _init_y(length_seconds, frame_rate_hz)
    for root_child in root:
        if root_child.tag == 'transcription':
            for event in root_child:
                if event.tag != 'event':
                    raise ValueError('Unexpected XML element, expected event, got ' + event.tag)
                for event_child in event:
                    if event_child.tag == 'onsetSec':
                        onset_time = float(event_child.text)
                        index = _onset_index(onset_time, frame_rate_hz)
                        # _set_onset_label_orig_with_neighbors(y, y_actual_onset_only, index)
                        _set_onset_label_adjusted_with_neighbors(y, y_actual_onset_only, index, dataset, frame_rate_hz)
            break

    return y, y_actual_onset_only


def read_y_csv(path_to_csv, length_seconds, frame_rate_hz, dataset):
    """Read labels of a wave file (csv format, dataset 4). Returns a tuple (y_part, y_actual_onset_only_part)."""

    y = _init_y(length_seconds, frame_rate_hz)
    y_actual_onset_only = _init_y(length_seconds, frame_rate_hz)
    with open(path_to_csv) as f:
        for line in f:
            line_split = line.rstrip().split(',')
            onset_time = float(line_split[0])
            index = _onset_index(onset_time, frame_rate_hz)
            _set_onset_label_adjusted_with_neighbors(y, y_actual_onset_only, index, dataset, frame_rate_hz)

    return y, y_actual_onset_only


def _init_y(length_seconds, frame_rate_hz):
    return np.zeros(int(round(frame_rate_hz * length_seconds)), dtype=np.int8)


def _onset_index(onset_time, frame_rate_hz):
    return int(onset_time * frame_rate_hz)


def _set_onset_label_orig(y, y_actual_onset_only, index):
    y[index] = 1
    y_actual_onset_only[index] = 1


def _set_onset_label_orig_with_neighbors(y, y_actual_onset_only, index):
    # index = 5: start = 4, end = 7, 4:7 = 1 -> 4, 5, 6 = 1
    start = max(0, index - 1)
    end = min(len(y), index + 2)
    y[start:end] = 1
    y_actual_onset_only[index] = 1


def _set_onset_label_adjusted_with_neighbors(y, y_actual_onset_only, index, dataset, frame_rate_hz):
    adjustment = int(round(DATASET_CORRECTIONS[dataset] * frame_rate_hz))
    index += adjustment

    start = max(0, index - 1)
    end = min(len(y), index + 2)
    y[start:end] = 1
    y_actual_onset_only[index] = 1


def read_onset_times(path_to_truth, dataset, truth_format, onset_group_threshold_seconds):
    if truth_format == 'xml':
        onset_times = _read_onset_times_xml(path_to_truth, dataset)
    elif truth_format == 'csv':
        onset_times = _read_onset_times_csv(path_to_truth, dataset)
    else:
        raise ValueError('Unknown truth format')

    onset_times = sorted(onset_times)
    onset_times_grouped = group_onsets(onset_times, onset_group_threshold_seconds)

    return onset_times_grouped


def _read_onset_times_xml(path_to_xml, dataset):
    tree = ElementTree.parse(path_to_xml)
    root = tree.getroot()
    onset_times = []
    for root_child in root:
        if root_child.tag == 'transcription':
            for event in root_child:
                if event.tag != 'event':
                    raise ValueError('Unexpected XML element, expected event, got ' + event.tag)
                onset_time = None
                for event_child in event:
                    if event_child.tag == 'onsetSec':
                        onset_time = float(event_child.text) + DATASET_CORRECTIONS[dataset]
                if onset_time is not None:
                    onset_times.append(onset_time)
                else:
                    raise ValueError('File {} does not contain onset information.'.format(path_to_xml))
            break

    return onset_times


def _read_onset_times_csv(path_to_csv, dataset):
    onset_times = []
    with open(path_to_csv) as f:
        for line in f:
            line_split = line.rstrip().split(',')
            onset_time = float(line_split[0])
            onset_times.append(onset_time)

    return onset_times


def group_onsets(onset_times, onset_group_threshold_seconds, epsilon=1e-6):
    """Assumes onset times are sorted.

    Group onsets in a way that onsets closer than onset_group_threshold_seconds belong to the same group.
    """

    if len(onset_times) == 0:
        return onset_times

    onset_times_grouped = []
    last_onset = None
    onset_group_start = None
    for onset_time in onset_times:
        if last_onset is not None and onset_time - last_onset > onset_group_threshold_seconds + epsilon:
            onset_times_grouped.append(onset_group_start)
            onset_group_start = None
        last_onset = onset_time
        if onset_group_start is None:
            onset_group_start = onset_time
    onset_times_grouped.append(onset_group_start)

    return onset_times_grouped
