import numpy as np
from os import listdir
import os.path
from os.path import isdir, isfile
import soundfile
from xml.etree import ElementTree
from warnings import warn

DATA_DIR = r'..\data'

# Correction of onset times in seconds (see onset_detection.read_data._set_onset_label_adjusted_with_neighbors)
DATASET_CORRECTIONS = {
    # IDMT-SMT-GUITAR_V2 dataset1
    1: 0.0,
    # IDMT-SMT-GUITAR_V2 dataset2
    2: 0.03,
    # IDMT-SMT-GUITAR_V2 dataset3
    3: 0.0,
    # IDMT-SMT-GUITAR_V2 dataset4
    4: 0.0,
    # recordings
    5: 0.0,
    # IDMT-SMT-AUDIO-EFFECTS monophonic
    6: 0.0,
    # IDMT-SMT-AUDIO-EFFECTS polyphonic
    7: 0.0,
    # generated
    8: 0.0,
}


def get_wav_and_truth_files(active_datasets):
    """Get wave files and truth information. Return a tuple (wav_file_paths, truth_dataset_format_tuples)

    Input:
    active_datasets: set of datasets to be loaded

    Output:
    wav_file_paths: List of wave file paths
    truth_dataset_format_tuples: List of tuples (path_to_truth_file, dataset, format)

    dataset labels: one of 1, 2, 3, 4, 5, 6, 7, 8
    truth formats: one of 'csv', 'xml'
    """

    dir_tuples = []
    if 1 in active_datasets:
        path_to_ds_1 = os.path.join(DATA_DIR, r'IDMT-SMT-GUITAR_V2\dataset1')
        for guitar_desc in listdir(path_to_ds_1):
            dir_tuples.append((
                os.path.join(path_to_ds_1, guitar_desc, 'audio'),
                os.path.join(path_to_ds_1, guitar_desc, 'annotation'),
                1,
            ))
    if 2 in active_datasets:
        dir_tuples.append((
            os.path.join(DATA_DIR, r'IDMT-SMT-GUITAR_V2\dataset2\audio'),
            os.path.join(DATA_DIR, r'IDMT-SMT-GUITAR_V2\dataset2\annotation'),
            2,
        ))
    if 3 in active_datasets:
        dir_tuples.append((
            os.path.join(DATA_DIR, r'IDMT-SMT-GUITAR_V2\dataset3\audio'),
            os.path.join(DATA_DIR, r'IDMT-SMT-GUITAR_V2\dataset3\annotation'),
            3,
        ))
    if 5 in active_datasets:
        dir_tuples.append((
            os.path.join(DATA_DIR, r'recordings\audio'),
            os.path.join(DATA_DIR, r'recordings\annotation'),
            5,
        ))
    if 6 in active_datasets:
        path_to_ds = os.path.join(DATA_DIR, 'IDMT-SMT-AUDIO-EFFECTS', 'Gitarre monophon')
        for effect_desc in listdir(os.path.join(path_to_ds, 'Samples')):
            dir_tuples.append((
                os.path.join(path_to_ds, 'Samples', effect_desc),
                os.path.join(path_to_ds, 'annotation', effect_desc),
                6,
            ))
    if 7 in active_datasets:
        path_to_ds = os.path.join(DATA_DIR, 'IDMT-SMT-AUDIO-EFFECTS', 'Gitarre polyphon')
        for effect_desc in listdir(os.path.join(path_to_ds, 'Samples')):
            dir_tuples.append((
                os.path.join(path_to_ds, 'Samples', effect_desc),
                os.path.join(path_to_ds, 'annotation', effect_desc),
                7,
            ))
    if 8 in active_datasets:
        dir_tuples.append((
            os.path.join(DATA_DIR, r'generated\audio'),
            os.path.join(DATA_DIR, r'generated\annotation'),
            8,
        ))

    wav_file_paths = []
    truth_dataset_format_tuples = []
    for audio_dir, annotation_dir, ds in dir_tuples:
        for wav_file in listdir(audio_dir):
            path_to_wav = os.path.join(audio_dir, wav_file)
            if wav_file.endswith('.wav'):
                path_to_xml = os.path.join(annotation_dir, wav_file.replace('.wav', '.xml'))
                if isfile(path_to_xml):
                    wav_file_paths.append(path_to_wav)
                    truth_dataset_format_tuples.append((path_to_xml, ds, 'xml'))
                else:
                    warn('Skipping ' + wav_file + ', no truth found.')
            else:
                warn('Skipping ' + path_to_wav + ', not a .wav file.')

    if 4 in active_datasets:
        for path_to_ds in [
            os.path.join(DATA_DIR, r'IDMT-SMT-GUITAR_V2\dataset4\Career SG'),
            os.path.join(DATA_DIR, r'IDMT-SMT-GUITAR_V2\dataset4\Ibanez 2820')
        ]:
            for tempo in listdir(path_to_ds):
                path_to_tempo = os.path.join(path_to_ds, tempo)
                for genre in listdir(path_to_tempo):
                    path_to_genre = os.path.join(path_to_tempo, genre)
                    path_to_audio = os.path.join(path_to_genre, 'audio')
                    for wav_file in listdir(path_to_audio):
                        path_to_wav = os.path.join(path_to_audio, wav_file)
                        if wav_file.endswith('.wav'):
                            path_to_onsets = os.path.join(path_to_genre, 'annotation', 'onsets')
                            if isdir(path_to_onsets):
                                path_to_csv = os.path.join(path_to_onsets, wav_file.replace('.wav', '.csv'))
                                if isfile(path_to_csv):
                                    wav_file_paths.append(path_to_wav)
                                    truth_dataset_format_tuples.append((path_to_csv, 4, 'csv'))
                                else:
                                    # TODO fallback to other formats
                                    warn('Skipping ' + path_to_wav + ', no truth csv found.')
                            else:
                                warn('Skipping ' + path_to_wav + ', no onset folder.')
                        else:
                            warn('Skipping ' + path_to_wav + ', not a .wav file.')

    return wav_file_paths, truth_dataset_format_tuples


def read_X_y(path_to_wav, frame_rate_hz, expected_sample_rate, subsampling_step,
             path_to_truth, truth_format, dataset):
    """Read samples and labels of a wave file.

    Returns a tuple (X_part, y_part, y_actual_onset_only_part)
    """

    X_part, length_seconds = read_X(path_to_wav, frame_rate_hz, expected_sample_rate, subsampling_step)
    if X_part is not None:
        y_part, y_actual_onset_only_part = read_y(truth_format, path_to_truth, length_seconds, frame_rate_hz, dataset)
        if X_part.shape[0] != y_part.shape[0]:
            raise ValueError('X_part vs. y_part shape mismatch: ' + str(X_part.shape[0]) + ' != ' + str(y_part.shape[0]))
        return X_part, y_part, y_actual_onset_only_part
    else:
        return None, None, None


def read_X(path_to_wav, frame_rate_hz, expected_sample_rate, subsampling_step):
    """Read samples of a wave file. Returns a tuple (sample_np_array, length_seconds).

    sample_np_array.shape = (n_frames, ceil(sample_rate / frame_rate_hz / subsampling_step))
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
    X = []
    # Cut off last samples
    while offset <= len(samples) - samples_per_frame:
        X.append(samples[offset:offset + samples_per_frame:subsampling_step])
        offset += samples_per_frame

    X = np.array(X)
    return X, offset / sample_rate


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
                        _set_onset_label_adjusted_with_neighbors(y, y_actual_onset_only, index, dataset)
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
            # _set_onset_label_orig_with_neighbors(y, y_actual_onset_only, index)
            _set_onset_label_adjusted_with_neighbors(y, y_actual_onset_only, index, dataset)

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


def _set_onset_label_adjusted_with_neighbors(y, y_actual_onset_only, index, dataset):
    """Adjusted by fitting a model on dataset 4 with original labels and setting the offset per dataset to where
    the prediction results were best using this model."""

    # No adjustment needed for 1 and 3.
    # The labels of dataset 4 seem to be on spot - the onset is visible around the original label.
    if dataset == 2:
        index += 3

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
