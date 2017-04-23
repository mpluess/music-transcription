import numpy as np
from os import listdir
import os.path
from os.path import isdir, isfile
import soundfile
from xml.etree import ElementTree
from warnings import warn


def read_X(path_to_wav, frame_rate_hz, expected_sample_rate, subsampling_step):
    # scipy.io.wavfile is not able to read 24-bit data hence the need to use this alternative library
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


def read_y_xml(path_to_xml, length_seconds, frame_rate_hz, dataset):
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
                        _set_onset_label_orig_with_neighbors(y, y_actual_onset_only, index)
                        # _set_onset_label_adjusted_with_neighbors(y, y_actual_onset_only, index, dataset)
            break

    return y, y_actual_onset_only


def read_y_csv(path_to_csv, length_seconds, frame_rate_hz, dataset):
    y = _init_y(length_seconds, frame_rate_hz)
    y_actual_onset_only = _init_y(length_seconds, frame_rate_hz)
    with open(path_to_csv) as f:
        for line in f:
            line_split = line.rstrip().split(',')
            onset_time = float(line_split[0])
            index = _onset_index(onset_time, frame_rate_hz)
            _set_onset_label_orig_with_neighbors(y, y_actual_onset_only, index)
            # _set_onset_label_adjusted_with_neighbors(y, y_actual_onset_only, index, dataset)

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


# def _set_onset_label_adjusted_range(y, y_start_only, index, dataset):
#     """Adjusted manually based on a few randomly sampled visualizations of onsets in the time domain per dataset."""
#
#     start = index
#     end = index
#     if dataset == 1:
#         # Python-style indices: start included, end not included
#         start += -2
#         end += 1
#     elif dataset == 2:
#         start += 2
#         end += 5
#     elif dataset == 3:
#         start += 0
#         end += 3
#     elif dataset == 4:
#         start += 0
#         end += 2
#     else:
#         raise ValueError('Invalid dataset label')
#
#     start = max(0, start)
#     end = min(len(y), end)
#     if end - start > 0:
#         # y[index] = 1
#         y[start:end] = 1
#         y_start_only[start] = 1


def read_X_y(path_to_wav, frame_rate_hz, expected_sample_rate, subsampling_step,
             path_to_truth, truth_format, dataset):
    X_part, length_seconds = read_X(path_to_wav, frame_rate_hz, expected_sample_rate, subsampling_step)
    if X_part is not None:
        if truth_format == 'xml':
            y_part, y_actual_onset_only_part = read_y_xml(path_to_truth, length_seconds, frame_rate_hz, dataset)
        elif truth_format == 'csv':
            y_part, y_actual_onset_only_part = read_y_csv(path_to_truth, length_seconds, frame_rate_hz, dataset)
        else:
            raise ValueError('Unknown truth format')

        if X_part.shape[0] != y_part.shape[0]:
            raise ValueError('X_part vs. y_part shape mismatch: ' + str(X_part.shape[0]) + ' != ' + str(y_part.shape[0]))
        return X_part, y_part, y_actual_onset_only_part
    else:
        return None, None, None


def read_data(active_datasets, frame_rate_hz, expected_sample_rate, subsampling_step):
    # TODO dataset einchecken, damit aenderungen zentral gemacht werden (anderes, privates repo)
    dir_tuples = []
    if 1 in active_datasets:
        path_to_ds_1 = r'data\IDMT-SMT-GUITAR_V2\dataset1'
        for guitar_desc in listdir(path_to_ds_1):
            dir_tuples.append((
                os.path.join(path_to_ds_1, guitar_desc, 'audio'),
                os.path.join(path_to_ds_1, guitar_desc, 'annotation'),
                1,
            ))

    if 2 in active_datasets:
        dir_tuples.append((
            r'data\IDMT-SMT-GUITAR_V2\dataset2\audio',
            r'data\IDMT-SMT-GUITAR_V2\dataset2\annotation',
            2,
        ))
    if 3 in active_datasets:
        dir_tuples.append((
            r'data\IDMT-SMT-GUITAR_V2\dataset3\audio',
            r'data\IDMT-SMT-GUITAR_V2\dataset3\annotation',
            3,
        ))

    file_tuples = []
    for audio_dir, annotation_dir, ds in dir_tuples:
        for wav_file in listdir(audio_dir):
            path_to_wav = os.path.join(audio_dir, wav_file)
            if wav_file.endswith('.wav'):
                path_to_xml = os.path.join(annotation_dir, wav_file.replace('.wav', '.xml'))
                if isfile(path_to_xml):
                    file_tuples.append((path_to_wav, path_to_xml, ds, 'xml'))
                else:
                    warn('Skipping ' + wav_file + ', no truth found.')
            else:
                warn('Skipping ' + path_to_wav + ', not a .wav file.')

    if 4 in active_datasets:
        for path_to_ds in [r'data\IDMT-SMT-GUITAR_V2\dataset4\Career SG', r'data\IDMT-SMT-GUITAR_V2\dataset4\Ibanez 2820']:
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
                                    file_tuples.append((path_to_wav, path_to_csv, 4, 'csv'))
                                else:
                                    # TODO fallback to other formats
                                    warn('Skipping ' + path_to_wav + ', no truth csv found.')
                            else:
                                warn('Skipping ' + path_to_wav + ', no onset folder.')
                        else:
                            warn('Skipping ' + path_to_wav + ', not a .wav file.')

    X_parts = []
    y_parts = []
    y_actual_onset_only_parts = []
    ds_labels = []
    for path_to_wav, path_to_truth, dataset, truth_format in file_tuples:
        X_part, y_part, y_actual_onset_only_part = read_X_y(path_to_wav, frame_rate_hz, expected_sample_rate,
                                                            subsampling_step, path_to_truth, truth_format, dataset)
        if X_part is not None and y_part is not None and y_actual_onset_only_part is not None:
            X_parts.append(X_part)
            y_parts.append(y_part)
            y_actual_onset_only_parts.append(y_actual_onset_only_part)
            ds_labels.append(dataset)

    return X_parts, y_parts, y_actual_onset_only_parts, ds_labels
