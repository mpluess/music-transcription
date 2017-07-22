from xml.etree import ElementTree
from warnings import warn
from sklearn.preprocessing import MultiLabelBinarizer

import music_transcription.pitch_detection.read_data as pitch_read_data
from music_transcription.read_data import DATASET_CORRECTIONS


def read_samples(path_to_wav, expected_sample_rate, subsampling_step, frame_rate_hz=None):
    # return (subsampled) samples
    return pitch_read_data.read_samples(path_to_wav, expected_sample_rate, subsampling_step, frame_rate_hz)


def read_data_y(wav_file_paths, truth_dataset_format_tuples, sample_rate, subsampling_step, n_strings,
                onset_group_threshold_seconds=0.05, frame_rate_hz=None):
    list_of_samples = []
    list_of_onset_times = []
    list_of_pitches = []
    list_of_strings = []
    wav_file_paths_valid = []
    truth_dataset_format_tuples_valid = []
    for path_to_wav, truth_dataset_format_tuple in zip(wav_file_paths, truth_dataset_format_tuples):
        path_to_xml, dataset, truth_format = truth_dataset_format_tuple
        if truth_format != 'xml':
            raise ValueError('Unsupported format {}'.format(truth_format))
        samples = read_samples(path_to_wav, sample_rate, subsampling_step, frame_rate_hz=frame_rate_hz)
        onset_times_grouped, pitches_grouped, strings_grouped = _read_onset_times_strings(path_to_xml,
                                                                                         dataset,
                                                                                         n_strings,
                                                                                         onset_group_threshold_seconds)
        if samples is not None and onset_times_grouped is not None and strings_grouped is not None:
            list_of_samples.append(samples)
            list_of_onset_times.append(onset_times_grouped)
            list_of_pitches.append(pitches_grouped)
            list_of_strings.append(strings_grouped)
            wav_file_paths_valid.append(path_to_wav)
            truth_dataset_format_tuples_valid.append(truth_dataset_format_tuple)

    # label_binarizer = MultiLabelBinarizer(classes=range(1, n_strings + 1))
    # label_binarizer.fit(None)  # fit needs to be called before transform
    #
    # string_groups_flat = [string_group for strings_grouped in list_of_strings for string_group in strings_grouped]
    # assert len(string_groups_flat) == sum([len(onset_times) for onset_times in list_of_onset_times])
    #
    # pitch_groups_flat = [pitch_group for pitches_grouped in list_of_pitches for pitch_group in pitches_grouped]
    # assert len(pitch_groups_flat) == len(string_groups_flat)
    #
    # y = label_binarizer.transform(string_groups_flat)

    data = (list_of_samples, list_of_onset_times, list_of_pitches, list_of_strings)
    return data, wav_file_paths_valid, truth_dataset_format_tuples_valid


# read y from XML
def _read_onset_times_strings(path_to_xml, dataset, n_strings, onset_group_threshold_seconds):
    tree = ElementTree.parse(path_to_xml)
    root = tree.getroot()
    onset_times = []
    pitches = []
    strings = []
    for root_child in root:
        if root_child.tag == 'transcription':
            for event in root_child:
                if event.tag != 'event':
                    raise ValueError('Unexpected XML element, expected event, got ' + event.tag)
                onset_time = None
                string = None
                for event_child in event:
                    if event_child.tag == 'onsetSec':
                        onset_time = float(event_child.text) + DATASET_CORRECTIONS[dataset]
                    elif event_child.tag == 'stringNumber':
                        string = int(event_child.text)
                    # TODO use globalParameter -> instrumentTuning for automatic pitch/string-fret error detection
                    elif event_child.tag == 'pitch':
                        pitch = int(event_child.text)
                if onset_time is not None and string is not None:
                    if 1 <= string <= n_strings:
                        onset_times.append(onset_time)
                        pitches.append(pitch)
                        strings.append(string)
                    else:
                        warn('Skipping {}, string {} is out of range.'.format(path_to_xml, string))
                        return None, None
                else:
                    raise ValueError('File {} misses onset or string information: onset_time={}, string={}'.format(
                        path_to_xml, onset_time, string))
            break

    onset_string_tuples_sorted = sorted(zip(onset_times, pitches, strings), key=lambda t: t[0])
    onset_times = [t[0] for t in onset_string_tuples_sorted]
    pitches = [t[1] for t in onset_string_tuples_sorted]
    strings = [t[2] for t in onset_string_tuples_sorted]

    onsets_grouped, strings_grouped = pitch_read_data._group_onsets(onset_times, strings, onset_group_threshold_seconds)
    _, pitches_grouped = pitch_read_data._group_onsets(onset_times, pitches, onset_group_threshold_seconds)
    return onsets_grouped, pitches_grouped, strings_grouped
