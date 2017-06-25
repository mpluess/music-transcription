from sklearn.preprocessing import MultiLabelBinarizer
import soundfile
from warnings import warn
from xml.etree import ElementTree

import music_transcription.onset_detection.read_data

# Correction of onset times in seconds (see onset_detection.read_data._set_onset_label_adjusted_with_neighbors)
DATASET_CORRECTIONS = music_transcription.onset_detection.read_data.DATASET_CORRECTIONS


def get_wav_and_truth_files(active_datasets):
    return music_transcription.onset_detection.read_data.get_wav_and_truth_files(active_datasets)


def read_data_y(wav_file_paths, truth_dataset_format_tuples,
                frame_rate_hz, sample_rate, subsampling_step,
                min_pitch, max_pitch, onset_group_threshold_seconds=0.03):
    list_of_samples = []
    list_of_onset_times = []
    list_of_pitches = []
    wav_file_paths_valid = []
    truth_dataset_format_tuples_valid = []
    for path_to_wav, truth_dataset_format_tuple in zip(wav_file_paths, truth_dataset_format_tuples):
        path_to_xml, dataset, truth_format = truth_dataset_format_tuple
        if truth_format != 'xml':
            raise ValueError('Unsupported format {}'.format(truth_format))
        samples = read_samples(path_to_wav, frame_rate_hz, sample_rate, subsampling_step)
        onset_times_grouped, pitches_grouped = _read_onset_times_pitches(path_to_xml, min_pitch, max_pitch, dataset,
                                                                         onset_group_threshold_seconds)
        if samples is not None and onset_times_grouped is not None and pitches_grouped is not None:
            list_of_samples.append(samples)
            list_of_onset_times.append(onset_times_grouped)
            list_of_pitches.append(pitches_grouped)
            wav_file_paths_valid.append(path_to_wav)
            truth_dataset_format_tuples_valid.append(truth_dataset_format_tuple)

    label_binarizer = MultiLabelBinarizer(classes=range(min_pitch, max_pitch + 1))
    label_binarizer.fit(None)

    pitch_groups_flat = [pitch_group for pitches_grouped in list_of_pitches for pitch_group in pitches_grouped]
    assert len(pitch_groups_flat) == sum([len(onset_times) for onset_times in list_of_onset_times])
    y = label_binarizer.transform(pitch_groups_flat)

    return (list_of_samples, list_of_onset_times), y, wav_file_paths_valid, truth_dataset_format_tuples_valid


def read_samples(path_to_wav, frame_rate_hz, expected_sample_rate, subsampling_step):
    """WARNING: Only use this function if you don't have labels for a file. Otherwise always use read_data_y, even
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
    elif sample_rate % frame_rate_hz != 0:
        raise ValueError('Sample rate ' + str(sample_rate) + ' % frame rate ' + str(frame_rate_hz) + ' != 0')

    return samples[::subsampling_step]


def _read_onset_times_pitches(path_to_xml, min_pitch, max_pitch, dataset, onset_group_threshold_seconds):
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

    onset_pitch_tuples_sorted = sorted(zip(onset_times, pitches), key=lambda t: t[0])
    onset_times = [t[0] for t in onset_pitch_tuples_sorted]
    pitches = [t[1] for t in onset_pitch_tuples_sorted]

    onset_times_grouped, pitches_grouped = _group_onsets(onset_times, pitches, onset_group_threshold_seconds)

    return onset_times_grouped, pitches_grouped


def _group_onsets(onset_times, pitches, onset_group_threshold_seconds, epsilon=1e-6):
    """Assumes onset times are sorted (pitch_detection.read_data.read_y does that).

    Group onsets and corresponding pitches in a way that onsets closer than onset_group_threshold_seconds
    belong to the same group.
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
