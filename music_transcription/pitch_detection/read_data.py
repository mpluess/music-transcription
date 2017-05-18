import soundfile
from warnings import warn
from xml.etree import ElementTree

import music_transcription.onset_detection.read_data

# Correction of onset times in seconds (see onset_detection.read_data._set_onset_label_adjusted_with_neighbors)
DATASET_CORRECTIONS = {
    1: 0.0,
    2: 0.03,
    3: 0.0,
    4: 0.0,
}


def get_wav_and_truth_files(active_datasets):
    return music_transcription.onset_detection.read_data.get_wav_and_truth_files(active_datasets)


def read_X_y(wav_file_paths, truth_dataset_format_tuples,
             frame_rate_hz, sample_rate, subsampling_step,
             min_pitch, max_pitch):
    list_of_samples = []
    list_of_onset_times = []
    list_of_pitches = []
    for path_to_wav, (path_to_xml, dataset, truth_format) in zip(wav_file_paths, truth_dataset_format_tuples):
        if truth_format != 'xml':
            raise ValueError('Unsupported format {}'.format(truth_format))
        samples = read_samples(path_to_wav, frame_rate_hz, sample_rate, subsampling_step)
        onset_times, pitches = read_onset_times_pitches(path_to_xml, min_pitch, max_pitch, dataset)
        if samples is not None and onset_times is not None and pitches is not None:
            list_of_samples.append(samples)
            list_of_onset_times.append(onset_times)
            list_of_pitches.append(pitches)

    return (list_of_samples, list_of_onset_times), list_of_pitches


def read_samples(path_to_wav, frame_rate_hz, expected_sample_rate, subsampling_step):
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


def read_onset_times_pitches(path_to_xml, min_pitch, max_pitch, dataset):
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

    return onset_times, pitches
