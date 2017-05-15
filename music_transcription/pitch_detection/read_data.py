import soundfile
from warnings import warn
from xml.etree import ElementTree

import music_transcription.onset_detection.read_data


def get_wav_and_truth_files(active_datasets):
    return music_transcription.onset_detection.read_data.get_wav_and_truth_files(active_datasets)


def read_X(path_to_wav, frame_rate_hz, expected_sample_rate, subsampling_step):
    samples, sample_rate = soundfile.read(path_to_wav)
    if len(samples.shape) > 1:
        warn('Skipping ' + path_to_wav + ', cannot handle stereo signal.')
        return None, -1
    elif sample_rate != expected_sample_rate:
        warn('Skipping ' + path_to_wav +
             ', sample rate ' + str(sample_rate) + ' != expected sample rate ' + str(expected_sample_rate) + '.')
        return None, -1
    elif sample_rate % frame_rate_hz != 0:
        raise ValueError('Sample rate ' + str(sample_rate) + ' % frame rate ' + str(frame_rate_hz) + ' != 0')

    return samples[::subsampling_step]


def read_y(path_to_xml):
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
                        onset_time = float(event_child.text)
                    elif event_child.tag == 'pitch':
                        pitch = int(event_child.text)
                if onset_time is not None and pitch is not None:
                    onset_times.append(onset_time)
                    pitches.append(pitch)
                else:
                    raise ValueError('File {} does not contain both onset and pitch information: onset_time={}, pitch={}'.format(path_to_xml, onset_time, pitch))
            break

    return onset_times, pitches
