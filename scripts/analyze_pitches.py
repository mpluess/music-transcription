from collections import defaultdict
from xml.etree import ElementTree

from music_transcription.pitch_detection.read_data import get_wav_and_truth_files, read_y


def get_tuning(path_to_xml):
    tree = ElementTree.parse(path_to_xml)
    root = tree.getroot()
    for root_child in root:
        if root_child.tag == 'globalParameter':
            for param in root_child:
                if param.tag == 'instrumentTuning':
                    return param.text
            break

    return 'MISSING'

active_datasets = {1, 2, 3}

wav_file_paths, truth_dataset_format_tuples = get_wav_and_truth_files(active_datasets)
pitches = defaultdict(int)
tunings = defaultdict(int)
for path_to_xml, dataset, truth_format in truth_dataset_format_tuples:
    if truth_format != 'xml':
        raise ValueError('Unsupported format {}'.format(truth_format))
    _, pitch_list = read_y(path_to_xml)
    for pitch in pitch_list:
        pitches[pitch] += 1
    tunings[get_tuning(path_to_xml)] += 1

print('PITCHES')
print('min pitch = {}'.format(min(pitches.keys())))
print('max pitch = {}'.format(max(pitches.keys())))
print('nr of pitches = {}'.format(sum(pitches.values())))
for pitch, count in sorted(pitches.items()):
    print('{}: {}'.format(pitch, count))

print('')
print('TUNINGS')
print('nr of files = {}'.format(sum(tunings.values())))
for tuning, count in sorted(tunings.items()):
    print('{}: {}'.format(tuning, count))
