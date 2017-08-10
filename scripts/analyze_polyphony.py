"""Count the number of files with at least one polyphonic onset as well as the number of polyphonic onsets."""

from collections import defaultdict
from xml.etree import ElementTree

from music_transcription.read_data import get_wav_and_truth_files


def get_onsets_csv(path_to_csv):
    onsets = []
    with open(path_to_csv) as f:
        for line in f:
            line_split = line.rstrip().split(',')
            onsets.append(float(line_split[0]))

    return sorted(onsets)


def get_onsets_xml(path_to_xml):
    tree = ElementTree.parse(path_to_xml)
    root = tree.getroot()
    onsets = []
    for root_child in root:
        if root_child.tag == 'transcription':
            for event in root_child:
                if event.tag != 'event':
                    raise ValueError('Unexpected XML element, expected event, got ' + event.tag)
                for event_child in event:
                    if event_child.tag == 'onsetSec':
                        onsets.append(float(event_child.text))
            break

    return sorted(onsets)

active_datasets = {1, 2, 3, 4, 9, 10, 11}
MAX_POLYPHONY_DIFF = 0.05

wav_file_paths, truth_dataset_format_tuples = get_wav_and_truth_files(active_datasets)

# print(len(file_tuples))
# keyfunc = lambda t: t[2]
# for k, g in groupby(sorted(file_tuples, key=keyfunc), key=keyfunc):
#     print('{} {}'.format(k, len(list(g))))

counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for path_to_wav, (path_to_truth, dataset, truth_format) in zip(wav_file_paths, truth_dataset_format_tuples):
    if truth_format == 'csv':
        onsets = get_onsets_csv(path_to_truth)
    elif truth_format == 'xml':
        onsets = get_onsets_xml(path_to_truth)
    else:
        raise ValueError('unknown truth format')

    for i in range(len(onsets)):
        if i > 0 and onsets[i] - onsets[i - 1] < MAX_POLYPHONY_DIFF:
            counts[dataset][path_to_wav]['poly'] += 1
        elif i < len(onsets) - 1 and onsets[i + 1] - onsets[i] < MAX_POLYPHONY_DIFF:
            counts[dataset][path_to_wav]['poly'] += 1
        else:
            counts[dataset][path_to_wav]['mono'] += 1

for dataset in sorted(active_datasets):
    print(dataset)
    print('n_files={}'.format(len(counts[dataset].keys())))
    print('n_files_poly={}'.format(len([path_to_wav for path_to_wav in counts[dataset].keys()
                                        if counts[dataset][path_to_wav]['poly'] > 0])))
    print('n_onsets_mono={}'.format(sum([counts[dataset][path_to_wav]['mono']
                                         for path_to_wav in counts[dataset].keys()])))
    print('n_onsets_poly={}'.format(sum([counts[dataset][path_to_wav]['poly']
                                         for path_to_wav in counts[dataset].keys()])))
