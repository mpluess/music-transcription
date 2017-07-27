"""Count the number of files with at least one polyphonic onset as well as the number of polyphonic onsets."""

from collections import defaultdict
import os
from os import listdir
from os.path import isdir, isfile
from warnings import warn
from xml.etree import ElementTree

DATA_DIR = r'..\data'


def read_file_tuples(active_datasets):
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
                                    file_tuples.append((path_to_wav, path_to_csv, 4, 'csv'))
                                else:
                                    # TODO fallback to other formats
                                    warn('Skipping ' + path_to_wav + ', no truth csv found.')
                            else:
                                warn('Skipping ' + path_to_wav + ', no onset folder.')
                        else:
                            warn('Skipping ' + path_to_wav + ', not a .wav file.')

    return file_tuples


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

active_datasets = {1, 2, 3, 4}
MAX_POLYPHONY_DIFF = 0.05

file_tuples = read_file_tuples(active_datasets)

# print(len(file_tuples))
# keyfunc = lambda t: t[2]
# for k, g in groupby(sorted(file_tuples, key=keyfunc), key=keyfunc):
#     print('{} {}'.format(k, len(list(g))))

counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for path_to_wav, path_to_truth, dataset, truth_format in file_tuples:
    if truth_format == 'csv':
        onsets = get_onsets_csv(path_to_truth)
    elif truth_format == 'xml':
        onsets = get_onsets_xml(path_to_truth)
    else:
        raise ValueError('unknown truth format')

    last_onset_time = -1.0
    for onset_time in onsets:
        if onset_time - last_onset_time < MAX_POLYPHONY_DIFF:
            counts[dataset][path_to_wav]['poly'] += 1
        else:
            counts[dataset][path_to_wav]['mono'] += 1
        last_onset_time = onset_time

print(len(file_tuples))
for dataset in sorted(active_datasets):
    print(dataset)
    print('n_files={}'.format(len(counts[dataset].keys())))
    print('n_files_poly={}'.format(len([path_to_wav for path_to_wav in counts[dataset].keys()
                                        if counts[dataset][path_to_wav]['poly'] > 0])))
    print('n_onsets_mono={}'.format(sum([counts[dataset][path_to_wav]['mono']
                                         for path_to_wav in counts[dataset].keys()])))
    print('n_onsets_poly={}'.format(sum([counts[dataset][path_to_wav]['poly']
                                         for path_to_wav in counts[dataset].keys()])))
