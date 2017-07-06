from collections import defaultdict
from os import listdir
from os.path import isdir, isfile, join
from warnings import warn
from xml.etree import ElementTree


def merge_truth(root_dir):
    pitches = read_pitches(root_dir)
    n_files_written = 0
    for dataset_part in listdir(root_dir):
        if dataset_part == 'Gitarre monophon':
            path_to_dataset_part = join(root_dir, dataset_part)
            if isdir(path_to_dataset_part):
                print(dataset_part)
                path_to_samples = join(path_to_dataset_part, 'Samples')
                for fx_folder in listdir(path_to_samples):
                    path_to_fx_folder = join(path_to_samples, fx_folder)
                    if isdir(path_to_fx_folder):
                        print(fx_folder)
                        for wav_file in listdir(path_to_fx_folder):
                            path_to_wav_file = join(path_to_fx_folder, wav_file)
                            if isfile(path_to_wav_file) and wav_file.endswith('.wav'):
                                file_id = wav_file[:-4]
                                path_to_onset_file = join(path_to_dataset_part, 'Onsets', fx_folder, file_id + '.txt')
                                if isfile(path_to_onset_file):
                                    with open(path_to_onset_file) as f:
                                        onset_time = f.read().rstrip()
                                    if onset_time == '':
                                        warn('No onset available for {}'.format(path_to_wav_file))
                                    else:
                                        write_truth_file(
                                            join(path_to_dataset_part, 'annotation', fx_folder, file_id + '.xml'),
                                            pitches[dataset_part][fx_folder][file_id],
                                            onset_time
                                        )
                                        n_files_written += 1
                                else:
                                    warn('No onset file found for {}'.format(path_to_wav_file))
                            else:
                                warn('Not a WAV file: {}'.format(path_to_wav_file))
                    else:
                        warn('Not an FX folder: {}'.format(path_to_fx_folder))
                print('')

    print('n_files_written={}'.format(n_files_written))


def read_pitches(root_dir):
    pitches = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: dict())))
    for dataset_part in listdir(root_dir):
        if dataset_part == 'Gitarre monophon':
            path_to_dataset_part = join(root_dir, dataset_part)
            if isdir(path_to_dataset_part):
                path_to_lists = join(path_to_dataset_part, 'Lists')
                for fx_folder in listdir(path_to_lists):
                    path_to_fx_folder = join(path_to_lists, fx_folder)
                    if isdir(path_to_fx_folder):
                        for xml_file in listdir(path_to_fx_folder):
                            path_to_xml_file = join(path_to_fx_folder, xml_file)
                            if isfile(path_to_xml_file) and xml_file.endswith('.xml'):
                                tree = ElementTree.parse(path_to_xml_file)
                                root = tree.getroot()
                                for root_child in root:
                                    if root_child.tag == 'audiofile':
                                        pitch = None
                                        file_id = None
                                        for attribute in root_child:
                                            if attribute.tag == 'midinr':
                                                pitch = attribute.text
                                            elif attribute.tag == 'fileID':
                                                file_id = attribute.text
                                        if pitch is None:
                                            warn('No pitch in {}'.format(path_to_xml_file))
                                        elif file_id is None:
                                            warn('No file id in {}'.format(path_to_xml_file))
                                        else:
                                            pitches[dataset_part][fx_folder][file_id] = pitch
                            else:
                                warn('Not an XML file: {}'.format(path_to_xml_file))
                    else:
                        warn('Not an FX folder: {}'.format(path_to_fx_folder))

    return pitches


# TODO Refactor to music_transcription.fileformat.truth.write_truth_file
def write_truth_file(path_to_truth_file, pitch, onset_time):
    with open(path_to_truth_file, 'w') as f:
        f.write('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n')
        f.write('<instrumentRecording>\n')
        f.write('  <globalParameter></globalParameter>\n')
        f.write('  <transcription>\n')
        f.write('    <event>\n')
        f.write('      <pitch>{}</pitch>\n'.format(pitch))
        f.write('      <onsetSec>{}</onsetSec>\n'.format(onset_time))
        f.write('    </event>\n')
        f.write('  </transcription>\n')
        f.write('</instrumentRecording>\n')

root_dir = '../data/IDMT-SMT-AUDIO-EFFECTS'
merge_truth(root_dir)
