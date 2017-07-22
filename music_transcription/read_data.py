from os import listdir
import os.path
from os.path import isdir, isfile

from warnings import warn

# Correction of onset times in seconds.
# Adjusted by fitting a model on dataset 4 with original labels and setting the offset per dataset to where
# the prediction results were best using this model."""
# No adjustment needed for 1 and 3.
# The labels of dataset 4 seem to be on spot - the onset is visible around the original label.
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
    # IDMT-SMT-AUDIO-EFFECTS monophonic NoFX
    9: 0.0,
    # IDMT-SMT-AUDIO-EFFECTS polyphonic NoFX
    10: 0.0,
    # generated 41, 42, 75-88
    11: 0.0,
}


def get_wav_and_truth_files(active_datasets, data_dir=r'..\data'):
    """Get wave files and truth information. Return a tuple (wav_file_paths, truth_dataset_format_tuples)

    Input:
    active_datasets: set of datasets to be loaded

    Output:
    wav_file_paths: List of wave file paths
    truth_dataset_format_tuples: List of tuples (path_to_truth_file, dataset, format)

    dataset labels: one of 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11
    truth formats: one of 'csv', 'xml'
    """

    dir_tuples = []
    if 1 in active_datasets:
        path_to_ds_1 = os.path.join(data_dir, r'IDMT-SMT-GUITAR_V2\dataset1')
        for guitar_desc in listdir(path_to_ds_1):
            dir_tuples.append((
                os.path.join(path_to_ds_1, guitar_desc, 'audio'),
                os.path.join(path_to_ds_1, guitar_desc, 'annotation'),
                1,
            ))
    if 2 in active_datasets:
        dir_tuples.append((
            os.path.join(data_dir, r'IDMT-SMT-GUITAR_V2\dataset2\audio'),
            os.path.join(data_dir, r'IDMT-SMT-GUITAR_V2\dataset2\annotation'),
            2,
        ))
    if 3 in active_datasets:
        dir_tuples.append((
            os.path.join(data_dir, r'IDMT-SMT-GUITAR_V2\dataset3\audio'),
            os.path.join(data_dir, r'IDMT-SMT-GUITAR_V2\dataset3\annotation'),
            3,
        ))
    if 5 in active_datasets:
        dir_tuples.append((
            os.path.join(data_dir, r'recordings\audio'),
            os.path.join(data_dir, r'recordings\annotation'),
            5,
        ))
    if 6 in active_datasets:
        path_to_ds = os.path.join(data_dir, 'IDMT-SMT-AUDIO-EFFECTS', 'Gitarre monophon')
        for effect_desc in listdir(os.path.join(path_to_ds, 'Samples')):
            dir_tuples.append((
                os.path.join(path_to_ds, 'Samples', effect_desc),
                os.path.join(path_to_ds, 'annotation', effect_desc),
                6,
            ))
    if 7 in active_datasets:
        path_to_ds = os.path.join(data_dir, 'IDMT-SMT-AUDIO-EFFECTS', 'Gitarre polyphon')
        for effect_desc in listdir(os.path.join(path_to_ds, 'Samples')):
            dir_tuples.append((
                os.path.join(path_to_ds, 'Samples', effect_desc),
                os.path.join(path_to_ds, 'annotation', effect_desc),
                7,
            ))
    if 8 in active_datasets:
        dir_tuples.append((
            os.path.join(data_dir, r'generated\all\audio'),
            os.path.join(data_dir, r'generated\all\annotation'),
            8,
        ))
    if 9 in active_datasets:
        path_to_ds = os.path.join(data_dir, 'IDMT-SMT-AUDIO-EFFECTS', 'Gitarre monophon')
        dir_tuples.append((
            os.path.join(path_to_ds, 'Samples', 'NoFX'),
            os.path.join(path_to_ds, 'annotation', 'NoFX'),
            9,
        ))
    if 10 in active_datasets:
        path_to_ds = os.path.join(data_dir, 'IDMT-SMT-AUDIO-EFFECTS', 'Gitarre polyphon')
        dir_tuples.append((
            os.path.join(path_to_ds, 'Samples', 'NoFX'),
            os.path.join(path_to_ds, 'annotation', 'NoFX'),
            10,
        ))
    if 11 in active_datasets:
        dir_tuples.append((
            os.path.join(data_dir, r'generated\filtered\audio'),
            os.path.join(data_dir, r'generated\filtered\annotation'),
            11,
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
            os.path.join(data_dir, r'IDMT-SMT-GUITAR_V2\dataset4\Career SG'),
            os.path.join(data_dir, r'IDMT-SMT-GUITAR_V2\dataset4\Ibanez 2820')
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
