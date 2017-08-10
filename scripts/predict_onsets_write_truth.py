from os import listdir
from os.path import isdir, isfile, join
from warnings import warn

from music_transcription.onset_detection.cnn_onset_detection import CnnOnsetDetector

onset_detector = CnnOnsetDetector.from_zip('../models/onset_detection/ds1-4_100-perc.zip')
root_dir = '../data/IDMT-SMT-AUDIO-EFFECTS'
for dataset_part in listdir(root_dir):
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
                        onset_times = onset_detector.predict_onsets(path_to_wav_file)
                        with open(join(path_to_dataset_part, 'Onsets', fx_folder, wav_file[:-4] + '.txt'), 'w') as f:
                            if len(onset_times) == 0:
                                warn('No onset detected in file {}'.format(path_to_wav_file))
                            else:
                                if len(onset_times) > 1:
                                    warn('More than one onset detected in file {}: {}'.format(path_to_wav_file, onset_times))
                                f.write('{}\n'.format(onset_times[0]))
                    else:
                        warn('Not a WAV file: {}'.format(path_to_wav_file))
            else:
                warn('Not an FX folder: {}'.format(path_to_fx_folder))
        print('')
