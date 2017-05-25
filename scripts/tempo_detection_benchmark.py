import numpy as np
import matplotlib.pyplot as plt
from music_transcription.beat_conversion.simple_beat_conversion import SimpleBeatConverter
from music_transcription.onset_detection.cnn_onset_detection import CnnOnsetDetector
from music_transcription.pitch_detection.read_data import get_wav_and_truth_files

# CONFIG
DATA_DIR = r'..\data'

onset_detector = CnnOnsetDetector.from_zip(
    '../models/onset_detection/20170511-3-channels_ds1-4_80-perc_adjusted-labels.zip')

beat_converter = SimpleBeatConverter()

active_datasets = {4}
wav_file_paths, truth_dataset_format_tuples = get_wav_and_truth_files(active_datasets)

triples = []
for wav_path in wav_file_paths:
    filename = wav_path.split('\\').pop().split('BPM.')[0]
    bpmstr = filename.split('_').pop()
    bpm_truth = int(bpmstr)

    onset_times_seconds = onset_detector.predict_onsets(wav_path)

    bpm_aubio = beat_converter.determine_pitch_aubio(wav_path)
    bpm_onset_times = beat_converter.determine_pitch_from_onsets(onset_times_seconds)
    triples.append((bpm_truth, bpm_aubio, bpm_onset_times))

triples.sort()
bpm_gt = []
bpm_a = []
bpm_o = []
for gt, a, o in triples:
    bpm_gt.append(gt)
    bpm_a.append(a)
    bpm_o.append(o)

print(bpm_gt)
print('')
print(bpm_a)
print('')
print(bpm_o)
print('')
gt = np.array(bpm_gt)
diff_a = np.array(bpm_a) - gt
diff_o = np.array(bpm_o) - gt
print(diff_a)
print('')
print(diff_o)

x_ax = range(len(bpm_gt))
plt.plot(x_ax, bpm_gt, '-g', x_ax, bpm_a, '-r', x_ax, bpm_o, '-b')
plt.show()

diff_a.sort()
diff_o.sort()

plt.figure()
plt.scatter(x_ax, diff_a, c='red', alpha=0.5)
plt.scatter(range(1000, 1000+len(bpm_gt)), diff_o, c='blue', alpha=0.5)
plt.show()
