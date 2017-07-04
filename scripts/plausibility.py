import os

from music_transcription.pitch_detection.read_data import _read_onset_times_pitches
from music_transcription.string_fret_detection.plausibility import get_all_fret_possibilities

DATA_DIR = r'..\data'
onset_group_threshold_seconds = 0.05

'''
[[0, 8, 5, 7, -1, -1],
 [0, 8, -1, 10, 12, -1],
 [0, -1, 12, 10, 12, -1],
 [-1, 8, 9, 10, 12, -1],
 [0, -1, 12, -1, 12, 20],
 [0, -1, -1, 17, 12, 20],
 [-1, -1, 12, 14, 12, 20],
 [0, -1, 12, -1, 15, 17],
 [0, -1, -1, 17, 15, 17],
 [-1, -1, 12, 14, 15, 17]]
 
 [[-1, 8, 9, 10, 12, -1], [-1, -1, 12, 14, 12, 20], [-1, -1, 12, 14, 15, 17]]  36.369 /  0.059463 / 140
 [[-1, 8, 9, 10, 12, -1], [3, 5, 5, 7, -1, -1],     [-1, -1, 12, 14, 15, 17]]  33 / 0.03 / 135

'''

# get all xml files of dataset 7
xml_files = []
path_to_ds = os.path.join(DATA_DIR, 'IDMT-SMT-AUDIO-EFFECTS', 'Gitarre polyphon', 'annotation')
for effect_desc in os.listdir(path_to_ds):
    xml_dir = os.path.join(path_to_ds, effect_desc)
    for xml_file in os.listdir(xml_dir):
        xml_files.append(os.path.join(xml_dir, xml_file))

notes_set = set()
for xml in xml_files[-30:]:
    _, pitches = _read_onset_times_pitches(xml, 40, 88, 7, onset_group_threshold_seconds)
    notes_set.add(tuple(sorted(pitches[0])))

for n in notes_set:
    print(n, '->', get_all_fret_possibilities(n))
