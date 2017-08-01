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
 
 
 
(57, 62, 64) -> [[0, 3, 2, -1, -1, -1], [-1, 5, 7, 7, -1, -1], [-1, -1, 9, 12, 12, -1], [-1, -1, -1, 14, 17, 17]]
(56, 60, 63, 67) -> [[3, 4, 5, 6, -1, -1], [-1, 8, 8, 10, 11, -1], [-1, -1, 12, 13, 15, 16]]
(57, 61, 64, 68) -> [[4, 5, 6, 7, -1, -1], [-1, 9, 9, 11, 12, -1], [-1, -1, 13, 14, 16, 17]]
(57, 61, 64) -> [[0, 2, 2, -1, -1, -1], [-1, 5, 6, 7, -1, -1], [-1, -1, 9, 11, 12, -1], [-1, -1, -1, 14, 16, 17]]
(57, 61, 64, 67) -> [[3, 5, 6, 7, -1, -1], [-1, 8, 9, 11, 12, -1], [-1, -1, 12, 14, 16, 17]]
(57, 64, 69) -> [[5, 5, 2, -1, -1, -1], [-1, 10, 9, 7, -1, -1], [-1, -1, 14, 14, 12, -1], [-1, -1, -1, 19, 19, 17]]
(57, 60, 64) -> [[0, 1, 2, -1, -1, -1], [-1, 5, 5, 7, -1, -1], [-1, -1, 9, 10, 12, -1], [-1, -1, -1, 14, 15, 17]]
(56, 59, 63, 66) -> [[2, 4, 4, 6, -1, -1], [-1, 7, 8, 9, 11, -1], [-1, -1, 11, 13, 14, 16]]
(57, 60, 64, 67) -> [[3, 5, 5, 7, -1, -1], [-1, 8, 9, 10, 12, -1], [-1, -1, 12, 14, 15, 17]]
(56, 60, 63, 66) -> [[2, 4, 5, 6, -1, -1], [-1, 7, 8, 10, 11, -1], [-1, -1, 11, 13, 15, 16]]

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
