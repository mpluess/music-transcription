# determine

import os
from music_transcription.pitch_detection.read_data import _read_onset_times_pitches

DATA_DIR = r'..\..\data'
MAX_FRET = 24
MAX_DISTANCE_MM = 135  # max finger spread distance  # 140
FIRST_FRET_WIDTH = 33  # 36.369
FRET_WIDTH_DECREASE = 1.03  # 1.05946263
onset_group_threshold_seconds = 0.03


def get_all_fret_possibilities(notes, tuning=(64, 59, 55, 50, 45, 40)):
    solutions = [[-1] * len(tuning)]
    for note in sorted(notes):
        print(solutions)
        new_solutions = []
        for sol in solutions:
            for i in range(len(tuning)):
                if sol[i] == -1 and tuning[i] <= note <= tuning[i] + MAX_FRET:
                    new_sol = sol.copy()
                    new_sol[i] = note - tuning[i]

                    min_fret = max_fret = max(new_sol)
                    if max_fret > 0:
                        for ns in new_sol:  # get min fret but without empty strings!
                            if 0 < ns < min_fret:
                                min_fret = ns

                        # calculate finger spread width in mm (formula for 25.5 inch scale (648mm))
                        distance_mm = 0
                        for f in range(min_fret, max_fret):
                            distance_mm += FIRST_FRET_WIDTH / FRET_WIDTH_DECREASE ** f
                        if distance_mm > MAX_DISTANCE_MM:  # skip if too far apart!
                            continue

                        # check for non-played strings in the middle
                        min_idx = 0
                        while new_sol[min_idx] == -1 and min_idx < len(new_sol):
                            min_idx += 1
                        max_idx = len(new_sol)
                        while new_sol[max_idx-1] == -1 and max_idx > 0:
                            max_idx -= 1
                        c_none = new_sol[min_idx:max_idx].count(-1)
                        # c_empty = new_sol[min_idx:max_idx].count(0)

                        # skip if more than one none-played string or if better solutions are available
                        if c_none > 1 or (c_none > 0 and len(new_solutions) > 0):
                            continue

                    new_solutions.append(new_sol)
        solutions = new_solutions
    return solutions


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

for xml in xml_files[-30:]:
    _, pitches = _read_onset_times_pitches(xml, 40, 88, 7, onset_group_threshold_seconds)
    print(sorted(pitches[0]))