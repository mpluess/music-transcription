from collections import defaultdict

from music_transcription.pitch_detection.read_data import get_wav_and_truth_files, _read_onset_times_pitches


def count_pitches(active_datasets):
    wav_file_paths, truth_dataset_format_tuples = get_wav_and_truth_files(active_datasets)
    pitches = defaultdict(int)
    for path_to_xml, dataset, truth_format in truth_dataset_format_tuples:
        if truth_format != 'xml':
            raise ValueError('Unsupported format {}'.format(truth_format))
        _, pitches_grouped = _read_onset_times_pitches(path_to_xml, 40, 88, dataset, 0.05)
        if pitches_grouped is not None:
            for pitch_group in pitches_grouped:
                for pitch in pitch_group:
                    pitches[pitch] += 1

    return pitches


def print_pitches(pitches):
    print('PITCHES')
    print('min pitch = {}'.format(min(pitches.keys())))
    print('max pitch = {}'.format(max(pitches.keys())))
    print('nr of pitches = {}'.format(sum(pitches.values())))
    for pitch, count in sorted(pitches.items()):
        print('{}: {}'.format(pitch, count))

pitches_overall = defaultdict(int)
for active_datasets in [{1, 2, 3}, {9}, {10}, {11}]:
    pitches = count_pitches(active_datasets)
    print(active_datasets)
    print_pitches(pitches)
    for pitch, count in pitches.items():
        pitches_overall[pitch] += count

print('Overall')
print_pitches(pitches_overall)
