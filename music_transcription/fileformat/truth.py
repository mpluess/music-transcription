def write_truth_file(truth_file_paths, list_of_onset_times, list_of_list_of_pitches):
    assert len(truth_file_paths) == len(list_of_onset_times)
    assert len(list_of_onset_times) == len(list_of_list_of_pitches)

    for path_to_truth_file, onset_times, list_of_pitches in zip(truth_file_paths, list_of_onset_times, list_of_list_of_pitches):
        assert len(onset_times) == len(list_of_pitches)
        with open(path_to_truth_file, 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n')
            f.write('<instrumentRecording>\n')
            f.write('  <globalParameter></globalParameter>\n')
            f.write('  <transcription>\n')
            for onset_time, pitches in zip(onset_times, list_of_pitches):
                for pitch in pitches:
                    f.write('    <event>\n')
                    f.write('      <pitch>{}</pitch>\n'.format(pitch))
                    f.write('      <onsetSec>{}</onsetSec>\n'.format(onset_time))
                    f.write('    </event>\n')
            f.write('  </transcription>\n')
            f.write('</instrumentRecording>\n')
