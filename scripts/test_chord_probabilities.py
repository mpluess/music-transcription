import os

from music_transcription.beat_transformation.simple_beat_transformation import SimpleBeatTransformer
from music_transcription.fileformat.guitar_pro.gp5_writer import write_gp5
from music_transcription.fileformat.guitar_pro.utils import Measure, Track
from music_transcription.onset_detection.read_data import read_onset_times
from music_transcription.pitch_detection.read_data import _read_onset_times_pitches
from music_transcription.string_fret_detection.sequence_string_fret_detection import SequenceStringFretDetection

# path_to_wav = r'..\data\recordings\audio\instrumental_lead.wav'
# path_to_truth = r'..\data\recordings\annotation\instrumental_lead.xml'
# tempo = 144
# shortest_note = 0.5  # 1/8

path_to_wav = r'..\data\recordings\audio\mim-riff1-short-slow.wav'
path_to_truth = r'..\data\recordings\annotation\mim-riff1-short-slow.xml'
tempo = 88
shortest_note = 0.25  # 1/16

tuning = (64, 59, 55, 50, 45, 40)
n_frets = 24

onset_times_seconds = read_onset_times(path_to_truth, 1, 'xml', 0.05)
onset_times_grouped, list_of_pitch_sets = _read_onset_times_pitches(path_to_truth, 40, 88, 1, 0.05)
assert onset_times_seconds == onset_times_grouped

string_fret_detector = SequenceStringFretDetection(tuning, n_frets)
list_of_string_lists, list_of_fret_lists = string_fret_detector.predict_strings_and_frets(
    None, onset_times_seconds, list_of_pitch_sets
)

beat_transformer = SimpleBeatTransformer(shortest_note=shortest_note, beats_per_measure=float(4.0))
beats = beat_transformer.transform(path_to_wav, onset_times_seconds,
                                   list_of_string_lists, list_of_fret_lists, tempo)

measures = []
for i, measure in enumerate(beats):
    if i == 0:
        measures.append(Measure(4, 4, beam8notes=(2, 2, 2, 2)))
    else:
        measures.append(Measure())

tracks = [Track("Electric Guitar", len(tuning), tuning + (-1,), 1, 1, 2, n_frets, 0, (200, 55, 55, 0), 25)]

recording_name = os.path.basename(path_to_wav).rstrip('.wav')
path_to_gp5 = recording_name + '.gp5'  # default: recording_name.gp5 in cwd

write_gp5(measures, tracks, beats, tempo=tempo, outfile=path_to_gp5)
