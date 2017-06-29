from sklearn.model_selection import train_test_split

from music_transcription.pitch_detection.cnn_pitch_detection import CnnPitchDetector
from music_transcription.pitch_detection.cnn_cqt_pitch_detection import CnnCqtPitchDetector
from music_transcription.pitch_detection.read_data import get_wav_and_truth_files

# active_datasets = {1, 2, 3}
# active_datasets = {6}
# active_datasets = {1, 2, 3, 6}
active_datasets = {7}

wav_file_paths, truth_dataset_format_tuples = get_wav_and_truth_files(active_datasets)

(
    wav_file_paths_train, wav_file_paths_test,
    truth_dataset_format_tuples_train, truth_dataset_format_tuples_test
) = train_test_split(
    wav_file_paths, truth_dataset_format_tuples, test_size=0.2, random_state=42
)

# pitch_detector = CnnPitchDetector()
pitch_detector = CnnCqtPitchDetector()
# pitch_detector = CnnCqtPitchDetector(cqt_configs=[
#     {
#         'hop_length': 256,
#         'fmin': 55.0,
#         'n_bins': 60,
#         'bins_per_octave': 12,
#         'scale': False,
#     },
#     {
#         'hop_length': 512,
#         'fmin': 55.0,
#         'n_bins': 180,
#         'bins_per_octave': 36,
#         'scale': False,
#     },
#     {
#         'hop_length': 1024,
#         'fmin': 55.0,
#         'n_bins': 300,
#         'bins_per_octave': 60,
#         'scale': False,
#     },
# ])
pitch_detector.fit(
    wav_file_paths_train, truth_dataset_format_tuples_train,
    wav_file_paths_test, truth_dataset_format_tuples_test,
)
pitch_detector.save('../models/pitch_detection/20170629_1443_cqt_audio_effects_poly_80-perc_onset-group-thresh-0.05.zip')
