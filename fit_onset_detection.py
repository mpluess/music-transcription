from onset_detection.onset_detectors import CnnFeatureExtractor
from onset_detection.read_data import get_wav_and_truth_files

wav_file_paths, truth_dataset_format_tuples = get_wav_and_truth_files({1})
cfe = CnnFeatureExtractor()
X = cfe.fit_transform(wav_file_paths)
