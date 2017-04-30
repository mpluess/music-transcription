# import numpy as np

from onset_detection.onset_detectors import CnnOnsetDetector

wav_file_paths = [r'data\IDMT-SMT-GUITAR_V2\dataset2\audio\AR_Lick5_FN.wav']

onset_detector = CnnOnsetDetector.from_zip('models/20170430_1-channel_ds2_adjusted-labels.zip')
y_predicted = onset_detector.predict(wav_file_paths)
frame_rate_hz = onset_detector.feature_extractor.frame_rate_hz

# y_predicted = np.array([0, 1, 0, 0, 0, 1], dtype=np.int8)
# frame_rate_hz = 100

onset_times_seconds = [index / frame_rate_hz for index in y_predicted.nonzero()]
print(onset_times_seconds)
