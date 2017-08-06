from keras.utils import plot_model
import os

from music_transcription.pitch_detection.cnn_cqt_pitch_detection import CnnCqtPitchDetector

pitch_detector = CnnCqtPitchDetector.from_zip(
    os.path.join('..', 'models', 'pitch_detection',
                 '20170718_1224_cqt_ds12391011_100-perc_optimized-params_proba-thresh-0.3.zip')
)
plot_model(pitch_detector.model, to_file='../tmp/pitch_detection.png', show_shapes=True)
