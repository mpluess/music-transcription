from keras.layers import Activation, Concatenate, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Input, Model
from keras.utils import plot_model
import os

from music_transcription.pitch_detection.cnn_cqt_pitch_detection import CnnCqtPitchDetector

# pitch_detector = CnnCqtPitchDetector.from_zip(
#     os.path.join('..', 'models', 'pitch_detection', 'cqt_ds12391011_100-perc_proba-thresh-0.3.zip')
# )
# plot_model(pitch_detector.model, to_file='../tmp/pitch_detection.png', show_shapes=True)

inputs = []
conv_blocks = []

spectrogram = Input(shape=(1, 36, 180))
inputs.append(spectrogram)

# 10 small filters finding local patterns applicable to different pitches
conv = Conv2D(10, (10, 3), padding='valid')(spectrogram)
conv = MaxPooling2D(pool_size=(6, 3))(conv)
conv = Flatten()(conv)
conv_blocks.append(conv)

# 512 filters spanning the whole pitch bandwidth of the guitar
conv = Conv2D(512, (10, 180), strides=(5, 1), padding='valid')(spectrogram)
conv = MaxPooling2D(pool_size=(2, 1))(conv)
conv = Flatten()(conv)
conv_blocks.append(conv)

# Concatenate convolutional blocks and feed them to a feed forward NN with one hidden layer.
z = Concatenate()(conv_blocks)
z = Dense(256)(z)
output = Dense(49, activation='sigmoid')(z)

model = Model(inputs, output)
model.compile(loss='binary_crossentropy', optimizer='adam')
model.summary()
plot_model(model, to_file='../tmp/pitch_detection_minimal.png', show_shapes=True)
