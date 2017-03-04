import numpy as np
from os import listdir
import os.path
from os.path import isfile
import soundfile
from xml.etree import ElementTree
from warnings import warn


def read_X(path_to_wav, frame_rate_hz):
    # scipy.io.wavfile is not able to read 24-bit data hence the need to change this alternative library
    samples, sample_rate = soundfile.read(path_to_wav)
    if sample_rate % frame_rate_hz != 0:
        raise ValueError('Sample rate ' + str(sample_rate) + ' % frame rate ' + str(frame_rate_hz) + ' != 0')
    samples_per_frame = int(sample_rate / frame_rate_hz)
    offset = 0
    # TODO subsampling
    X = []
    # Cut off last samples
    while offset <= len(samples) - samples_per_frame:
        X.append(samples[offset:offset + samples_per_frame])
        offset += samples_per_frame

    X = np.array(X)
    return X, X.size / sample_rate


# TODO check that this is perfectly in sync with the actual onset to learn the right thing.
def read_y(path_to_xml, frame_rate_hz, length_seconds):
    tree = ElementTree.parse(path_to_xml)
    root = tree.getroot()
    y = np.zeros(int(frame_rate_hz * length_seconds), dtype=np.int8)
    for root_child in root:
        if root_child.tag == 'transcription':
            for event in root_child:
                if event.tag != 'event':
                    raise ValueError('Unexpected XML element, expected event, got ' + event.tag)
                for event_child in event:
                    if event_child.tag == 'onsetSec':
                        index = int(float(event_child.text) * frame_rate_hz)
                        y[index] = 1
            break

    y = np.reshape(y, (-1, 1))
    return y

# TODO dataset einchecken, damit aenderungen zentral gemacht werden (anderes, privates repo)
frame_rate_hz = 100
data_dir = 'data/IDMT-SMT-GUITAR_V2/dataset2'
audio_dir = os.path.join(data_dir, 'audio')
annotation_dir = os.path.join(data_dir, 'annotation')
X = None
y = None
for wav_file in listdir(audio_dir):
    if wav_file.endswith('.wav'):
        X_part, length_seconds = read_X(os.path.join(audio_dir, wav_file), frame_rate_hz)
        path_to_xml = os.path.join(annotation_dir, wav_file.replace('.wav', '.xml'))
        if isfile(path_to_xml):
            y_part = read_y(path_to_xml, frame_rate_hz, length_seconds)
            # TODO is there a faster way than this vstacking?
            if X is None:
                X = X_part
            else:
                X = np.vstack((X, X_part))
            if y is None:
                y = y_part
            else:
                y = np.vstack((y, y_part))
        else:
            warn('No truth found for ' + wav_file + ', skipping file.')

print(X.shape)
print(y.shape)
