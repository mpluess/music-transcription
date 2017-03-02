import numpy as np
import soundfile
from xml.etree import ElementTree


def read_X(path_to_wav):
    # scipy.io.wavfile is not able to read 24-bit data hence the need to change this alternative library
    samples, sample_rate = soundfile.read(path_to_wav)
    


# TODO check that this is perfectly in sync with the actual onset to learn the right thing.
def read_y(path_to_xml, frame_rate_hz, end_time_s):
    tree = ElementTree.parse(path_to_xml)
    root = tree.getroot()
    y = np.zeros(int(frame_rate_hz * end_time_s), dtype=np.int8)
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

    return y

read_X(r'D:\Users\Michel\Documents\FH\module\8_IP6\input\IDMT-SMT-GUITAR_V2\dataset2\audio\AR_Lick1_FN.wav')
# print(read_y(
#     r'D:\Users\Michel\Documents\FH\module\8_IP6\input\IDMT-SMT-GUITAR_V2\dataset2\annotation\AR_Lick1_FN.xml',
#     frame_rate_hz=100,
#     end_time_s=10.0
# ))
