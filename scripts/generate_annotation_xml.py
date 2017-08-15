""" create a skeleton XML file for labelling with the onsets predicted by our onset detector """

from xml.etree import ElementTree
from xml.dom import minidom

from music_transcription.onset_detection.cnn_onset_detection import CnnOnsetDetector


def _prettify(elem):
    """Return a pretty-printed XML string for the Element. """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def write_xml(path_to_xml, onsets):
    root = ElementTree.Element('instrumentRecording')

    gp = ElementTree.SubElement(root, 'globalParameter')
    afn = ElementTree.SubElement(gp, 'audioFileName')
    afn.text = path_to_xml.split('/').pop().split('\\').pop().replace('.wav', '.xml')
    inst = ElementTree.SubElement(gp, 'instrument')
    inst.text = 'EGUI'
    im = ElementTree.SubElement(gp, 'instrumentModel')
    im.text = 'Dean B Zelinsky'
    pus = ElementTree.SubElement(gp, 'pickUpSetting')
    pus.text = 'unknown'
    it = ElementTree.SubElement(gp, 'instrumentTuning')
    it.text = '40  45  50  55  59  64'
    rd = ElementTree.SubElement(gp, 'recordingDate')
    rd.text = '17-07-03'
    ra = ElementTree.SubElement(gp, 'recordingArtist')
    ra.text = 'Silvan Laube'
    ibm = ElementTree.SubElement(gp, 'instrumentBodyMaterial')
    ibm.text = 'unknown'
    ism = ElementTree.SubElement(gp, 'instrumentStringMaterial')
    ism.text = 'Steel'
    comp = ElementTree.SubElement(gp, 'composer')
    comp.text = 'Silvan Laube / Lotrify'

    transcription = ElementTree.SubElement(root, 'transcription')

    for onset in onsets:
        event = ElementTree.SubElement(transcription, 'event')
        on = ElementTree.SubElement(event, 'onsetSec')
        on.text = str(onset)
        pitch = ElementTree.SubElement(event, 'pitch')
        pitch.text = '42'  # default!
        # off = ElementTree.SubElement(event, 'offsetSec')
        # off.text = str(onset)
        fret = ElementTree.SubElement(event, 'fretNumber')
        fret.text = '2'
        string = ElementTree.SubElement(event, 'stringNumber')
        string.text = '1'

    f = open(path_to_xml, 'w')
    f.write(_prettify(root))

    return


# CONFIG
wav_path = r'..\data\recordings\audio\instrumental_rythm2.wav'
xml_path = r'..\data\recordings\annotation\instrumental_rythm2.xml'

onset_detector = CnnOnsetDetector.from_zip('../models/onset_detection/ds1-4_100-perc.zip')
onset_times_seconds = onset_detector.predict_onsets(wav_path)
write_xml(xml_path, onset_times_seconds)
