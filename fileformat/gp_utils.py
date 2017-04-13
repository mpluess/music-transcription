from collections import namedtuple

G_ENCODING = 'cp1252'
GP5_VERSIONS = ['FICHIER GUITAR PRO v5.00', 'FICHIER GUITAR PRO v5.10']

Header = namedtuple("Header", [
    'title', 'subtitle', 'interpret', 'album', 'author_words', 'author_music',
    'copyright', 'tab_author', 'instructions', 'notes'
])

Lyrics = namedtuple("Lyrics", ['track', 'lines'])

Measure = namedtuple("Measure", [
    'numerator',  # Numerator of the (key) signature
    'denominator',  # Denominator of the (key) signature
    'repeat_open',  # is this a beginning of a repeat?
    'repeat_close',  # num of repeats
    'repeat_alternative',  # num of alternate ending
    'marker_name',
    'marker_color',
    'majKey',
    'minorKey',
    'double_bar',
    'beam8notes',  # "beam eight notes by" array, usually 2 - 2 - 2 - 2
    'triplet_feel'  # 0: none, 1: eights, 2: 16th
])

Track = namedtuple("Track", [
    'name', 'nStrings', 'tuning', 'midiPort', 'channel', 'channelE', 'frets', 'capo', 'color'
])

Beat = namedtuple("Beat", [
    'notes', 'duration', 'pause', 'empty', 'dotted', 'ntuple_enters', 'ntuple_times',
    'chord', 'text', 'effect', 'mix_change'
])

Chord = namedtuple("Chord", ['base_fret', 'fretArr'])

Effect = namedtuple("Effect", ['fadein', 'vibrato', 'tap_slap_pop', 'bend'])

Bend = namedtuple("Bend", ['points'])

MixChange = namedtuple('MixChange', [
    'instrument',
    'tempo', 'tempo_duration',
    'volume', 'volume_duration',
    'pan', 'pan_duration',
    'chorus', 'chorus_duration',
    'reverb', 'reverb_duration',
    'phaser', 'phaser_duration',
    'tremolo', 'tremolo_duration'
])

Note = namedtuple("Note", ['fret', 'tied', 'dead', 'ghost', 'dynamic'])


def empty_chord():
    return Chord(0, [-1, -1, -1, -1, -1, -1, -1])


def empty_effect():
    return Effect(False, False, 0, empty_bend())


def empty_bend():
    return Bend([])
