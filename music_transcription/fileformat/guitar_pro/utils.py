from collections import namedtuple

G_ENCODING = 'cp1252'
GP5_VERSIONS = ['FICHIER GUITAR PRO v5.00', 'FICHIER GUITAR PRO v5.10']
GP_PAGE_SETUP_LINES = [
    "%TITLE%",
    "%SUBTITLE%",
    "%ARTIST%",
    "%ALBUM%",
    "Words by %WORDS%",
    "Music by %MUSIC%",
    "Words & Music by %WORDSMUSIC%",
    "Copyright %COPYRIGHT%",
    "All Rights Reserved - International Copyright Secured",
    "Page %N%/%P%",
    "Moderate"
]

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
    'marker_name',  # String
    'marker_color',  # 4-tuple
    'majKey',
    'minorKey',
    'double_bar',
    'beam8notes',  # "beam eight notes by" array, usually 2 - 2 - 2 - 2
    'triplet_feel'  # 0: none, 1: eights, 2: 16th
])

Track = namedtuple("Track", [
    'name', 'nStrings', 'tuning', 'midiPort', 'channel', 'channelE', 'frets', 'capo', 'color', 'instrument'
])


def beat(notes, duration=0, pause=False, empty=False, dotted=False,
         ntuple_feel=(0, 0), chord=None, text="", effect=None, mix_change=None):
    return Beat(notes, duration, pause, empty, dotted, ntuple_feel, chord, text, effect, mix_change)

Beat = namedtuple("Beat", [
    'notes', 'duration', 'pause', 'empty', 'dotted', 'ntuple_feel', 'chord', 'text', 'effect', 'mix_change'
])

Chord = namedtuple("Chord", ['name', 'base_fret', 'fretArr'])

BeatEffect = namedtuple("BeatEffect", ['fadein', 'vibrato', 'tap_slap_pop', 'bend'])

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


def note(fret=0, tied=False, dead=False, ghost=False, dynamic=6, acc=False, heavy_acc=False, effect=None):
    return Note(fret, tied, dead, ghost, dynamic, acc, heavy_acc, effect)

Note = namedtuple("Note", [
    'fret', 'tied', 'dead', 'ghost', 'dynamic', 'accentuated', 'heavy_accentuated', 'effect'
])

NoteEffect = namedtuple("NoteEffect", [
    'is_hammer', 'is_let_ring', 'is_vibrato', 'is_palm_mute', 'is_staccato',
    'tremolo_picking', 'slide', 'harmonic', 'trill', 'bend', 'grace'
])

Grace = namedtuple("Grace", [
    'fret', 'dynamic', 'transition', 'duration', 'is_dead', 'is_on_beat'
])


def empty_chord():
    return Chord(0, [-1, -1, -1, -1, -1, -1, -1])


def empty_beat_effect():
    return BeatEffect(False, False, 0, empty_bend())


def empty_bend():
    return Bend([])


# for the most common signatures, this method gives some possible beam8notes value
def calc_beam8notes(numerator, denominator):
    while denominator < 8:
        denominator *= 2
        numerator *= 2
    if numerator > 12 or denominator > 8:
        return 0, 0, 0, 0

    b8n = [0, 0, 0, 0]
    total = 0
    if numerator <= 8:
        for i in range(4):
            b8n[i] = 2 if numerator - total >= 2 else numerator - total
            total += b8n[i]
    else:
        b8n = [2, 2, 2, 2]
        total = 8
        for i in range(numerator - total):
            b8n[i] += 1

    return b8n[0], b8n[1], b8n[2], b8n[3]

