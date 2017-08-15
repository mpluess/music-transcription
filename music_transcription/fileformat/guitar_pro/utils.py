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


class Header:
    def __init__(self, title='', subtitle='', interpret='', album='', author_words='', author_music='', copyright='',
                 tab_author='', instructions='', notes=''):
        self.title = title
        self.subtitle = subtitle
        self.interpret = interpret
        self.album = album
        self.author_words = author_words
        self.author_music = author_music
        self.copyright = copyright
        self.tab_author = tab_author
        self.instructions = instructions
        self.notes = notes


class Lyrics:
    def __init__(self, track, lines):
        self.track = track
        self.lines = lines


class Measure:
    def __init__(self, numerator=0, denominator=0, repeat_open=False, repeat_close=0, repeat_alternative=0,
                 marker_name="", marker_color=(0, 0, 0, 0), majKey=0, minorKey=0, double_bar=False,
                 beam8notes=None, triplet_feel=0):
        self.numerator = numerator  # Numerator of the (key) signature
        self.denominator = denominator  # Denominator of the (key) signature
        self.repeat_open = repeat_open  # is this a beginning of a repeat?
        self.repeat_close = repeat_close  # num of repeats
        self.repeat_alternative = repeat_alternative  # num of alternate ending
        self.marker_name = marker_name  # String
        self.marker_color = marker_color  # 4-tuple
        self.majKey = majKey
        self.minorKey = minorKey
        self.double_bar = double_bar
        self.beam8notes = beam8notes  # "beam eight notes by" array, usually 2 - 2 - 2 - 2
        self.triplet_feel = triplet_feel  # 0: none, 1: eights, 2: 16th


class Track:
    def __init__(self, name="track 1", nStrings=6, tuning=(64, 59, 55, 50, 45, 40, -1),
                 midiPort=1, channel=1, channelE=2, frets=24, capo=0, color=(255, 0, 0, 0), instrument=25):
        self.name = name
        self.nStrings = nStrings
        self.tuning = tuning
        self.midiPort = midiPort
        self.channel = channel
        self.channelE = channelE
        self.frets = frets
        self.capo = capo
        self.color = color
        self.instrument = instrument


class Beat:
    def __init__(self, notes, duration=0, pause=False, empty=False, dotted=False,
                 ntuple_feel=(0, 0), chord=None, text="", effect=None, mix_change=None):
        self.notes = notes
        self.duration = duration
        self.pause = pause
        self.empty = empty
        self.dotted = dotted
        self.ntuple_feel = ntuple_feel
        self.chord = chord
        self.text = text
        self.effect = effect
        self.mix_change = mix_change

    def __eq__(self, other):
        if other is None:
            return False
        for i in range(len(self.notes)):
            if self.notes[i] != other.notes[i]:
                return False
        return (self.duration == other.duration and
                self.pause == other.pause and
                self.empty == other.empty and
                self.dotted == other.dotted and
                self.ntuple_feel == other.ntuple_feel and
                self.chord == other.chord and
                self.text == other.text and
                self.effect == other.effect and
                self.mix_change == other.mix_change)

    def __ne__(self, other):
        return not self.__eq__(other)


class Chord:
    def __init__(self, name="", base_fret=0, fretArr=None):
        self.name = name
        self.base_fret = base_fret
        self.fretArr = [-1, -1, -1, -1, -1, -1, -1] if fretArr is None else fretArr

    def __eq__(self, other):
        return (other is not None and
                self.name == other.name and
                self.base_fret == other.base_fret and
                self.fretArr == other.fretArr)

    def __ne__(self, other):
        return not self.__eq__(other)


class Bend:
    def __init__(self, points=None):
        self.points = [] if points is None else points

    def __eq__(self, other):
        return other is not None and self.points == other.points

    def __ne__(self, other):
        return not self.__eq__(other)


class BeatEffect:
    def __init__(self, fadein=False, vibrato=False, tap_slap_pop=0, bend=Bend()):
        self.fadein = fadein
        self.vibrato = vibrato
        self.tap_slap_pop = tap_slap_pop
        self.bend = bend

    def __eq__(self, other):
        return (other is not None and
                self.fadein == other.fadein and
                self.vibrato == other.vibrato and
                self.tap_slap_pop == other.tap_slap_pop and
                self.bend == other.bend)

    def __ne__(self, other):
        return not self.__eq__(other)


class MixChange:
    def __init__(self, instrument, tempo, tempo_duration, volume, volume_duration, pan, pan_duration,
                 chorus, chorus_duration, reverb, reverb_duration, phaser, phaser_duration, tremolo, tremolo_duration):
        self.instrument = instrument
        self.tempo = tempo
        self.tempo_duration = tempo_duration
        self.volume = volume
        self.volume_duration = volume_duration
        self.pan = pan
        self.pan_duration = pan_duration
        self.chorus = chorus
        self.chorus_duration = chorus_duration
        self.reverb = reverb
        self.reverb_duration = reverb_duration
        self.phaser = phaser
        self.phaser_duration = phaser_duration
        self.tremolo = tremolo
        self.tremolo_duration = tremolo_duration

    def __eq__(self, other):
        return (other is not None and
                self.instrument == other.instrument and
                self.tempo == other.tempo and
                self.tempo_duration == other.tempo_duration and
                self.volume == other.volume and
                self.volume_duration == other.volume_duration and
                self.pan == other.pan and
                self.pan_duration == other.pan_duration and
                self.chorus == other.chorus and
                self.chorus_duration == other.chorus_duration and
                self.reverb == other.reverb and
                self.reverb_duration == other.reverb_duration and
                self.phaser == other.phaser and
                self.phaser_duration == other.phaser_duration and
                self.tremolo == other.tremolo and
                self.tremolo_duration == other.tremolo_duration)

    def __ne__(self, other):
        return not self.__eq__(other)


class Note:
    def __init__(self, fret=0, tied=False, dead=False, ghost=False, dynamic=6,
                 accentuated=False, heavy_accentuated=False, effect=None):
        self.fret = fret
        self.tied = tied
        self.dead = dead
        self.ghost = ghost
        self.dynamic = dynamic
        self.accentuated = accentuated
        self.heavy_accentuated = heavy_accentuated
        self.effect = effect

    def __eq__(self, other):
        return (other is not None and
                self.fret == other.fret and
                self.tied == other.tied and
                self.dead == other.dead and
                self.ghost == other.ghost and
                self.dynamic == other.dynamic and
                self.accentuated == other.accentuated and
                self.heavy_accentuated == other.heavy_accentuated and
                self.effect == other.effect)

    def __ne__(self, other):
        return not self.__eq__(other)


class NoteEffect:
    def __init__(self, is_hammer, is_let_ring, is_vibrato, is_palm_mute, is_staccato, tremolo_picking,
                 slide, harmonic, trill, bend, grace):
        self.is_hammer = is_hammer
        self.is_let_ring = is_let_ring
        self.is_vibrato = is_vibrato
        self.is_palm_mute = is_palm_mute
        self.is_staccato = is_staccato
        self.tremolo_picking = tremolo_picking
        self.slide = slide
        self.harmonic = harmonic
        self.trill = trill
        self.bend = bend
        self.grace = grace

    def __eq__(self, other):
        return (other is not None and
                self.is_hammer == other.is_hammer and
                self.is_let_ring == other.is_let_ring and
                self.is_vibrato == other.is_vibrato and
                self.is_palm_mute == other.is_palm_mute and
                self.is_staccato == other.is_staccato and
                self.tremolo_picking == other.tremolo_picking and
                self.slide == other.slide and
                self.harmonic == other.harmonic and
                self.trill == other.trill and
                self.bend == other.bend and
                self.grace == other.grace)

    def __ne__(self, other):
        return not self.__eq__(other)


class Grace:
    def __init__(self, fret, dynamic, transition, duration, is_dead, is_on_beat):
        self.fret = fret
        self.dynamic = dynamic
        self.transition = transition
        self.duration = duration
        self.is_dead = is_dead
        self.is_on_beat = is_on_beat

    def __eq__(self, other):
        return (other is not None and
                self.fret == other.fret and
                self.dynamic == other.dynamic and
                self.transition == other.transition and
                self.duration == other.duration and
                self.is_dead == other.is_dead and
                self.is_on_beat == other.is_on_beat)

    def __ne__(self, other):
        return not self.__eq__(other)


def calc_beam8notes(numerator, denominator):
    """ for the most common signatures, this method gives some possible beam8notes value """
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
