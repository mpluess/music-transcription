# this code was written inspired by this spec: http://dguitar.sourceforge.net/GP4format.html
# also tuxguitar-gtp org.herac.tuxguitar.io.gtp GP5InputStream
# and kguitar-code/convert/convertgtp.cpp

import struct
import logging
from collections import namedtuple

# logging.basicConfig(level=logging.DEBUG) #filename='example.log'


class GP5File:
    G_ENCODING = 'cp1252'
    GP5_VERSIONS = ['FICHIER GUITAR PRO v5.00', 'FICHIER GUITAR PRO v5.10']

    def __init__(self, filename):
        self.file = open(filename, 'rb')

        self.file.seek(0, 2)
        logging.debug("file size: {}".format(self.file.tell()))
        self.file.seek(0)

        # read all data
        self.__read_version()
        self.__read_header()
        self.__read_lyrics()
        self.__read_print_setup()
        self.__read_meta_info()
        self.__read_midi_channels()
        some_weird_padding = self.file.seek(42, 1)  # Some weird padding, filled with FFs
        self.nMeasures = self.read_int()
        self.nTracks = self.read_int()
        self.__read_measures()
        self.__read_tracks()
        self.file.seek(1 if self.vMinor > 0 else 2, 1)  # skip (vMinor ? 1 : 2)

        self.__read_notes()
        self.file.close()

    def read_int(self):
        return struct.unpack('<i', self.file.read(4))[0]  # <i = little-endian LSB to MSB

    def read_short(self):
        return struct.unpack('<h', self.file.read(2))[0]  # <i = little-endian LSB to MSB

    def read_byte(self):
        return struct.unpack('<b', self.file.read(1))[0]  # <i = little-endian LSB to MSB

    def read_string(self):
        s_len = max(ord(self.file.read(1)), 0)
        return str(self.file.read(s_len), self.G_ENCODING), s_len

    def read_block_string(self, block_size=0):
        if block_size == 0:
            block_size = self.read_int() - 1
            logging.debug("read int (block_size) '{}', new pos: {}".format(block_size, self.file.tell()))
        s, s_len = self.read_string()
        assert s_len <= block_size, "ERROR: s_len ({}) cannot be bigger than block_size ({})".format(s_len, block_size)
        self.file.seek(block_size - s_len, 1)
        logging.debug("read: '{}', block_size:{}, len:{}, new pos: {}".format(s, block_size, s_len, self.file.tell()))
        return s

    def read_color(self):  # rgba
        return (ord(self.file.read(1)), ord(self.file.read(1)), ord(self.file.read(1)), ord(self.file.read(1)))

    def __read_version(self):  # 1. Version
        self.version = self.read_block_string(30)
        assert self.version in self.GP5_VERSIONS, 'ERROR: invalid version "{0}"'.format(self.version)
        self.vMinor = 0 if self.version == self.GP5_VERSIONS[0] else 10

    def __read_header(self):  # 2. Song Info
        self.title = self.read_block_string()
        self.subtitle = self.read_block_string()
        self.interpret = self.read_block_string()
        self.album = self.read_block_string()
        self.author_words = self.read_block_string()
        self.author_music = self.read_block_string()  # new in .gp5
        self.copyright = self.read_block_string()
        self.tab_author = self.read_block_string()
        self.instructions = self.read_block_string()

        self.notes = ""
        self.notice_lines = self.read_int()
        logging.debug("read int (notice_lines) '{}', new pos: {}".format(self.notice_lines, self.file.tell()))
        for i in range(self.notice_lines):
            self.notes = self.notes + "\n" + self.read_block_string()

    def __read_lyrics(self):  # 3. Lyrics
        self.lyric_track = self.read_int()
        logging.debug("read int (track) '{}', new pos: {}".format(self.lyric_track, self.file.tell()))
        self.lyrics = []
        for i in range(5):  # for each of the 5 possible lines
            bar = self.read_int()
            logging.debug("read int (bar) '{}', new pos: {}".format(bar, self.file.tell()))
            block_size = self.read_int()
            logging.debug("read int (block_size) '{}', new pos: {}".format(block_size, self.file.tell()))
            s = str(self.file.read(block_size), self.G_ENCODING)
            logging.debug("read lyric '{}', block_size:{}, new pos: {}".format(s, block_size, self.file.tell()))
            self.lyrics.append((bar, s))

    # 4. other info

    def __read_print_setup(self):  # 4.1 print setup
        self.print_setup = self.file.read(49) if self.vMinor > 0 else self.file.read(30)
        self.print_setup_lines = []
        for i in range(11):
            self.print_setup_lines.append((self.file.read(4), self.read_string()))

    def __read_meta_info(self):  # 4.2 tempo, key, octave
        self.tempo = self.read_int()
        logging.debug("read int (tempo) '{}', new pos: {}".format(self.tempo, self.file.tell()))

        if self.vMinor > 0:
            self.file.seek(1, 1)  # don't know what is skipped here

        self.key = self.read_byte()  # ord(file.read(1))
        self.octave = self.read_int()  # alternative: skip 3, then read byte

    def __read_midi_channels(self):  # 4.3 Midi Channels
        self.midi_channels = []
        for i in range(64):
            self.midi_channels.append([
                self.read_int(),  # instrument
                ord(self.file.read(1)),  # volume
                ord(self.file.read(1)),  # balance
                ord(self.file.read(1)),  # chorus
                ord(self.file.read(1)),  # reverb
                ord(self.file.read(1)),  # phaser
                ord(self.file.read(1)),  # tremolo
                ord(self.file.read(1)),  # blank1
                ord(self.file.read(1))  # blank2
            ])

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

    def __read_measures(self):  # 5. measures
        self.measures = []
        for i in range(self.nMeasures):
            # init
            marker_name = majKey = minorKey = 0
            marker_color = beam8notes = (0, 0, 0, 0)

            flags = ord(self.file.read(1))

            numerator = ord(self.file.read(1)) if flags & 0x01 else 0
            denominator = ord(self.file.read(1)) if flags & 0x02 else 0
            repeat_open = ((flags & 0x04) != 0)
            repeat_close = ord(self.file.read(1)) if flags & 0x08 else 0
            # repeat_alternative = ord(file.read(1)) if flags & 0x10 else 0
            if flags & 0x20:  # 32 = bit5: Marker
                marker_name = self.read_block_string()
                marker_color = self.read_color()
            repeat_alternative = ord(self.file.read(1)) if flags & 0x10 else 0
            if flags & 0x40:  # 64 = bit6: Key change
                majKey = self.read_byte()  # ord(file.read(1))
                minorKey = self.read_byte()  # ord(file.read(1))
            double_bar = ((flags & 0x80) != 0)  # 128 = bit7: double bar?

            if flags & 0x03:  # if 1 or 2 is set (change in measure)
                beam8notes = self.read_color()  # beam8notes = file.read(4)
            if not flags & 0x10: # 16 (was) NOT set
                unknown = ord(self.file.read(1))  # unknown byte?

            triplet_feel = ord(self.file.read(1))
            unknown2 = ord(self.file.read(1))  # unknown 2

            self.measures.append(self.Measure(
                numerator, denominator, repeat_open, repeat_close, repeat_alternative,
                marker_name, marker_color, majKey, minorKey, double_bar, beam8notes, triplet_feel
            ))
            #print(numerator, denominator, repeat_open, repeat_close, repeat_alternative, marker_name, marker_color, majKey, minorKey, double_bar, beam8notes, unknown, triplet_feel, unknown2)

    Track = namedtuple("Track", [
        'name', 'nStrings', 'tuning', 'midiPort', 'channel', 'channelE', 'frets', 'capo', 'color'
    ])

    def __read_tracks(self):  # 6. Tracks
        self.tracklist = []
        for i in range(self.nTracks):
            flags = ord(self.file.read(1))

            if i > 0 and self.vMinor == 0:
                self.file.seek(1, 1)  # skip(1)

            name = self.read_block_string(40)
            nStrings = self.read_int()
            tuning = (
                self.read_int(),
                self.read_int(),
                self.read_int(),
                self.read_int(),
                self.read_int(),
                self.read_int(),
                self.read_int()
            )
            midiPort = self.read_int()
            channel = self.read_int()
            channelE = self.read_int()
            frets = self.read_int()
            capo = self.read_int()
            color = self.read_color()

            self.file.seek(49 if self.vMinor > 0 else 44, 1)  # skip (vMinor ? 49 : 44)
            if self.vMinor > 0:
                str1 = self.read_block_string()
                str2 = self.read_block_string()

            self.tracklist.append(self.Track(
                name, nStrings, tuning, midiPort, channel, channelE, frets, capo, color
            ))
            #print('"' + name + '":', nStrings, tuning, midiPort, channel, channelE, frets, capo, color)

    Beat = namedtuple("Beat", ['notes', 'duration', 'pause', 'empty', 'dotted', 'ntuple_enters', 'ntuple_times', 'chord', 'text', 'effect', 'mix_change'])

    def __read_notes(self):
        self.notes = []
        for m in range(self.nMeasures):
            self.notes.append([])
            for t in range(self.nTracks):
                self.notes[m].append(([],[]))
                for v in range(2):  # every track has two voices
                    nBeats = self.read_int()
                    #print('m:{}, track:{}, v:{}, beats:{}'.format(m,t,v,nBeats))
                    for b in range(nBeats):
                        flags = ord(self.file.read(1))
                        status = ord(self.file.read(1)) if flags & 0x40 else 0x01  # 64 = bit6: 0x00 = empty, 0x02 = rest
                        duration = self.read_byte()  # ... -1=half, 0=quarter, 1=eigth, ...
                        pause = (status == 0x02)
                        empty = (status == 0x00)
                        dotted = (flags & 0x01) != 0
                        ntuple_enters = self.read_int() if flags & 0x20 else 0  # n-tuple
                        ntuple_times = 8 if ntuple_enters > 8 else (4 if ntuple_enters > 4 else (2 if ntuple_enters > 0 else 0))  # tripplet feel
                        chord = self.__read_chord() if flags & 0x02 else self.empty_chord()  # chord diagram
                        text = self.read_block_string() if flags & 0x04 else ""  # text
                        effect = self.__read_beat_effect() if flags & 0x08 else 0  # effect
                        mix_change = self.__read_mix_change() if flags & 0x10 else None  # mix change!

                        stringFlags = ord(self.file.read(1))
                        #print('\tstring flags:', stringFlags, self.tracklist[t].nStrings)
                        notes = []
                        for i in range(6, -1, -1):  # for every string
                            if stringFlags & (1 << i) and (6 - i <= self.tracklist[t].nStrings):
                                notes.append(self.__read_note())
                                #print(notes[len(notes)-1])

                        self.file.seek(1, 1)  # skip(1)
                        skipFlag = ord(self.file.read(1))
                        if skipFlag & 0x08:
                            self.file.seek(1, 1)  # skip(1)

                        self.notes[m][t][v].append(self.Beat(
                            notes, duration, pause, empty, dotted, ntuple_enters, ntuple_times, chord, text, effect, mix_change
                        ))
                        # return (!voice.isEmpty() ? duration.getTime() : 0 );

                self.file.seek(1, 1)  # skip(1)

    Chord = namedtuple("Chord", ['base_fret', 'fretArr'])

    def empty_chord(self):
        return self.Chord(0, [-1, -1, -1, -1, -1, -1, -1])

    def __read_chord(self):
        self.file.seek(17, 1)  # skip(17)
        name = self.read_block_string(21)
        self.file.seek(4, 1)  # skip(4)
        basefret = self.read_int()
        frets = []
        for f in range(7):
            frets[f] = self.read_int()  # -1 = unplayed, 0 = no fret
            self.file.seek(32, 1)  # skip(32)
        return self.Chord(basefret, frets)

    def __read_chord_gp4style(self):
        flags = ord(self.file.read(1))  # 0x01 = gp4 chord...
        sharp = ord(self.file.read(1))
        blanks = self.file.read(3)  # gp3-compatibility
        root = self.read_byte()  # -1: custom, 0:C, 1:C#, ...
        type = ord(
            self.file.read(1))  # 0:M,1:7,2:7M,3:6,4:m,5:m7,6:m7M,7:m6,8:sus2,9:sus4,10:7sus2,11:7sus4,12:dim,13:aug,14:5
        nine_eleven_thirteen = ord(self.file.read(1))
        bass = self.read_int()
        dim_aug = self.read_int()  # 0:perfect, 1:augmented, 2:diminished
        add = self.read_byte()  # 'add'-note present in chord?
        name = self.read_block_string(21)
        blanks = self.file.read(2)  # gp3-compatibility
        fifth = self.read_byte()
        ninth = self.read_byte()  # This tonality is valid only if the value "Nine, Eleven or Thirteen" is 11 or 13.
        eleventh = self.read_byte()  # This tonality is valid only if the value "Nine, Eleven or Thirteen" is 13.
        basefret = self.read_int()
        for f in range(7):
            fret = self.read_int()  # -1 = unplayed, 0 = no fret
        nbarres = ord(self.file.read(1))
        for b in range(5):
            bar = ord(self.file.read(1))  # fret number of bar
        for b in range(5):
            bar_start = ord(self.file.read(1))  # fret number of bar
        for b in range(5):
            bar_end = ord(self.file.read(1))  # fret number of bar
        omission1 = ord(self.file.read(1))
        omission3 = ord(self.file.read(1))
        omission5 = ord(self.file.read(1))
        omission7 = ord(self.file.read(1))  # 0 = not in the chord
        omission9 = ord(self.file.read(1))  # 1 = in the chord
        omission11 = ord(self.file.read(1))
        omission13 = ord(self.file.read(1))
        blanks = self.file.read(1)  # gp3-compatibility
        for f in range(7):
            fingering = self.read_byte()  # -2 unknown, -1 X, 0: thumb, 1: index,... 4:little
        showFingering = ord(self.file.read(1))  # 1 = do display, 0 = mask
        return self.Chord(basefret, [-1, -1, -1, -1, -1, -1, -1])

    Effect = namedtuple("Effect", ['fadein', 'vibrato', 'tap_slap_pop', 'bend'])

    def empty_effect(self):
        return self.Effect(False, False, 0, self.empty_bend())

    def __read_beat_effect(self):
        flags1 = ord(self.file.read(1))
        flags2 = ord(self.file.read(1))
        fadein = flags1 & 0x10 != 0
        vibrato = flags1 & 0x02 != 0
        tap_slap_pop = ord(self.file.read(1)) if flags1 & 0x20 else 0  # 1:tap,2:slap,3:pop
        bend = self.__read_bend() if flags2 & 0x04 else self.empty_bend()
        upstroke = ord(self.file.read(1)) if flags1 & 0x40 else 0  # fastness 1 (128th) - 6 (quarters)
        downstroke = ord(self.file.read(1)) if flags1 & 0x40 else 0  # fastness 1 (128th) - 6 (quarters)
        pickstroke = ord(self.file.read(1)) if flags2 & 0x02 else 0  # (probably also not used) 1 = up, 2 = down
        return self.Effect(fadein, vibrato, tap_slap_pop, bend)

    Bend = namedtuple("Bend", ['points'])

    def empty_bend(self):
        return self.Bend([])

    def __read_bend(self):
        type = ord(self.file.read(1))  # http://dguitar.sourceforge.net/GP4format.html#Bends
        val = self.read_int()  # 100 per note (e.g. 25 = quarter note)
        nPoints = self.read_int()
        points = []
        for p in range(nPoints):
            time_pos = self.read_int()  # pos from prev point. 0-60 and is sixties of the note duration
            v = self.read_int()  # value of this point
            vibrato = ord(self.file.read(1))  # 0:none, 1:fast, 2:avg, 3:slow
            points.append((time_pos, v, vibrato))
        return self.Bend(points)

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

    def __read_mix_change(self):
        instrument = self.read_byte()  # number of new instrument. -1 = no change

        self.file.seek(16, 1)  # skip(16) (gp4 compatibility???)

        volume = self.read_byte()  # new volume. -1 = no change
        pan = self.read_byte()  # new pan. -1 = no change
        chorus = self.read_byte()  # new chorus. -1 = no change
        reverb = self.read_byte()  # new reverb. -1 = no change
        phaser = self.read_byte()  # new phaser. -1 = no change
        tremolo = self.read_byte()  # new tremolo. -1 = no change

        temponame = self.read_block_string()
        tempo = self.read_int()  # new tempo. -1 = no change

        volume_duration = self.read_byte() if volume > -1 else 0  # volume change duratin in beats.
        pan_duration = self.read_byte() if pan > -1 else 0  # pan change duratin in beats.
        chorus_duration = self.read_byte() if chorus > -1 else 0  # chorus change duratin in beats.
        reverb_duration = self.read_byte() if reverb > -1 else 0  # reverb change duratin in beats.
        phaser_duration = self.read_byte() if phaser > -1 else 0  # phaser change duratin in beats.
        tremolo_duration = self.read_byte() if tremolo > -1 else 0  # tremolo change duratin in beats.
        tempo_duration = self.read_byte() if tempo > -1 else 0  # tempo change duratin in beats.
        unused_byte_v5_1 = self.read_byte() if tempo > -1 and self.vMinor > 0 else 0  # some byte only used in versino 5.1

        flags = ord(self.file.read(1))  # changes apply only to current track?
        self.file.seek(1, 1)  # skip(1)

        str1 = self.read_block_string() if self.vMinor > 0 else ""  # ?
        str2 = self.read_block_string() if self.vMinor > 0 else ""  # ?

        return self.MixChange(
            instrument, tempo, tempo_duration, volume, volume_duration, pan, pan_duration, chorus, chorus_duration,
            reverb, reverb_duration, phaser, phaser_duration, tremolo, tremolo_duration
        )

    Note = namedtuple("Note", ['fret', 'tied', 'dead', 'ghost', 'dynamic'])

    def __read_note(self):
        flags = ord(self.file.read(1))

        accentuated = (flags & 0x40) != 0  # bit6
        type = ord(self.file.read(1)) if flags & 0x20 else 0  # normal, ghost(dead), tied
        tied = (type == 0x02)
        dead = (type == 0x03)
        dynamic = ord(self.file.read(1)) if flags & 0x10 else 6  # 1:ppp, 2:pp,3:p,4:mp,5:mf,6:?f?,7:f,8:ff,9:fff
        fret = ord(self.file.read(1)) if flags & 0x20 else 0
        fretVal = self.__get_last_played_note_on_this_string() if tied else fret  # ToDo: get last played note on this string
        fretVal = fretVal if fretVal >= 0 and fretVal < 100 else 0  # set to zero if out of bounds
        fingeringLeft = self.read_byte() if flags & 0x80 else -1  # -1=nothing, 0:thumb, 1:index,...
        fingeringRight = self.read_byte() if flags & 0x80 else -1  # -1=nothing, 0:thumb, 1:index,...

        if (flags & 0x01):  # bit1
            self.file.seek(8, 1)  # skip(8)
        self.file.seek(1, 1)  # skip(1)

        effect = self.__read_note_effect() if flags & 0x08 else self.empty_effect()  # effect
        ghost = (flags & 0x04) != 0  # bit2
        heavyAccentuated = (flags & 0x02) != 0  # bit1

        return self.Note(fretVal, tied, dead, ghost, dynamic)

    def __get_last_played_note_on_this_string(self):
        return 24  # TODO get last played note on this string

    def __read_note_effect(self):
        flags1 = ord(self.file.read(1))
        flags2 = ord(self.file.read(1))
        bend = self.__read_bend() if flags1 & 0x01 else self.empty_bend()
        grace = self.__read_grace() if flags1 & 0x10 else ()
        tremolopicking = ord(self.file.read(1)) if flags2 & 0x04 else 0  # 1=8th, 2=16th, 3=32th
        slide = ord(self.file.read(1)) if flags2 & 0x08 else 0  # tuxguitar knows only true/false and ignores the byte
        harmonic = ord(self.file.read(1)) if flags2 & 0x10 else 0  # 1=natural, 2=artificial, 3=tapped, 4=pinch, 5=semi
        trill = (ord(self.file.read(1)), ord(self.file.read(1))) if flags2 & 0x20 else (
        0, 0)  # (fret, period) period = 1=16th, 2=32th, 3=64th
        isHammer = flags1 & 0x02 != 0
        isLetRing = flags1 & 0x08 != 0
        isVibrato = flags2 & 0x40 != 0
        isPalmMute = flags2 & 0x02 != 0
        isStaccato = flags2 & 0x01 != 0
        return self.Effect(isHammer, isVibrato, harmonic, bend)

    def __read_grace(self):
        fret = ord(self.file.read(1))
        dynamic = ord(self.file.read(1))
        transition = self.read_byte()  # 0:none, 1:slide,2:bend,3:hammer
        duration = ord(self.file.read(1))
        flags = ord(self.file.read(1))
        isDead = flags & 0x01 != 0
        isOnBeat = flags & 0x02 != 0
        return (fret, dynamic, transition, duration, isDead, isOnBeat)
