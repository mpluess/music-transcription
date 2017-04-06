# this code was written inspired by this spec: http://dguitar.sourceforge.net/GP4format.html
# also tuxguitar-gtp org.herac.tuxguitar.io.gtp GP5InputStream
# and kguitar-code/convert/convertgtp.cpp (especially for changes between gp4 and gp5)

import struct
import logging
from collections import namedtuple

#logging.basicConfig(level=logging.DEBUG) #filename='example.log'
G_ENCODING = 'cp1252'

def read_int(file):
    return struct.unpack('<i', file.read(4))[0] # <i = little-endian LSB to MSB

def read_short(file):
    return struct.unpack('<h', file.read(2))[0]  # <i = little-endian LSB to MSB

def read_byte(file):
    return struct.unpack('<b', file.read(1))[0]  # <i = little-endian LSB to MSB

def read_string(file):
    s_len = max(ord(file.read(1)), 0)
    return str(file.read(s_len), G_ENCODING), s_len

def read_block_string(file, block_size=0):
    if block_size == 0:
        block_size = read_int(file) - 1
        logging.debug("read int (block_size) '{}', new pos: {}".format(block_size, file.tell()))
    str, s_len = read_string(file)
    assert s_len <= block_size, "ERROR: s_len ({}) cannot be bigger than block_size ({})".format(s_len, block_size)
    file.seek(block_size - s_len,1)
    logging.debug("read str '{}', block_size:{}, s_len:{}, new pos: {}".format(str, block_size, s_len, file.tell()))
    return str

def read_lyric(file):
    bar = read_int(file)
    logging.debug("read int (bar) '{}', new pos: {}".format(bar, file.tell()))
    block_size = read_int(file)
    logging.debug("read int (block_size) '{}', new pos: {}".format(block_size, file.tell()))
    s = str(file.read(block_size), G_ENCODING)
    logging.debug("read lyric '{}', block_size:{}, new pos: {}".format(s, block_size, file.tell()))
    return (bar,s)

def read_color(file):
    return (ord(file.read(1)), ord(file.read(1)), ord(file.read(1)), ord(file.read(1)))

filename = "test.gp5"
versions = ['FICHIER GUITAR PRO v5.00', 'FICHIER GUITAR PRO v5.10']
file = open(filename, 'rb')

file.seek(0, 2)
logging.debug("file size: {}".format(file.tell()))
file.seek(0)

# 1. Version
version = read_block_string(file, 30)
assert version in versions, 'ERROR: invalid version "{0}"'.format(version)
vMinor = 0 if version == versions[0] else 10

# 2. Song Info
title = read_block_string(file)
subtitle = read_block_string(file)
interpret = read_block_string(file)
album = read_block_string(file)
author_words = read_block_string(file)
author_music = read_block_string(file) # new in .gp5
copyright = read_block_string(file)
tab_author = read_block_string(file)
instructions = read_block_string(file)

notes = ""
notice_lines = read_int(file)
logging.debug("read int (notice_lines) '{}', new pos: {}".format(notice_lines, file.tell()))
for i in range(notice_lines):
    notes = notes + "\n" + read_block_string(file)

# 3. Lyrics
track = read_int(file)
logging.debug("read int (track) '{}', new pos: {}".format(track, file.tell()))
lyrics = []
for i in range(5): # for each of the 5 possible lines
    lyrics.append(read_lyric(file))
    #lyrics.append(read_block_string(file))


# 4. other info
# 4.1 print setup
if vMinor > 0:
    print_setup = file.read(49)
else:
    print_setup = file.read(30)

print_setup_lines = []
for i in range(11):
    print_setup_lines.append((file.read(4), read_string(file)))

# 4.2 tempo, key, octave
tempo = read_int(file)
logging.debug("read int (tempo) '{}', new pos: {}".format(tempo, file.tell()))

if vMinor > 0:
    dont_know_what_is_skipped_here = file.read(1)

key = read_byte(file) #ord(file.read(1))
octave = read_int(file) # alternative: skip 3, then read byte

# 4.3 Midi Channels
channels = []
for i in range(64):
    channels.append([
        read_int(file),    #instrument
        ord(file.read(1)), #volume
        ord(file.read(1)), #balance
        ord(file.read(1)), #chorus
        ord(file.read(1)), #reverb
        ord(file.read(1)), #phaser
        ord(file.read(1)), #tremolo
        ord(file.read(1)), #blank1
        ord(file.read(1))  #blank2
    ])

some_weird_padding_FF = file.read(42) # Some weird padding, filled with FFs

# 4.4 measures, tracks
measures = read_int(file)
nTracks = read_int(file)

# 5. measures    -    Measure = namedtuple("Measure", ['flags','numerator','...'])
for i in range(measures):
    #init
    markerName = markerColor = majKey = minorKey = unknown = 0
    markerColor = beam8notes = (0,0,0,0)

    flags = ord(file.read(1))

    numerator = ord(file.read(1)) if flags & 0x01 else 0 #Numerator of the (key) signature
    denominator = ord(file.read(1)) if flags & 0x02 else 0 #Denominator of the (key) signature
    repeatOpen = ((flags & 0x04) != 0) # is this a beginning of a repeat?
    repeatClose = ord(file.read(1)) if flags & 0x08 else 0 # num of repeats
    #repeatAlternative = ord(file.read(1)) if mheader & 0x10 else 0 # 16: num of alternate ending
    if flags & 0x20: # 32 = bit5: Marker
        markerName = read_block_string(file)
        markerColor = read_color(file)
    repeatAlternative = ord(file.read(1)) if flags & 0x10 else 0 # 16: num of alternate ending (seems like this comes AFTER marker?
    if flags & 0x40:  # 64 = bit6: Key change
        majKey = read_byte(file) #ord(file.read(1))
        minorKey = read_byte(file) #ord(file.read(1))
    doubleBar = ((flags & 0x80) != 0)  # 128 = bit7: double bar?

    if flags & 0x03: # if 1 or 2 is set (change in measure)
        #beam8notes = file.read(4) # "beam eight notes by" array, usually 2 - 2 - 2 - 2
        beam8notes = read_color(file)  # "beam eight notes by" array, usually 2 - 2 - 2 - 2
    if not flags & 0x10: # 16 (was) NOT set..?
        unknown = ord(file.read(1)) # unknown byte?

    tripletFeel = ord(file.read(1)) # 0: none, 1: eights, 2: 16th
    unknown2 = ord(file.read(1)) #unknown 2

    print(numerator, denominator, repeatOpen, repeatClose, repeatAlternative, markerName,
          markerColor, majKey, minorKey, doubleBar, beam8notes, unknown, tripletFeel, unknown2)

print()

# 6. Tracks
Track = namedtuple("Track", ['name','nStrings','frets'])
tracklist = []
for i in range(nTracks):
    flags = ord(file.read(1))

    if i > 0 and vMinor == 0:
        skip1 = ord(file.read(1))
        print("skip1:",skip1)

    tname = read_block_string(file, 40)
    numStrings = read_int(file)
    tuning = (read_int(file),read_int(file),read_int(file),read_int(file),read_int(file),read_int(file),read_int(file))
    midiPort = read_int(file)
    channel = read_int(file)
    channelE = read_int(file)
    frets = read_int(file)
    capo = read_int(file)
    tcolor = read_color(file)

    #file.seek(49 if vMinor > 0 else 44, 1) # skip (vMinor ? 49 : 44)


    unknown = file.read(49 if vMinor > 0 else 44)
    if vMinor > 0:
        str1 = read_block_string(file)
        str2 = read_block_string(file)

    tracklist.append(Track(tname, numStrings, frets))
    print('"' + tname + '":', numStrings, tuning, midiPort, channel, channelE, frets, capo, tcolor)


file.seek(1 if vMinor > 0 else 2, 1) # skip (vMinor ? 1 : 2)

Chord = namedtuple("Chord", ['base_fret','fretArr'])
def empty_chord():
    return Chord(0,[-1,-1,-1,-1,-1,-1,-1])

def read_chord(file):
    file.seek(17, 1) # skip(17)
    name = read_block_string(file, 21)
    file.seek(4, 1)  # skip(4)
    basefret = read_int(file)
    frets = []
    for f in range(7):
        frets[f] = read_int(file) #-1 = unplayed, 0 = no fret
    file.seek(32, 1)  # skip(32)
    return Chord(basefret, frets)

def read_chord_gp4style(file):
    cheader = ord(file.read(1)) #0x01 = gp4 chord...
    sharp = ord(file.read(1))
    blanks = file.read(3) # gp3-compatibility
    root = read_byte(file) # -1: custom, 0:C, 1:C#, ...
    type = ord(file.read(1)) #0:M,1:7,2:7M,3:6,4:m,5:m7,6:m7M,7:m6,8:sus2,9:sus4,10:7sus2,11:7sus4,12:dim,13:aug,14:5
    nine_eleven_thirteen = ord(file.read(1))
    bass = read_int(file)
    dim_aug = read_int(file) #0:perfect, 1:augmented, 2:diminished
    add = read_byte(file) # 'add'-note present in chord?
    name = read_block_string(file, 21)
    blanks = file.read(2) # gp3-compatibility
    fifth = read_byte(file)
    ninth = read_byte(file) #This tonality is valid only if the value "Nine, Eleven or Thirteen" is 11 or 13.
    eleventh = read_byte(file) #This tonality is valid only if the value "Nine, Eleven or Thirteen" is 13.
    basefret = read_int(file)
    for f in range(7):
        fret = read_int(file) #-1 = unplayed, 0 = no fret
    nbarres = ord(file.read(1))
    for b in range(5):
        bar = ord(file.read(1)) # fret number of bar
    for b in range(5):
        bar_start = ord(file.read(1)) # fret number of bar
    for b in range(5):
        bar_end = ord(file.read(1)) # fret number of bar
    omission1 = ord(file.read(1))
    omission3 = ord(file.read(1))
    omission5 = ord(file.read(1))
    omission7 = ord(file.read(1)) # 0 = not in the chord
    omission9 = ord(file.read(1)) # 1 = in the chord
    omission11 = ord(file.read(1))
    omission13 = ord(file.read(1))
    blanks = file.read(1)  # gp3-compatibility
    for f in range(7):
        fingering = read_byte(file) #-2 unknown, -1 X, 0: thumb, 1: index,... 4:little
    showFingering = ord(file.read(1)) # 1 = do display, 0 = mask
    return Chord(basefret,[-1,-1,-1,-1,-1,-1,-1])

Effect = namedtuple("Effect", ['fadein','vibrato','tap_slap_pop','bend'])
def empty_effect():
    return Effect(False,False,0,empty_bend())

def read_beat_effect(file):
    flags1 = ord(file.read(1))
    flags2 = ord(file.read(1))
    fadein = flags1 & 0x10 != 0;
    vibrato = flags1 & 0x02 != 0;
    tap_slap_pop = ord(file.read(1)) if flags1 & 0x20 else 0 # 1:tap,2:slap,3:pop
    bend = read_bend(file) if flags2 & 0x04 else empty_bend()
    upstroke = ord(file.read(1)) if flags1 & 0x40 else 0 # fastness 1 (128th) - 6 (quarters)
    downstroke = ord(file.read(1)) if flags1 & 0x40 else 0  # fastness 1 (128th) - 6 (quarters)
    pickstroke = ord(file.read(1)) if flags2 & 0x02 else 0  # (probably also not used) 1 = up, 2 = down
    return Effect(fadein, vibrato, tap_slap_pop, bend)

def read_note_effect(file):
    flags1 = ord(file.read(1))
    flags2 = ord(file.read(1))
    bend = read_bend(file) if flags1 & 0x01 else empty_bend()
    grace = read_grace(file) if flags1 & 0x10 else ()
    tremolopicking = ord(file.read(1)) if flags2 & 0x04 else 0 # 1=8th, 2=16th, 3=32th
    slide = ord(file.read(1)) if flags2 & 0x08 else 0 # tuxguitar knows only true/false and ignores the byte
    harmonic = ord(file.read(1)) if flags2 & 0x10 else 0  # 1=natural, 2=artificial, 3=tapped, 4=pinch, 5=semi
    trill = (ord(file.read(1)), ord(file.read(1))) if flags2 & 0x20 else (0,0) # (fret, period) period = 1=16th, 2=32th, 3=64th
    isHammer = flags1 & 0x02 != 0
    isLetRing = flags1 & 0x08 != 0
    isVibrato = flags2 & 0x40 != 0
    isPalmMute = flags2 & 0x02 != 0
    isStaccato = flags2 & 0x01 != 0
    return Effect(isHammer, isVibrato, harmonic, bend)

Bend = namedtuple("Bend", ['points'])
def empty_bend():
    return Bend([])

def read_bend(file):
    type = ord(file.read(1)) #http://dguitar.sourceforge.net/GP4format.html#Bends
    val = read_int(file) # 100 per note (e.g. 25 = quarter note)
    nPoints = read_int(file)
    points = []
    for p in range(nPoints):
        time_pos = read_int(file) #pos from prev point. 0-60 and is sixties of the note duration
        v = read_int(file) # value of this ponint
        vibrato = ord(file.read(1)) #0:none, 1:fast, 2:avg, 3:slow
        points.append((time_pos, v, vibrato))
    return Bend(points)

def get_last_played_note_on_this_string():
    return 24 # TODO get last played note on this string

def read_grace(file):
    fret = ord(file.read(1))
    dynamic = ord(file.read(1))
    transition = read_byte(file) #0:none, 1:slide,2:bend,3:hammer
    duration = ord(file.read(1))
    flags = ord(file.read(1))
    isDead = flags & 0x01 != 0
    isOnBeat = flags & 0x02 != 0
    return (fret, dynamic, transition, duration, isDead, isOnBeat)

Note = namedtuple("Note", ['fret','tied','dead','ghost','dynamic'])
def read_note(file, track):
    flags = ord(file.read(1))

    accentuated = (flags & 0x40) != 0 # bit6
    type = ord(file.read(1)) if flags & 0x20 else 0 # normal, ghost(dead), tied
    tied = (type == 0x02)
    dead = (type == 0x03)
    dynamic = ord(file.read(1)) if flags & 0x10 else 6  # 1:ppp, 2:pp,3:p,4:mp,5:mf,6:?f?,7:f,8:ff,9:fff
    fret = ord(file.read(1)) if flags & 0x20 else 0
    fretVal = get_last_played_note_on_this_string() if tied else fret #ToDo: get last played note on this string
    fretVal = fretVal if fretVal >= 0 and fretVal < 100 else 0 # set to zero if out of bounds
    fingeringLeft = read_byte(file) if flags & 0x80 else -1 # -1=nothing, 0:thumb, 1:index,...
    fingeringRight = read_byte(file) if flags & 0x80 else -1 # -1=nothing, 0:thumb, 1:index,...

    if (flags & 0x01): # bit1
        file.seek(8, 1) # skip(8)
    file.seek(1, 1)  # skip(1)

    effect = read_note_effect(file) if flags & 0x08 else empty_effect()  # effect
    ghost = (flags & 0x04) != 0  # bit2
    heavyAccentuated = (flags & 0x02) != 0 # bit1

    print("\t\t", fretVal, tied, dead, ghost, dynamic)
    return Note(fretVal, tied, dead, ghost, dynamic)

print()
for m in range(measures):
    for t in range(nTracks):
        for v in range(2):  # every track has two voices
            nBeats = read_int(file)
            print("Measure:{}, Track:{}, voice:{}, nBeats:{}".format(m,t,v,nBeats))
            for b in range(nBeats):
                flags = ord(file.read(1))
                status = ord(file.read(1)) if flags & 0x40 else 0x01 #64 = bit6: 0x00 = empty, 0x02 = rest
                pause = (status == 0x02)
                empty = (status == 0x00)
                duration = read_byte(file) # ... -1=half, 0=quarter, 1=eigth, ...
                dotted = (flags & 0x01) != 0
                enters = read_int(file) if flags & 0x20 else 0 # n-tuple
                times = 8 if enters > 8 else (4 if enters > 4 else (2 if enters > 0 else 0)) # tripplet feel
                chord = read_chord(file) if flags & 0x02 else empty_chord() # chord diagram
                text = read_block_string(file) if flags & 0x04 else "" # text
                effect = read_beat_effect(file) if flags & 0x08 else 0  # effect

                if flags & 0x10: # mix change!
                    instrument = read_byte(file)  # number of new instrument. -1 = no change

                    file.seek(16, 1)  # skip(16) (gp4 compatibility???)

                    volume = read_byte(file)  # new volume. -1 = no change
                    pan = read_byte(file)  # new pan. -1 = no change
                    chorus = read_byte(file)  # new chorus. -1 = no change
                    reverb = read_byte(file)  # new reverb. -1 = no change
                    phaser = read_byte(file)  # new phaser. -1 = no change
                    tremolo = read_byte(file)  # new tremolo. -1 = no change

                    temponame = read_block_string(file)
                    tempo = read_int(file)  # new tempo. -1 = no change

                    volume_duration = read_byte(file) if volume > -1 else 0  # volume change duratin in beats.
                    pan_duration = read_byte(file) if pan > -1 else 0  # pan change duratin in beats.
                    chorus_duration = read_byte(file) if chorus > -1 else 0  # chorus change duratin in beats.
                    reverb_duration = read_byte(file) if reverb > -1 else 0  # reverb change duratin in beats.
                    phaser_duration = read_byte(file) if phaser > -1 else 0  # phaser change duratin in beats.
                    tremolo_duration = read_byte(file) if tremolo > -1 else 0  # tremolo change duratin in beats.
                    tempo_duration = read_byte(file) if tempo > -1 else 0  # tempo change duratin in beats.
                    unused_byte_v5_1 = read_byte(file) if tempo > -1 and vMinor > 0 else 0  # some byte only used in versino 5.1

                    flags = ord(file.read(1))  # changes apply only to current track?
                    file.seek(1, 1)  # skip(1)

                    str1 = read_block_string(file) if vMinor > 0 else ""  # ?
                    str2 = read_block_string(file) if vMinor > 0 else ""  # ?
                # end mix change

                stringFlags = ord(file.read(1))
                print("\t", stringFlags, tracklist[t].nStrings)
                for i in range(6,-1,-1):  # for every string
                    if stringFlags & (1 << i) and (6 - i <= tracklist[t].nStrings):
                        note = read_note(file, tracklist[t])

                file.seek(1,1) # skip(1)
                skipFlag = ord(file.read(1))
                if skipFlag & 0x08:
                    file.seek(1, 1)  # skip(1)

                #return (!voice.isEmpty() ? duration.getTime() : 0 );

        file.seek(1, 1)  # skip(1)


print_header = False

if print_header:
    print("version: ", version)
    print("title: ", title)
    print("subtitle: ", subtitle)
    print("interpret: ", interpret)
    print("album: ", album)
    print("author_words: ", author_words)
    print("author_music: ", author_music)
    print("copyright: ", copyright)
    print("tab_author: ", tab_author)
    print("instructions: ", instructions)
    print("notes: ", notes)
    print("lyrics:")
    for i in range(5): # for each of the 5 possible lines
        print(lyrics[i])

    print("tempo: ", tempo)
    print("octave: ", octave)
    print("channels: ")
    for i in range(64): # 64 channels
        print(channels[i])
    print("measures: ", measures)
    print("tracks: ", nTracks)