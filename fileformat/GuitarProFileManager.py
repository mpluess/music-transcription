import struct
import logging

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
tracks = read_int(file)

# 5. measures
for i in range(measures):
    #init
    markerName = markerColor = majKey = minorKey = unknown = 0
    markerColor = beam8notes = (0,0,0,0)

    mheader = ord(file.read(1))

    numerator = ord(file.read(1)) if mheader & 0x01 else 0 #Numerator of the (key) signature
    denominator = ord(file.read(1)) if mheader & 0x02 else 0 #Denominator of the (key) signature
    repeatOpen = ((mheader & 0x04) != 0) # is this a beginning of a repeat?
    repeatClose = ord(file.read(1)) if mheader & 0x08 else 0 # num of repeats
    #repeatAlternative = ord(file.read(1)) if mheader & 0x10 else 0 # 16: num of alternate ending
    if mheader & 0x20: # 32: Marker
        markerName = read_block_string(file)
        markerColor = read_color(file)
    repeatAlternative = ord(file.read(1)) if mheader & 0x10 else 0 # 16: num of alternate ending (seems like this comes AFTER marker?
    if mheader & 0x40:  # 64: Key change
        majKey = read_byte(file) #ord(file.read(1))
        minorKey = read_byte(file) #ord(file.read(1))
    doubleBar = ((mheader & 0x80) != 0)  # 128: double bar?

    if mheader & 0x03: # if 1 or 2 is set (change in measure)
        #beam8notes = file.read(4) # "beam eight notes by" array, usually 2 - 2 - 2 - 2
        beam8notes = read_color(file)  # "beam eight notes by" array, usually 2 - 2 - 2 - 2
    if not mheader & 0x10: # 16 (was) NOT set..?
        unknown = ord(file.read(1)) # unknown byte?

    tripletFeel = ord(file.read(1)) # 0: none, 1: eights, 2: 16th
    unknown2 = ord(file.read(1)) #unknown 2

    print(numerator, denominator, repeatOpen, repeatClose, repeatAlternative, markerName,
          markerColor, majKey, minorKey, doubleBar, beam8notes, unknown, tripletFeel, unknown2)

print()

for i in range(tracks):
    theader = ord(file.read(1))

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

    print(theader, tname, numStrings, tuning, midiPort, channel, channelE, frets, capo, tcolor, str1, ", ", str2, unknown)


file.seek(1 if vMinor > 0 else 2, 1) # skip (vMinor ? 1 : 2)

#for i in range(measures):
    #for j in range(tracks):
        # Todo



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
    print("tracks: ", tracks)