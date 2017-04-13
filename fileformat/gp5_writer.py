# this code was written inspired by this spec: http://dguitar.sourceforge.net/GP4format.html
# also tuxguitar-gtp org.herac.tuxguitar.io.gtp GP5OutputStream
# and kguitar-code/convert/convertgtp.cpp

import struct
from gp_utils import *


def _write_int(file, i):
    file.write(struct.pack('<i', i))  # <i = signed int, little-endian LSB to MSB


def _write_short(file, h):
    file.write(struct.pack('<h', h))  # <h = signed short, little-endian LSB to MSB


def _write_byte(file, b):
    file.write(struct.pack('<b', b))  # <b = signed char, little-endian LSB to MSB


def _write_unsigned_byte(file, b):
    file.write(struct.pack('<B', b))  # <B = signed char, little-endian LSB to MSB


def _write_string(file, s):
    _write_unsigned_byte(file, min(len(s), 255))
    file.write(s.encode(G_ENCODING))


def _write_block_string(file, s, block_size=0):
    if block_size == 0:
        _write_int(file, len(s) + 1)
    _write_string(file, s)
    if block_size > 0:  # fill with blanks
        assert (len(s) <= block_size)
        for i in range(len(s), block_size):
            _write_unsigned_byte(file, 0)


def write_gp5(measures=[], tracks=[], notes=[],
              tempo=120,
              header=None,
              lyrics=None,
              outfile="out.gp5"):
    version = 'FICHIER GUITAR PRO v5.00'  # for now only this version is suppored
    file = open(outfile, 'wb')

    _write_block_string(file, version, 30)  # write version
    _write_header(file, header)
    _write_lyrics(file, lyrics)
    _write_print_setup(file)
    _write_meta_info(file, tempo)
    _write_midi_channels(file)
    for i in range(42):
        _write_unsigned_byte(file, 0xFF)  # some weird padding, filled with FFs
    _write_int(file, len(measures))
    _write_int(file, len(tracks))
    _write_measures(file, measures)
    _write_tracks(file, tracks)
    _write_short(file, 0)  # skip 2 bytes
    _write_notes(file, notes)

    file.close()


def _write_header(file, header):
    # if header is not None:
    #     for s in header:
    #         _write_block_string(file, s)
    # else:
    #     for i in range(len(Header._fields) - 1):
    #         _write_block_string(file, "")

    for i in range(len(Header._fields) - 1):
        _write_block_string(file, "" if header is None else header[i])

    notes = [] if header is None else header.notes.split('\n')
    _write_int(file, len(notes))
    for s in notes:
        _write_block_string(file, s)


def _write_lyrics(file, lyrics):
    _write_int(file, 0 if lyrics is None else lyrics.track)
    for i in range(5):  # for each of the 5 possible lines
        if lyrics is not None and i >= len(lyrics.lines):
            lyrics = None
        _write_int(file, 0 if lyrics is None else lyrics.lines[i][0])  # bar
        _write_int(file, 0 if lyrics is None else len(lyrics.lines[i][1]))  # string length
        file.write(("" if lyrics is None else lyrics.lines[i][1]).encode(G_ENCODING))  # string


def _write_print_setup(file):
    _write_int(file, 210)  # page width
    _write_int(file, 297)  # page height
    _write_int(file, 10)  # margin left
    _write_int(file, 10)  # margin right
    _write_int(file, 15)  # margin top
    _write_int(file, 10)  # margin bottom
    _write_int(file, 100)  # score size percent

    _write_unsigned_byte(file, 0xff)  # view flags
    _write_unsigned_byte(file, 0x01)  # view flags

    for s in GP_PAGE_SETUP_LINES:
        _write_block_string(file, s)


def _write_meta_info(file, tempo):
    _write_int(file, tempo)
    _write_byte(file, 0)  # key
    _write_int(file, 0)  # octave


def _write_midi_channels(file):
    for i in range(64):
        _write_int(file, 30),  # instrument TODO write correct instrument (30=distorted guitar)
        _write_unsigned_byte(file, 13),  # volume
        _write_unsigned_byte(file, 0),  # balance
        _write_unsigned_byte(file, 0),  # chorus
        _write_unsigned_byte(file, 0),  # reverb
        _write_unsigned_byte(file, 0),  # phaser
        _write_unsigned_byte(file, 0),  # tremolo
        _write_unsigned_byte(file, 0),  # blank1
        _write_unsigned_byte(file, 0)  # blank2


def _write_measures(file, measures):
    for m in measures:
        _write_int(file, 0x00)  # flags ToDo


def _write_tracks(file, tracks):
    for t in tracks:
        _write_int(file, 0x00)  # flags ToDo


def _write_notes(file, notes):
    for n in notes:
        # ... more loops, write beats, ... ToDo
        _write_int(file, 0x00)  # flags ToDo


head = Header('a','b','c','d','e','f','g','h','i','j')
lyric = Lyrics(1, [(1, "I have no"), (4, "Idea what you talk"), (1, "about, wtf")])
write_gp5(header=head, lyrics=lyric)