# this code was written inspired by this spec: http://dguitar.sourceforge.net/GP4format.html
# also tuxguitar-gtp org.herac.tuxguitar.io.gtp GP5OutputStream
# and kguitar-code/convert/convertgtp.cpp

import struct
from gp_utils import *

_v_minor = 0


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
            _write_byte(file, 0)


def write_gp5(measures=0, tracks=0, notes=0,
              header=None,
              lyrics=None,
              version='FICHIER GUITAR PRO v5.10',
              outfile="out.gp5"):
    global _v_minor
    _v_minor = 0 if version == GP5_VERSIONS[0] else 10
    file = open(outfile, 'wb')

    _write_block_string(file, version, 30)  # write version
    _write_header(file, header)
    _write_lyrics(file, lyrics)

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



head = Header('a','b','c','d','e','f','g','h','i','j')
lyric = Lyrics(1, [(1, "I have no"), (4, "Idea what you talk"), (1, "about, wtf")])
write_gp5(header=head, lyrics=lyric)