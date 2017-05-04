# this code was written inspired by this spec: http://dguitar.sourceforge.net/GP4format.html
# also tuxguitar-gtp org.herac.tuxguitar.io.gtp GP5OutputStream
# and kguitar-code/convert/convertgtp.cpp

import struct
from collections import defaultdict

from music_transcription.fileformat.guitar_pro.utils import *


def write_gp5(measures, tracks, beats,
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
    _write_midi_channels(file, tracks)
    for i in range(42):
        _write_unsigned_byte(file, 0xFF)  # skip some weird padding, filled with FFs
    _write_int(file, len(measures))
    _write_int(file, len(tracks))
    _write_measures(file, measures)
    _write_tracks(file, tracks)
    _write_short(file, 0)  # skip 2 bytes
    _write_beats(file, beats)

    file.close()


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


def _write_color(file, color):
    for i in range(4):
        _write_unsigned_byte(file, color[i])


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


def _write_lyrics(file, lyrics: Lyrics):
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


def _write_midi_channels(file, tracks):
    channel_instrument = defaultdict(lambda: 25)  # 25 = default instrument (acoustic guitar)
    for t in tracks:
        channel_instrument[t.channel] = t.instrument
        channel_instrument[t.channelE] = t.instrument
    for i in range(64):
        _write_int(file, channel_instrument[i+1]),  # instrument
        _write_unsigned_byte(file, 13),  # volume
        _write_unsigned_byte(file, 8),  # balance
        _write_unsigned_byte(file, 0),  # chorus
        _write_unsigned_byte(file, 0),  # reverb
        _write_unsigned_byte(file, 0),  # phaser
        _write_unsigned_byte(file, 0),  # tremolo
        _write_unsigned_byte(file, 0),  # blank1
        _write_unsigned_byte(file, 0)  # blank2


def _write_measures(file, measures):
    for m in measures:
        flags = 0x01 if m.numerator > 0 else 0x00
        flags |= 0x02 if m.denominator > 0 else 0x00
        flags |= 0x04 if m.repeat_open else 0x00
        flags |= 0x08 if m.repeat_close > 0 else 0x00
        flags |= 0x10 if m.repeat_alternative > 0 else 0x00
        flags |= 0x20 if len(m.marker_name) > 0 else 0x00
        # flags |= 0x40 if [minKey changed or majKey changed] else 0x00   -> use numerator for indication?
        flags |= 0x80 if m.double_bar else 0x00

        _write_unsigned_byte(file, flags)  # flags

        _write_unsigned_byte(file, m.numerator) if flags & 0x01 else None
        _write_unsigned_byte(file, m.denominator) if flags & 0x02 else None
        _write_unsigned_byte(file, m.repeat_close) if flags & 0x08 else None
        if flags & 0x20:
            _write_block_string(file, m.marker_name)
            _write_color(file, m.marker_color)
        _write_unsigned_byte(file, m.repeat_alternative) if flags & 0x10 else None
        if flags & 0x40:
            _write_byte(file, m.majKey)
            _write_byte(file, m.minorKey)
        if flags & 0x03:  # if 1 or 2 is set (change in measure)
            b8n = calc_beam8notes(m.numerator, m.denominator) if m.beam8notes is None else m.beam8notes
            _write_color(file, b8n)
        if not flags & 0x10:  # 16 (was) NOT set..?
            _write_unsigned_byte(file, 0x00)  # unknown1
        _write_unsigned_byte(file, m.triplet_feel)  # 0: none, 1: eights, 2: 16th
        _write_unsigned_byte(file, 0x00)  # unknown2


def _write_tracks(file, tracks):
    file.seek(-1, 1)  # go 1 byte back as I am not sure where the header really needs to be
    for t in tracks:
        # more than 30 frets (usually 87) and/or using channel 10 (midi standard) indicate a drum track
        flags = 0x01 if t.frets > 30 or t.channel == 10 else 0x00
        _write_unsigned_byte(file, flags)  # flags
        _write_unsigned_byte(file, 8 | flags)  # flags again (?) [ not sure what happens here ]

        _write_block_string(file, t.name, 40)
        _write_int(file, t.nStrings)
        for i in range(7):
            _write_int(file, t.tuning[i])
        _write_int(file, t.midiPort)  # tux always writes 1
        _write_int(file, t.channel)
        _write_int(file, t.channelE)
        _write_int(file, t.frets)  # tux always writes 24
        _write_int(file, t.capo)
        _write_color(file, t.color)

        for b in [67, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, -1, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]:
            _write_byte(file, b)  # skipped values for version 5.00


def _write_beats(file, beats):
    for notes_m in beats:
        for motes_m_t in notes_m:
            for v in range(2):
                _write_int(file, len(motes_m_t[v]))  # nBeats
                for b in motes_m_t[v]:
                    flags = 0x01 if b.dotted else 0x00
                    flags |= 0x02 if b.chord is not None else 0x00
                    flags |= 0x04 if len(b.text) > 0 else 0x00
                    flags |= 0x08 if b.effect is not None else 0x00
                    flags |= 0x10 if b.mix_change is not None else 0x00
                    flags |= 0x20 if b.ntuple_feel[0] > 0 else 0x00
                    flags |= 0x40 if b.empty or b.pause else 0x00

                    _write_unsigned_byte(file, flags)  # flags

                    _write_unsigned_byte(file, 0x02 if b.pause else 0x00) if flags & 0x40 else None
                    _write_byte(file, b.duration)  # -2=whole, -1=half, 0=quarter, 1=eighth, ...
                    _write_int(file, b.ntuple_feel[0]) if flags & 0x20 else None
                    _write_chord(file, b.chord) if flags & 0x02 else None
                    _write_block_string(file, b.text) if flags & 0x04 else None
                    _write_beat_effect(file, b.effect) if flags & 0x08 else None
                    _write_mix_change(file, b.mix_change) if flags & 0x10 else None

                    string_flags = 0x00
                    for i in range(6, -1, -1):
                        if b.notes[6-i] is not None:
                            string_flags |= (1 << i)
                    _write_unsigned_byte(file, string_flags)
                    for i in range(6, -1, -1):
                        if b.notes[6-i] is not None:
                            _write_note(file, b.notes[6-i])

                    _write_short(file, 0)  # skip 2

            _write_unsigned_byte(file, 0x00)  # skip 1


def _write_chord(file, chord):
    for b in [1,1,0,0,0,12,0,0,-1,-1,-1,-1,0,0,0,0,0]:
        _write_byte(file, b)  # skipped values
    _write_block_string(file, chord.name, 21)
    _write_int(file, 0)  # skip 4
    _write_int(file, chord.base_fret)
    for f in range(7):
        _write_int(file, chord.fretArr[f])  # -1 = unplayed, 0 = no fret
    for i in range(32):
        _write_unsigned_byte(file, 0x00)  # skip 32


def _write_beat_effect(file, be):
    flags1 = 0x02 if be.vibrato else 0x00
    flags1 |= 0x10 if be.fadein else 0x00
    flags1 |= 0x20 if be.tap_slap_pop > 0 else 0x00
    # flags1 |= 0x40 if be.upstroke > 0 or be.downstroke > 0 else 0x00
    flags2 = 0x04 if be.bend is not None else 0x00
    # flags2 |= 0x02 if be.pickstroke > 0 else 0x00

    _write_unsigned_byte(file, flags1)
    _write_unsigned_byte(file, flags2)

    _write_unsigned_byte(file, be.tap_slap_pop) if flags1 & 0x20 else None  # 1:tap,2:slap,3:pop
    _write_bend(file, be.bend) if flags2 & 0x04 else None
    # _write_unsigned_byte(file, be.upstroke) if flags1 & 0x40 else None
    # _write_unsigned_byte(file, be.downstroke) if flags1 & 0x40 else None
    # _write_unsigned_byte(file, be.pickstroke) if flags2 & 0x02 else None


def _write_bend(file, bend):
    _write_unsigned_byte(file, 1)  # normal bend, see http://dguitar.sourceforge.net/GP4format.html#Bends
    _write_int(file, 0)  # value of general bend (ignored)
    _write_int(file, len(bend.points))
    for p in bend.points:
        _write_int(file, p[0])  # pos from prev point. 0-60 and is sixties of the note duration
        _write_int(file, p[1])  # value: 100 per note (e.g. 25 = quarter)
        _write_unsigned_byte(file, p[2])  # vibrato


def _write_mix_change(file, mc):
    _write_byte(file, mc.instrument)
    for i in range(16):
        _write_byte(file, -1)
    _write_byte(file, mc.volume)
    _write_byte(file, mc.pan)
    _write_byte(file, mc.chorus)
    _write_byte(file, mc.reverb)
    _write_byte(file, mc.phaser)
    _write_byte(file, mc.tremolo)

    _write_block_string(file, "")  # tempo name
    _write_int(file, mc.tempo)

    _write_byte(file, mc.volume_duration) if mc.volume > -1 else None
    _write_byte(file, mc.pan_duration) if mc.pan > -1 else None
    _write_byte(file, mc.chorus_duration) if mc.chorus > -1 else None
    _write_byte(file, mc.reverb_duration) if mc.reverb > -1 else None
    _write_byte(file, mc.phaser_duration) if mc.phaser > -1 else None
    _write_byte(file, mc.tremolo_duration) if mc.tremolo > -1 else None
    _write_byte(file, mc.tempo_duration) if mc.tempo > -1 else None

    _write_unsigned_byte(file, 0x01)
    _write_unsigned_byte(file, 0xff)


def _write_note(file, note):
    flags = 0x02 if note.heavy_accentuated else 0x00
    flags |= 0x04 if note.ghost else 0x00
    flags |= 0x08 if note.effect is not None else 0x00
    flags |= 0x10 if 0 < note.dynamic < 10 else 0x00  # 1:ppp, 9:fff
    flags |= 0x20 if note.tied or note.dead or 0 <= note.fret <= 100 else 0x00
    flags |= 0x40 if note.accentuated else 0x00

    _write_unsigned_byte(file, flags)
    _write_unsigned_byte(file, 0x02 if note.tied else (0x03 if note.dead else 0x01)) if flags & 0x20 else None
    _write_unsigned_byte(file, note.dynamic) if flags & 0x10 else None
    _write_unsigned_byte(file, note.fret) if flags & 0x20 else None

    _write_unsigned_byte(file, 0)  # skip 1
    _write_note_effect(file, note.effect) if flags & 0x08 else None


def _write_note_effect(file, ne):
    flags1 = 0x01 if ne.bend is not None else 0x00
    flags1 |= 0x02 if ne.is_hammer else 0x00
    flags1 |= 0x08 if ne.is_let_ring else 0x00
    flags1 |= 0x10 if ne.grace is not None else 0x00
    flags2 = 0x01 if ne.is_staccato else 0x00
    flags2 |= 0x02 if ne.is_palm_mute else 0x00
    flags2 |= 0x04 if ne.tremolo_picking > 0 else 0x00
    flags2 |= 0x08 if ne.slide > 0 else 0x00
    flags2 |= 0x10 if ne.harmonic > 0 else 0x00
    flags2 |= 0x20 if ne.trill is not None else 0x00
    flags2 |= 0x40 if ne.is_vibrato else 0x00

    _write_unsigned_byte(file, flags1)
    _write_unsigned_byte(file, flags2)
    _write_bend(file, ne.bend) if flags1 & 0x01 else None
    _write_grace(file, ne.grace) if flags1 & 0x10 else None
    _write_unsigned_byte(file, ne.tremolo_picking) if flags2 & 0x04 else None  # 1=8th, 2=16th, 3=32th
    _write_unsigned_byte(file, ne.slide) if flags2 & 0x08 else None
    _write_unsigned_byte(file, ne.harmonic) if flags2 & 0x10 else None  # 1=natural, 2=artificial, 3=tapped, 4=pinch, 5=semi
    if flags2 & 0x20:
        _write_unsigned_byte(file, ne.trill[0])
        _write_unsigned_byte(file, ne.trill[1])


def _write_grace(file, grace):
    _write_unsigned_byte(file, grace.fret)
    _write_unsigned_byte(file, grace.dynamic)
    _write_unsigned_byte(file, grace.transition)
    _write_unsigned_byte(file, grace.duration)
    flags = 0x01 if grace.is_dead else 0x00
    flags |= 0x02 if grace.is_on_beat else 0x00
    _write_unsigned_byte(file, flags)
