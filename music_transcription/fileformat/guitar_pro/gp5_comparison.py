from math import pow, log2
from copy import deepcopy

from music_transcription.fileformat.guitar_pro.utils import Beat


def retrieve_markers(f):
    markers = []
    cur_marker = ''
    cur_len = 0
    for m in f.measures:
        if m.marker_name > '':
            if cur_len > 0:
                markers.append((cur_marker, cur_len))
            cur_marker = m.marker_name
            cur_len = 0
        cur_len += 1
    markers.append((cur_marker, cur_len))
    return markers


# compares meta data such as tempo, number of measures/tracks, length of marked parts
def meta_comparison(f1, f2):
    for attr in ['tempo', 'nMeasures', 'nTracks']:
        if getattr(f1, attr) != getattr(f2, attr):
            print('attribute {} differs: {} -> {}'.format(attr, getattr(f1, attr), getattr(f2, attr)))

    markers1 = retrieve_markers(f1)
    markers2 = retrieve_markers(f2)
    common_markers = []

    for m1 in markers1:
        m2_len = 0
        for m2 in markers2:
            if m1[0] == m2[0]:
                m2_len = m2[1]
                markers2.remove(m2)
                break
        if m2_len == 0:
            print('new section in file 1: {} ({})'.format(m1[0], m1[1]))
        elif m2_len != m1[1]:
            print('section {} changed: {} -> {}'.format(m1[0], m1[1], m2_len))
        else:
            common_markers.append(m1[0])  # TODO also add when they have different lengths?
    for m2 in markers2:
        print('new section in file 2: {} ({})'.format(m2[0], m2[1]))

    return common_markers


def beats_equal(b1, b2, tuning):
    n1 = []
    n2 = []
    for i in range(7):
        if b1.notes[i] is not None:
            n1.append(b1.notes[i].fret + tuning[i] + (100 if b1.notes[i].tied else 0))
        if b2.notes[i] is not None:
            n2.append(b2.notes[i].fret + tuning[i] + (100 if b2.notes[i].tied else 0))
    if sorted(n1) != sorted(n2):
        return False
    return b1.duration == b2.duration


def beats_equal_positions(b1, b2):
    for i in range(7):
        if b1.notes[i] != b2.notes[i]:
            return False
    return b1.duration == b2.duration


def compare(f1, f2, t1, t2, tuning=(64, 59, 55, 50, 45, 40), common_markers=None, compare_positions=False):
    # remap to zero based index
    t1 -= 1
    t2 -= 1
    if common_markers is None:
        common_markers = []

    empty_measure = [([], []), ([], []), ([], [])]
    measures = []
    beats = []

    # loop through common measures
    m1 = m2 = m1_signature = m2_signature = 0
    current_measure = deepcopy(empty_measure)
    while m1 < f1.nMeasures and m2 < f2.nMeasures:
        # f1 has a marker which exists in both files -> skip second file until this marker
        if f1.measures[m1].marker_name in common_markers:
            marker = f1.measures[m1].marker_name
            common_markers.remove(marker)
            while f2.measures[m2].marker_name != marker:  # skip f2 until marker, add all measures to track 3
                current_measure[0][0].append(Beat([None] * 7, empty=True))  # add empty quarter
                current_measure[1][0].append(Beat([None] * 7, empty=True))  # add empty quarter
                current_measure[2] = f2.notes[m2][t2]  # current_measure[2][0] += f2.notes[m2][t2][0]

                if f2.measures[m2].denominator > 0:
                    m2_signature = f2.measures[m2].numerator / f2.measures[m2].denominator
                measures.append(f2.measures[m2])
                beats.append(current_measure)
                current_measure = deepcopy(empty_measure)
                m2 += 1
        # f2 has a marker which exists in both files -> skip first file until this marker
        elif f2.measures[m2].marker_name in common_markers:
            marker = f2.measures[m2].marker_name
            common_markers.remove(marker)
            while f1.measures[m1].marker_name != marker:  # skip f1 until marker, add all measures to track 2
                current_measure[0][0].append(Beat([None] * 7, empty=True))  # add empty quarter
                current_measure[1] = f1.notes[m1][t1]  # current_measure[1][0] += f1.notes[m1][t1][0]
                current_measure[2][0].append(Beat([None] * 7, empty=True))  # add empty quarter

                if f1.measures[m1].denominator > 0:
                    m1_signature = f1.measures[m1].numerator / f1.measures[m1].denominator
                measures.append(f1.measures[m1])
                beats.append(current_measure)
                current_measure = deepcopy(empty_measure)
                m1 += 1

        # read measure beat by beat
        idx1 = idx2 = 0
        dur1 = dur2 = 0
        while idx1 < len(f1.notes[m1][t1][0]) and idx2 < len(f2.notes[m2][t2][0]):
            b1 = f1.notes[m1][t1][0][idx1]
            b2 = f2.notes[m2][t2][0][idx2]
            dur1 += pow(2, -b1.duration)
            dur2 += pow(2, -b2.duration)
            eq = beats_equal_positions(b1, b2) if compare_positions else beats_equal(b1, b2, tuning)
            if eq:  # append to common track, append pauses to the others
                current_measure[0][0].append(b1)
                current_measure[1][0].append(Beat([None] * 7, duration=b1.duration, pause=True))
                current_measure[2][0].append(Beat([None] * 7, duration=b1.duration, pause=True))
            else:  # append respective beats to their tracks
                dur = pow(2, -b1.duration)
                current_measure[1][0].append(b1)
                current_measure[2][0].append(b2)
                # continue appending until durations are the same
                while dur1 != dur2:
                    if dur1 < dur2 and idx1 < len(f1.notes[m1][t1][0]) - 1:
                        idx1 += 1
                        b1 = f1.notes[m1][t1][0][idx1]
                        dur1 += pow(2, -b1.duration)
                        dur += pow(2, -b1.duration)  # add only when b1 is updated!
                        current_measure[1][0].append(b1)
                    elif idx2 < len(f2.notes[m2][t2][0]) - 1:
                        idx2 += 1
                        b2 = f2.notes[m2][t2][0][idx2]
                        dur2 += pow(2, -b2.duration)
                        current_measure[2][0].append(b2)
                    else:
                        break  # abort if nothing happens anymore

                while dur > 0:  # add pauses to track1
                    x = 4
                    while x > dur:
                        x /= 2
                    dur -= x
                    current_measure[0][0].append(Beat([None] * 7, duration=int(-log2(x)), pause=True))

            idx1 += 1
            idx2 += 1
            # end loop read measure beat by beat

        # read remaining beats if one of the track measures is longer than the other
        while idx1 < len(f1.notes[m1][t1][0]):
            current_measure[0][0].append(Beat([None] * 7, empty=True))  # add empty quarter
            current_measure[1][0].append(f1.notes[m1][t1][0][idx1])
            current_measure[2][0].append(Beat([None] * 7, empty=True))  # add empty quarter
            idx1 += 1
        while idx2 < len(f2.notes[m2][t2][0]):
            current_measure[0][0].append(Beat([None] * 7, empty=True))  # add empty quarter
            current_measure[1][0].append(Beat([None] * 7, empty=True))  # add empty quarter
            current_measure[2][0].append(f2.notes[m2][t2][0][idx2])
            idx2 += 1

        # append longer measure
        if f1.measures[m1].denominator > 0:
            m1_signature = f1.measures[m1].numerator / f1.measures[m1].denominator
        if f2.measures[m2].denominator > 0:
            m2_signature = f2.measures[m2].numerator / f2.measures[m2].denominator
        if m1_signature == m2_signature:
            measures.append(f1.measures[m1])
        else:
            print('time signature differ: {}/{} -> {}/{}'.format(f1.measures[m1].numerator,
                                                                 f1.measures[m1].denominator,
                                                                 f2.measures[m2].numerator,
                                                                 f2.measures[m2].denominator))
            measures.append(f2.measures[m2] if m2_signature > m1_signature else f1.measures[m1])

        beats.append(current_measure)
        current_measure = deepcopy(empty_measure)
        m1 += 1
        m2 += 1
        # end loop read common measures

    # read remaining measures for longer track
    while m1 < f1.nMeasures:
        current_measure[1] = f1.notes[m1][t1]  # current_measure[1][0] += f1.notes[m1][t1][0]
        measures.append(f1.measures[m1])
        beats.append(current_measure)
        current_measure = deepcopy(empty_measure)
        m1 += 1
    while m2 < f2.nMeasures:
        current_measure[2] = f2.notes[m2][t2]  # current_measure[2][0] += f2.notes[m2][t2][0]
        measures.append(f2.measures[m2])
        beats.append(current_measure)
        current_measure = deepcopy(empty_measure)
        m2 += 1

    return measures, beats  # compare_gp5.py "..\tmp\quintfall.gp5" "..\tmp\midi2gp5_output.gp5"
