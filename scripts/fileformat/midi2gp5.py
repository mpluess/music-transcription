from heapq import heappush, heappop
from copy import deepcopy
from music_transcription.fileformat.MIDI import midi2score
from music_transcription.fileformat.guitar_pro.gp5_writer import write_gp5
from music_transcription.fileformat.guitar_pro.utils import *

MIDI2GP5_ACCURACY_QUARTERS = 1
MIDI2GP5_ACCURACY_EIGHTS = 2
MIDI2GP5_ACCURACY_SIXTEENTHS = 4
MIDI2GP5_ACCURACY_32 = 8
MIDI2GP5_ACCURACY_64 = 16

MIDI2GP5_ACCURACY_TO_GP5DURATION = {1: 0, 2: 1, 4: 2, 8: 3, 16: 4}

G_ACCURACY = MIDI2GP5_ACCURACY_SIXTEENTHS
G_GP5DURATION = MIDI2GP5_ACCURACY_TO_GP5DURATION[G_ACCURACY]


def determine_tuning(min_note):
    if min_note < 28:  # 28 = E2 = std bass tuning. Assume a 5-string here!
        diff = max(0, 23 - min_note)
        return 43 - diff, 38 - diff, 33 - diff, 28 - diff, 23 - diff, -1, -1  # tuning (7 strings: highest to lowest)
    if min_note < 33:  # 33 = A2, that'd be a 1-step down 7-string. assume 4-string bass here!
        diff = max(0, 28 - min_note)
        return 43 - diff, 38 - diff, 33 - diff, 28 - diff, -1, -1, -1
    if min_note < 36:  # 36 = C3 on guitar, which would be Drop C. Assume 7-String here!
        diff = max(0, 35 - min_note)
        return 64 - diff, 59 - diff, 55 - diff, 50 - diff, 45 - diff, 40 - diff, 35 - diff
    if min_note <= 38:  # 38 = D3, that'd be drop D or C... or some n step down tuning. We assume drop tuning
        diff = max(0, 38 - min_note)
        return 64 - diff, 59 - diff, 55 - diff, 50 - diff, 45 - diff, 38 - diff, -1
    else:  # assume standard guitar tuning
        diff = max(0, 40 - min_note)
        return 64 - diff, 59 - diff, 55 - diff, 50 - diff, 45 - diff, 40 - diff, -1


def tied_or_none(notes_arr):
    print(notes_arr)
    result = True
    for nt in notes_arr:
        result = result and (nt is None or nt.tied)
    return result


def get_collapsed_lengths(num_notes):
    lengths = []
    duration = G_GP5DURATION
    while num_notes > 0 and duration > -2:
        if num_notes % 2 == 1:
            lengths.append(duration)
            num_notes -= 1
        num_notes = int(num_notes/2)
        duration -= 1
    for k in range(num_notes):
        lengths.append(duration)
    lengths.reverse()
    return lengths


def collapse_beats(beats):
    for x in range(len(beats)):
        for y in range(len(beats[x])):
            new_beats = []
            cur_beat = None
            cur_beat_tied = None
            cur_count = 0
            beats[x][y][0].append(beat([note(-2)]*7))  # dummy beat that triggers a change before the last beat
            for bt in beats[x][y][0]:
                if cur_beat is None:
                    cur_beat = bt
                    cur_count = 1
                elif tied_or_none(bt.notes) and bt.pause == cur_beat.pause:
                    cur_count += 1
                    cur_beat_tied = bt
                else:
                    c_notes = cur_beat.notes
                    print(cur_count, get_collapsed_lengths(cur_count))
                    for z in get_collapsed_lengths(cur_count):
                        new_beats.append(beat(c_notes, duration=z, pause=cur_beat.pause, empty=cur_beat.empty))
                        if cur_beat_tied is not None:
                            c_notes = cur_beat_tied.notes
                    cur_beat = bt
                    cur_count = 1
                    cur_beat_tied = None
            beats[x][y] = (new_beats, [])


Event = namedtuple("Event", ['time', 'track', 'event'])

meta_events = ['text_event', 'copyright_text_event', 'track_name', 'instrument_name', 'lyric', 'marker', 'cue_point', 'text_event_08', 'text_event_09', 'text_event_0a', 'text_event_0b', 'text_event_0c', 'text_event_0d', 'text_event_0e', 'text_event_0f', 'end_track', 'set_tempo', 'smpte_offset', 'time_signature', 'key_signature','sequencer_specific', 'raw_meta_event', 'sysex_f0', 'sysex_f7', 'song_position', 'song_select', 'tune_request']
midi_events = ['note', 'key_after_touch', 'control_change', 'patch_change', 'channel_after_touch', 'pitch_wheel_change']  #'note_off', 'note_on',

gp5_measures = []
gp5_tracks = []
gp5_beats = []

event_queue = []

track_mapping = {}
track_name = {}
track_instrument = {}
track_min_note = {}
track_max_note = {}

unused_midi_channels = list(range(1, 65))
unused_midi_channels.remove(10)  # remove drum track

# read file
with open('test.mid', 'rb') as f:
    midi = f.read()
score = midi2score(midi)
ticks_per_quarter = score[0]

for (i, track) in enumerate(score[1:]):
    # determine track mapping
    track_mapping[i] = -1
    track_name[i] = "Track" + str(i)
    track_instrument[i] = 25
    track_max_note[i] = -1
    track_min_note[i] = 128
    elements_to_push = []
    for event in track:
        if event[0] in midi_events:
            if event[0] == 'patch_change' and event[1] == 0:
                track_instrument[i] = event[3]
                if event[2] == 9:  # channel 10 (0-based here) indicates a drum track! (midi standard)
                    track_instrument[i] = -1 - track_instrument[i]  # save drums as negative number (-1 bc -0 == 0)
                print('initial patch [raw track:{}]: chan:{}, patch:{}'.format(i, event[2], event[3]))
            else:
                elements_to_push.append(event)
                if event[0] == 'note':  # start_time, duration, channel, note, velocity
                    track_max_note[i] = max(track_max_note[i], event[4])
                    track_min_note[i] = min(track_min_note[i], event[4])
            track_mapping[i] = i
        else:
            if event[0] == 'track_name':
                track_name[i] = event[2].decode('ISO-8859-1')  # decode('ascii','ignore') / decode('UTF-8')
            else:
                elements_to_push.append(event)
    for event in elements_to_push:
        heappush(event_queue, Event(event[1], track_mapping[i], event))

heappush(event_queue, Event(float('inf'), -1, ['song_end', float('inf')]))  # add end of song event

# rearrange track indices to fill gaps of tracks that only contain meta-events
idx = 0
for key, value in sorted(track_mapping.items()):
    if value > -1:  # actual track, no meta-track
        track_mapping[key] = idx
        idx += 1

        tuning = determine_tuning(track_min_note[key])
        nStrings = 7 - tuning.count(-1)
        channel1 = channel2 = 10  # default assume drum track
        if track_instrument[key] >= 0:  # not a drum track
            channel1 = min(unused_midi_channels)
            unused_midi_channels.remove(channel1)
            channel2 = min(unused_midi_channels)
            unused_midi_channels.remove(channel2)
        else:
            track_instrument[key] = -1 - track_instrument[key]  # get back correct instrument number
            tuning = (0, 0, 0, 0, 0, 0, 0)  # apparently fileformat writes this tuning for drums
            nStrings = 6  # apparently fileformat writes drum tracks have 6 strings
        gp5_tracks.append(Track(
            track_name[key],  # track name
            nStrings,  # number of strings
            tuning,  # tuning (7 strings: highest to lowest)
            1,  # midi port (default always 1)
            channel1,  # channel 1
            channel2,  # channel 2 (effects channel)
            87 if channel1 == 10 else 30,  # frets - use some reserve frets in case the tuning was determined wrongly
            0,  # capo
            (0 if channel1 == 10 else 255, 0, 0, 0),  # color (we use black for drums, red for guitar)
            track_instrument[key]  # instrument
        ))

track_struct = []
for t in gp5_tracks:
    track_struct.append(([], []))  # add t tuples with two empty lists (for each voice)
    print(t)


# init default values
cur_tempo = init_tempo = 120

cur_numerator = 4  # assume 4/4
cur_denominator = 4  # assume 4/4
time_signature_changed = True  # first measure needs to contain time signature

cur_marker_name = ""

cur_beat_start_ticks = 0.0  # current beat started at tick 0.0
next_beat_start_ticks = 4.0  # next beat starts at tick 4.0
cur_measure = 0  # start with measure 0
gp5_beats.append(deepcopy(track_struct))  # append empty measure 0

gp5_note_overflows = [(0, None)]*len(gp5_tracks)  # number of notes (of G_GP5DURATION) that didn't fit into last measure

for i in range(len(event_queue)):
    (time, track, event) = heappop(event_queue)
    ticks = event[1] / ticks_per_quarter

    if ticks >= next_beat_start_ticks:
        nn = cur_numerator if time_signature_changed else 0  # numerator
        dd = cur_denominator if time_signature_changed else 0  # denominator
        time_signature_changed = False

        # repeat_open repeat_close repeat_alt m_name marker_color maj_key min_key double_bar beam8notes triplet_feel
        gp5_measures.append(Measure(nn, dd, False, 0, 0, cur_marker_name, (0, 0, 0, 0), 0, 0, False, None, 0))
        cur_measure += 1

        if event[0] != 'song_end':  # don't create new measure in the end TODO maybe needed for overflow notes?
            gp5_beats.append(deepcopy(track_struct))  # append empty measure 0
            for j in range(len(gp5_note_overflows)):  # write overflowing notes
                of_cur_notes = min(gp5_note_overflows[j][0], int(4*cur_numerator/cur_denominator*G_ACCURACY))
                for b in range(of_cur_notes):
                    gp5_beats[cur_measure][j][0].append(
                        beat(gp5_note_overflows[j][1], duration=G_GP5DURATION)
                    )
                gp5_note_overflows[j] = (gp5_note_overflows[j][0] - of_cur_notes, gp5_note_overflows[j][1])

            cur_marker_name = ""  # reset name
            cur_beat_start_ticks = next_beat_start_ticks
            next_beat_start_ticks += (4*cur_numerator/cur_denominator)

    if event[0] == 'note':  # start_time, duration, channel, note, velocity
        dur = event[2] / ticks_per_quarter  # duration of the midi note [in quarters]
        dur_remaining = next_beat_start_ticks - ticks  # remaining quarters that fit in the current measure
        dur_past = ticks - cur_beat_start_ticks  # past quarters in current measure before current note
        n_notes = round(dur * G_ACCURACY)  # total notes to be written
        remaining_notes = round(dur_remaining * G_ACCURACY)  # max notes that can be written to current measure
        past_notes = round(dur_past * G_ACCURACY)  # note offset from measure start
        cur_notes = min(n_notes, remaining_notes)  # notes to write into current measure

        # at this point every note should be exactly of length G_ACCURACY
        cur_track = track_mapping[track]  # real track index (without meta-tracks)
        cur_gp5_beats = gp5_beats[cur_measure][cur_track][0]
        for b in range(len(cur_gp5_beats), past_notes):  # insert pauses
            cur_gp5_beats.append(beat([None]*7, duration=G_GP5DURATION, pause=True))

        overflow_notes = [None]*7
        is_tied = False  # first beat untied
        for cur_beat_idx in range(past_notes, past_notes + cur_notes):
            if len(cur_gp5_beats) <= cur_beat_idx:  # insert new beat if needed
                cur_gp5_beats.append(beat([None] * 7))
            assert len(cur_gp5_beats) > cur_beat_idx, "A_ERROR: len:{}, idx:{}".format(len(cur_gp5_beats), cur_beat_idx)
            notes = cur_gp5_beats[cur_beat_idx].notes
            tuning = gp5_tracks[cur_track].tuning
            for t in range(6, -1, -1):
                if 0 <= tuning[t] <= event[4] and t < gp5_tracks[cur_track].nStrings and notes[t] is None:
                    notes[t] = note(event[4] - tuning[t], tied=is_tied)
                    overflow_notes = notes
                    break  # TODO this doesnt make sure a note is written!
                    # better: get all prev notes, list all possible positions for each note -> get most plausible
                    # right now a high E is written on lowest string, when followed by a low E -> impossible!

            cur_gp5_beats[cur_beat_idx] = beat(notes, duration=G_GP5DURATION)
            is_tied = True  # following beats tied!

        gp5_note_overflows[cur_track] = [max(0, n_notes - cur_notes), overflow_notes]
        print('{}[{}]: note:{}, dur:{}, chan:{}, veloc:{}'.format(ticks, cur_track, event[4], dur, event[3], event[5]))
    elif event[0] == 'set_tempo':
        tempo = round(1 / (event[2] / 60000000))
        init_tempo = tempo if ticks == 0 else init_tempo  # update init tempo if necessary
        print('{}: set tempo: {}'.format(ticks, tempo))
    elif event[0] == 'time_signature':  # event, time, nn, dd , metronome_clicks, speed (number of 32ths to the quarter)
        cur_numerator = event[2]  # nn / numerator
        cur_denominator = pow(2, event[3])  # dd / log_denominator -> 2=quarter, 3=eights, etc.
        time_signature_changed = True
        next_beat_start_ticks = cur_beat_start_ticks + (4 * cur_numerator / cur_denominator)
        print('{}: time signature: {} {} {} {}'.format(ticks, event[2], event[3], event[4], event[5]))
    elif event[0] == 'marker':
        cur_marker_name = event[2].decode('ISO-8859-1')  # decode('ascii','ignore') / decode('UTF-8')
        print('{}: set marker: {}'.format(ticks, cur_marker_name))
    elif event[0] == 'patch_change':  # instrument change
        print('{}: patch change: chan:{}, patch:{}'.format(ticks, event[2], event[3]))
    elif event[0] == 'control_change':  # track volume, pan, usw. - not needed here
        dummy = 1  # print('{}: control change: chan:{}, control:{}, val:{}'.format(ticks, event[2],event[3],event[4]))
    elif event[0] == 'pitch_wheel_change':  # bend-release - not needed here
        dummy = 2  # print('{}: pitch wheel change: chan:{}, pitch wheel:{}'.format(ticks, event[2], event[3]))
    else:
        print('{}: -- unknown event: {}'.format(ticks, event[0]))


for m in gp5_measures:
    print(m)

collapse_beats(gp5_beats)

assert len(gp5_measures) == len(gp5_beats), "ERR: Mlen {}!={}".format(len(gp5_measures), len(gp5_beats))
if False:
    for m in range(len(gp5_beats)):
        print('Measure {}'.format(m+1))
        assert len(gp5_tracks) == len(gp5_beats[m]), "ERR: Tlen {}!={}".format(len(gp5_tracks), len(gp5_beats[m]))
        for t in range(len(gp5_beats[m])):
            print('\tTrack {}'.format(t + 1))
            for b in gp5_beats[m][t][0]:
                print("\t\tBeat")
                if b.pause or b.empty:
                    print("\t\t\t", b.notes)
                else:
                    for n in b.notes:
                        print("\t\t\t", n)
                print("\t\t\t dur:{}, dot:{}, pause:{}, empty:{}".format(b.duration, b.dotted, b.pause, b.empty))

write_gp5(gp5_measures, gp5_tracks, gp5_beats, init_tempo, outfile="../../tmp/midi2gp5_output.gp5")
