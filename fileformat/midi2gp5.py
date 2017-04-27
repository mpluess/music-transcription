from heapq import heappush, heappop, heapify
from collections import namedtuple
from MIDI import midi2score
from gp_utils import *
from gp5_writer import write_gp5


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


def calc_closest_duration(duration):  # return gp5-duration (-2 = whole, 0 = quarter, 2 = 16th) and "dotted"
    if duration > 4.5:
        return 4, True
    return 1, False


Event = namedtuple("Event", ['time', 'track', 'event'])

meta_events = ['text_event', 'copyright_text_event', 'track_name', 'instrument_name', 'lyric', 'marker', 'cue_point', 'text_event_08', 'text_event_09', 'text_event_0a', 'text_event_0b', 'text_event_0c', 'text_event_0d', 'text_event_0e', 'text_event_0f', 'end_track', 'set_tempo', 'smpte_offset', 'time_signature', 'key_signature','sequencer_specific', 'raw_meta_event', 'sysex_f0', 'sysex_f7', 'song_position', 'song_select', 'tune_request']
midi_events = ['note', 'key_after_touch', 'control_change', 'patch_change', 'channel_after_touch', 'pitch_wheel_change'] #'note_off', 'note_on',

gp5_measures = []
gp5_tracks = []
gp5_beats = []

event_queue = []

track_mapping = {}
track_name = {}
track_instrument = {}
track_min_note = {}
track_max_note = {}

unused_midi_channels = list(range(1,65))
unused_midi_channels.remove(10)  # remove drum track

# read file
with open('test.mid','rb') as f:
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
                    track_instrument[i] = -1 - track_instrument[i]  # save drums as negative number (-1 since -0 results in 0)
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

heappush(event_queue, Event(float('inf'), -1, ['song_end', float('inf')])) # add end of song event

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
            tuning = (0, 0, 0, 0, 0, 0, 0)  # apparently gp5 writes this tuning for drums
            nStrings = 6  # apparently gp5 writes drum tracks have 6 strings
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
gp5_beats.append(track_struct)  # append empty measure 0

for i in range(len(event_queue)):
    (time, track, event) = heappop(event_queue)
    ticks = event[1] / ticks_per_quarter

    if ticks >= next_beat_start_ticks:
        nn = cur_numerator if time_signature_changed else 0  # numerator
        dd = cur_denominator if time_signature_changed else 0  # denominator
        time_signature_changed = False

        # repeat_open repeat_close repeat_alt m_name marker_color maj_key min_key double_bar beam8notes triplet_feel
        gp5_measures.append(Measure(nn, dd, False, 0, 0, cur_marker_name, (0, 0, 0, 0), 0, 0, False, None, 0))
        gp5_beats.append(track_struct)  # append empty measure 0
        cur_measure += 1

        cur_marker_name = ""  # reset name
        cur_beat_start_ticks = next_beat_start_ticks
        next_beat_start_ticks += (4*cur_numerator/cur_denominator)

    if event[0] == 'note':  # start_time, duration, channel, note, velocity
        cur_track = track_mapping[track]
        dur = event[2] / ticks_per_quarter  # duration in quarters - gp5 uses -2 = whole, -1 = half, 0 = quarter etc.
        dur, dotted = calc_closest_duration(dur)
        print('{}: note:{}, dur:{}, chan:{}, velocity:{}'.format(ticks, event[4], dur, event[3], event[5], event[4]))
    elif event[0] == 'set_tempo':
        tempo = round(1 / (event[2] / 60000000))
        init_tempo = tempo if ticks == 0 else init_tempo  # update init tempo if necessary
        print('{}: set tempo: {}'.format(ticks, tempo))
    elif event[0] == 'time_signature':  # dtime, numerator, log_denominator (2=quarter, 3=eights), metronome_clicks, speed (number of 32ths to the quarter)
        cur_numerator = event[2]
        cur_denominator = pow(2,event[3])
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

if False:
    write_gp5(gp5_measures, gp5_tracks, gp5_beats, init_tempo, outfile="out.gp5")
