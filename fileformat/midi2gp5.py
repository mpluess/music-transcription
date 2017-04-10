from heapq import heappush, heappop, heapify
from collections import namedtuple
from MIDI import midi2score

with open('test.mid','rb') as f:
    midi = f.read()

score = midi2score(midi)
ticks_per_quarter = score[0]

eventqueue = []
Event = namedtuple("Event", ['time','track','event'])

track_mapping = {}

meta_events = ['text_event', 'copyright_text_event', 'track_name', 'instrument_name', 'lyric', 'marker', 'cue_point', 'text_event_08', 'text_event_09', 'text_event_0a', 'text_event_0b', 'text_event_0c', 'text_event_0d', 'text_event_0e', 'text_event_0f', 'end_track', 'set_tempo', 'smpte_offset', 'time_signature', 'key_signature','sequencer_specific', 'raw_meta_event', 'sysex_f0', 'sysex_f7', 'song_position', 'song_select', 'tune_request']
midi_events = ['note', 'key_after_touch', 'control_change', 'patch_change', 'channel_after_touch', 'pitch_wheel_change'] #'note_off', 'note_on',

for (i, track) in enumerate(score[1:]):
    track_mapping[i] = -1
    for event in track:
        heappush(eventqueue, Event(event[1], i, event))
        if(event[0] in midi_events):
            track_mapping[i] = i

# rearrange track indexs to fill gaps of tracks that only contain meta-events
print(track_mapping)
idx = 0
for key, value in sorted(track_mapping.items()):
    if value > -1:
        track_mapping[key] = idx
        idx += 1
print(track_mapping)


measure_length = 4 # default: assume 4/4

for i in range(len(eventqueue)):
    (time, track, event) = heappop(eventqueue)
    ticks = event[1] / ticks_per_quarter

    if event[0] == 'note':  # start_time, duration, channel, note, velocity
        dur = event[2] / ticks_per_quarter
        print('{}: note:{}, dur:{}, chan:{}, velocity:{}'.format(ticks, event[4], dur, event[3], event[5], event[4]))
    elif event[0] == 'set_tempo':
        tempo = round(1 / (event[2] / 60000000))
        print('{}: set tempo: {}'.format(ticks, tempo))
    elif event[0] == 'track_name':
        print('{}: set track name: {}'.format(ticks, event[2]))
    elif event[
        0] == 'time_signature':  # dtime, numerator, denominator (2=quarter, 3=eights), metronome clicks, number of 32ths to the quarter
        print('{}: time signature: {} {} {} {}'.format(ticks, event[2], event[3], event[4], event[5]))
    elif event[0] == 'marker':
        print('{}: set marker: {}'.format(ticks, event[2]))
    elif event[0] == 'patch_change':  # instrument change
        print('{}: patch change: chan:{}, patch:{}'.format(ticks, event[2], event[3]))
    elif event[0] == 'control_change':  # track lautstärke, pan, usw. - brauchen wir nicht
        dummy = 1  # print('{}: control change: chan:{}, control:{}, val:{}'.format(ticks, event[2],event[3],event[4]))
    elif event[0] == 'pitch_wheel_change':  # bend-release - brauchen wir nicht
        dummy = 2  # print('{}: pitch wheel change: chan:{}, pitch wheel:{}'.format(ticks, event[2], event[3]))
    else:
        print('{}: -- unknown event: {}'.format(ticks, event[0]))