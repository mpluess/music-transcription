from MIDI import midi2score

with open('test.mid','rb') as f:
    midi = f.read()

score = midi2score(midi)

ticks_per_quarter = score[0]

for tracknr in range(1, len(score)):
    print('Track', tracknr)
    for event in score[tracknr]:
        ticks = event[1]/ticks_per_quarter
        if event[0] == 'note': # start_time, duration, channel, note, velocity
            dur = event[2]/ticks_per_quarter
            print('{}: note:{}, dur:{}, chan:{}, velocity:{}'.format(ticks, event[4],dur,event[3],event[5],event[4]))
        elif event[0] == 'set_tempo':
            tempo = round(1/(event[2] / 60000000))
            print('{}: set tempo: {}'.format(ticks, tempo))
        elif event[0] == 'track_name':
            print('{}: set track name: {}'.format(ticks, event[2]))
        elif event[0] == 'time_signature': # dtime, numerator, denominator (2=quarter, 3=eights), metronome clicks, number of 32ths to the quarter
            print('{}: time signature: {} {} {} {}'.format(ticks, event[2], event[3], event[4], event[5]))
        elif event[0] == 'marker':
            print('{}: set marker: {}'.format(ticks, event[2]))
        elif event[0] == 'patch_change': # instrument change
            print('{}: patch change: chan:{}, patch:{}'.format(ticks, event[2], event[3]))
        elif event[0] == 'control_change': # track lautstärke, pan, usw. - brauchen wir nicht
            dummy = 1 #print('{}: control change: chan:{}, control:{}, val:{}'.format(ticks, event[2],event[3],event[4]))
        elif event[0] == 'pitch_wheel_change': # bend-release - brauchen wir nicht
            dummy = 2 #print('{}: pitch wheel change: chan:{}, pitch wheel:{}'.format(ticks, event[2], event[3]))
        else:
            print('{}: -- unknown event: {}'.format(ticks, event[0]))

#print(score[0])

