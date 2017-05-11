# https://github.com/aubio/aubio/blob/master/python/demos/demo_pitch.py
# 16th on 200bpm would result in onsets every 75ms
# a window size of 4096 is ~93ms (~160bpm)   /  3072 = ~70ms (~210bpm)  /  2048 = ~46ms (320bpm)

import aubio


class AubioPitchDetector:
    def __init__(self, tuning, n_frets):
        self.tuning = tuning
        self.n_frets = n_frets

    def predict_pitches_monophonic(self, path_to_wav_file, onset_times_seconds):
        window_size = 4096  # fft size = 93ms
        src = aubio.source(path_to_wav_file, hop_size=512)

        # methods = ['default', 'schmitt', 'fcomb', 'mcomb', 'yin', 'yinfft']  # in v 0.4.5 default = yinfft
        pitch_o = aubio.pitch("yinfft", window_size, src.hop_size, src.samplerate)
        pitch_o.set_unit("midi")  # output as midi-pitch (as opposed to Hz)
        # pitch_o.set_tolerance(0.5)  # default for yinfft is 0.85, but having a pitch is more important here.

        pitches = []
        s_last = 0
        offset = int(window_size / src.hop_size)
        offset2 = int(round(offset/2))
        for onset_time in onset_times_seconds:
            s_next = int(onset_time * src.samplerate / src.hop_size) + offset
            for i in range(s_last, s_next):  # skip unused pitches
                samples, read = src()
                pitch = int(round(pitch_o(samples)[0]))
            confidence = pitch_o.get_confidence()
            # possible Extension: TODO use all methods and get "best pitch" (most confident by most methods)

            print("pitch:{} with confidence:{} at:{}s".format(
                pitch,
                confidence,
                round(s_next*src.hop_size/src.samplerate, 3)
            ))

            for i in range(offset2):
                samples, read = src()
                pitch2 = int(round(pitch_o(samples)[0]))
                conf2 = pitch_o.get_confidence()
                if pitch != pitch2 and (pitch == 0 or conf2 > confidence):
                    pitch = pitch2
                    confidence = conf2
                    print("changed pitch to:{} with confidence:{} at:{}s".format(
                        pitch,
                        confidence,
                        round((s_next + i + 1) * src.hop_size / src.samplerate, 3)
                    ))

            pitches.append(pitch if pitch > 0 else 42)  # use 42 as default, as 0 not possible w gp5_writer
            s_last = s_next + offset2

        return pitches
