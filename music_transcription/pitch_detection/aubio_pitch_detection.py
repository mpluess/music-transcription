# https://github.com/aubio/aubio/blob/master/python/demos/demo_pitch.py

import aubio


class AubioPitchDetector:
    def __init__(self, tuning, n_frets):
        self.tuning = tuning
        self.n_frets = n_frets

    def predict_pitches_monophonic(self, path_to_wav_file, onset_times_seconds):
        window_size = 4096  # fft size
        src = aubio.source(path_to_wav_file, hop_size=512)

        # methods = ['default', 'schmitt', 'fcomb', 'mcomb', 'yin', 'yinfft']  # in v 0.4.5 default = yinfft
        pitch_o = aubio.pitch("yinfft", window_size, src.hop_size, src.samplerate)
        pitch_o.set_unit("midi")  # output as midi-pitch (as opposed to Hz)
        pitch_o.set_tolerance(0.05)  # default for yinfft is 0.85, but having a pitch is more important here.

        pitches = []
        last = 0
        offset = int(window_size / src.hop_size)
        for onset_time in onset_times_seconds:
            next = int(onset_time * src.samplerate / src.hop_size) + offset
            for i in range(last, next):
                samples, read = src()
                pitch = pitch_o(samples)[0]
            # confidence = pitch_o.get_confidence()  # if confidence < below threshold: pitch = 0
            # TODO (?) use all methods and get "best pitch"

            print("pitch:{} with confidence:{} at window:{}".format(int(round(pitch)), pitch_o.get_confidence(), next))
            pitches.append(int(round(pitch)) if pitch > 0 else 42)  # use 42 as default, as 0 not possible w gp5_writer
            last = next

        return pitches
