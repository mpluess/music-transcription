# https://github.com/aubio/aubio/blob/master/python/demos/demo_pitch.py
# 16th on 200bpm would result in onsets every 75ms
# a window size of 4096 is ~93ms (~160bpm)   /  3072 = ~70ms (~210bpm)  /  2048 = ~46ms (320bpm)

import aubio

from music_transcription.pitch_detection.abstract_pitch_detector import AbstractPitchDetector


class AubioPitchDetector(AbstractPitchDetector):
    def __init__(self, tuning=(64, 59, 55, 50, 45, 40), n_frets=24, window_size=4096, hop_size=512):
        super().__init__(tuning, n_frets)

        self.window_size = window_size
        self.hop_size = hop_size

    def predict(self, path_to_wav_file, onset_times_seconds):
        return self.pitch_sets_to_multilabel_matrix(self.predict_pitches(path_to_wav_file, onset_times_seconds))

    def predict_pitches(self, path_to_wav_file, onset_times_seconds):
        src = aubio.source(path_to_wav_file, hop_size=self.hop_size)

        # methods = ['default', 'schmitt', 'fcomb', 'mcomb', 'yin', 'yinfft']  # in v 0.4.5 default = yinfft
        pitch_o = aubio.pitch("yinfft", self.window_size, self.hop_size, src.samplerate)
        pitch_o.set_unit("midi")  # output as midi-pitch (as opposed to Hz)
        # pitch_o.set_tolerance(0.5)  # default for yinfft is 0.85, but having a pitch is more important here.

        list_of_pitch_sets = []
        s_last = 0
        offset = int(self.window_size / self.hop_size)
        offset2 = int(round(offset/2))
        for onset_time in onset_times_seconds:
            s_next = int(onset_time * src.samplerate / self.hop_size) + offset
            for i in range(s_last, s_next):  # skip unused pitches
                samples, read = src()
                pitch = int(round(pitch_o(samples)[0]))
            confidence = pitch_o.get_confidence()
            # possible Extension: TODO use all methods and get "best pitch" (most confident by most methods)

            # print("pitch:{} with confidence:{} at:{}s".format(
            #     pitch,
            #     confidence,
            #     round(s_next*self.hop_size/src.samplerate, 3)
            # ))

            for i in range(offset2):
                samples, read = src()
                pitch2 = int(round(pitch_o(samples)[0]))
                conf2 = pitch_o.get_confidence()
                if pitch != pitch2 and (pitch == 0 or conf2 > confidence):
                    pitch = pitch2
                    confidence = conf2
                    # print("changed pitch to:{} with confidence:{} at:{}s".format(
                    #     pitch,
                    #     confidence,
                    #     round((s_next + i + 1) * self.hop_size / src.samplerate, 3)
                    # ))
                confidence = max(confidence, conf2)  # update confidence if it was the same note

            if pitch < self.config['min_pitch']:
                print("WARNING: estimated pitch is lower than given tuning! => set to minimum:", pitch, '=>', self.config['min_pitch'])
                pitch = self.config['min_pitch']
            elif pitch > self.config['max_pitch']:
                print("WARNING: estimated pitch is higher than possible on this guitar configuration! => set to maximum:", pitch, '=>', self.config['max_pitch'])
                pitch = self.config['max_pitch']
            list_of_pitch_sets.append({pitch})
            s_last = s_next + offset2

        return list_of_pitch_sets
