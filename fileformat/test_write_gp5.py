from gp_utils import *
from gp5_writer import write_gp5

head = Header('a','b','c','d','e','f','g','h','i','j')
lyric = Lyrics(1, [(1, "I have no"), (4, "Idea what you talk"), (1, "about, wtf")])
measures = [
    Measure(0, 0, False, 0, 0, "", (0, 0, 0, 0), 0, 0, False, (0, 0, 0, 0), 0),
    Measure(0, 0, False, 0, 0, "", (0, 0, 0, 0), 0, 0, False, (0, 0, 0, 0), 0)
]
tracks = [  # name nStrings tuning midiPort channel channelE frets capo color
    Track("test track", 6, (65, 60, 55, 50, 45, 40, -1), 1, 1, 2, 25, 0, (255, 10, 10, 0))
]
notes = [
    [  # measure 1
        [  # track 1
            [  # voice 1: notes duration pause empty dotted ntuple_feel chord text effect mix_change
                beat([None, None, None, None, note(2), note(0), None], duration=-2)
            ], [] # voice 2
        ]
    ],
    [  # measure 2
        [  # track 1
            [  # voice 1: notes duration pause empty dotted ntuple_feel chord text effect mix_change
                beat([None, None, None, None, note(3), note(0), None], duration=-2)
            ], [] # voice 2
        ]
    ]
]
write_gp5(measures, tracks, notes, header=head, lyrics=lyric)