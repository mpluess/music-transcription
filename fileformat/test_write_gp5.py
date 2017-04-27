# sample of how to use the write_gp5 method from gp5_writer.py

from gp_utils import *
from gp5_writer import write_gp5

# numerator, denominator, repeat_open, repeat_close, repeat_alt, marker_name, marker_color,
# major_key, minor_key, double_bar, beam8notes, triplet_feel
measures = [  # attention: first measure needs to specify numerator, denominator and beam8notes
    Measure(4, 4, False, 0, 0, "", (0, 0, 0, 0), 0, 0, False, (2, 2, 2, 2), 0),
    Measure(0, 0, False, 0, 0, "", (0, 0, 0, 0), 0, 0, False, (0, 0, 0, 0), 0),
    Measure(0, 0, False, 0, 0, "", (0, 0, 0, 0), 0, 0, False, (0, 0, 0, 0), 0),
    Measure(0, 0, False, 0, 0, "", (0, 0, 0, 0), 0, 0, False, (0, 0, 0, 0), 0),
    Measure(0, 0, False, 0, 0, "", (0, 0, 0, 0), 0, 0, False, (0, 0, 0, 0), 0),
    Measure(0, 0, False, 0, 0, "", (0, 0, 0, 0), 0, 0, False, (0, 0, 0, 0), 0),
    Measure(0, 0, False, 0, 0, "", (0, 0, 0, 0), 0, 0, False, (0, 0, 0, 0), 0),
    Measure(0, 0, False, 0, 0, "", (0, 0, 0, 0), 0, 0, False, (0, 0, 0, 0), 0),
]

tracks = [  # name nStrings tuning midiPort channel channelE frets capo color
    Track("test track", 6, (65, 60, 55, 50, 45, 40, -1), 1, 1, 2, 25, 0, (200, 55, 55, 0), 30)
]

# make sure you have as many measures and tracks as defined above!
beats = [
    [  # measure 1
        (  # track 1
            [  # voice 1: notes duration pause empty dotted ntuple_feel chord text effect mix_change
                beat([None, None, None, None, note(2), note(0), None], duration=-2)
            ], []  # voice 2
        )
    ],
    [  # measure 2
        (  # track 1
            [  # voice 1: notes duration pause empty dotted ntuple_feel chord text effect mix_change
                beat([None, None, None, None, note(3), note(0), None], duration=-2)
            ], []  # voice 2
        )
    ],
    [  # measure 3
            (  # track 1
                [  # voice 1: notes duration pause empty dotted ntuple_feel chord text effect mix_change
                    beat([None, None, None, None, note(3), note(0), None], duration=-2)
                ], []  # voice 2
            )
        ],
    [  # measure 4
        (  # track 1
            [  # voice 1: notes duration pause empty dotted ntuple_feel chord text effect mix_change
                beat([None, None, None, None, note(3), note(0), None], duration=-2)
            ], []  # voice 2
        )
    ],
    [  # measure 5
        (  # track 1
            [  # voice 1: notes duration pause empty dotted ntuple_feel chord text effect mix_change
                beat([None, None, None, None, note(3), note(0), None], duration=-2)
            ], []  # voice 2
        )
    ],
    [  # measure 6
        (  # track 1
            [  # voice 1: notes duration pause empty dotted ntuple_feel chord text effect mix_change
                beat([None, None, None, None, note(3), note(0), None], duration=-2)
            ], []  # voice 2
        )
    ],
    [  # measure 7
        (  # track 1
            [  # voice 1: notes duration pause empty dotted ntuple_feel chord text effect mix_change
                beat([None, None, None, None, note(3), note(0), None], duration=-2)
            ], []  # voice 2
        )
    ],
    [  # measure 8
        (  # track 1
            [  # voice 1: notes duration pause empty dotted ntuple_feel chord text effect mix_change
                beat([None, None, None, None, note(3), note(0), None], duration=-2)
            ], []  # voice 2
        )
    ],
]

write_gp5(measures, tracks, beats, tempo=133, outfile="out.gp5",
          # header=Header('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'),
          # lyrics=Lyrics(1, [(1, "I have no"), (4, "Idea what you talk"), (1, "about, wtf")]),
          )
