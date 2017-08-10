import argparse
from copy import copy
import os
import sys

module_path = os.path.abspath('..')
if module_path not in sys.path:
    sys.path.append(module_path)
from music_transcription.fileformat.guitar_pro.gp5_reader import GP5File
from music_transcription.fileformat.guitar_pro.gp5_writer import write_gp5
from music_transcription.fileformat.guitar_pro.gp5_comparison import meta_comparison, compare

TUNING = (64, 59, 55, 50, 45, 40)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Transcribe MIDI file to notes and tabs (GP5)')
parser.add_argument('file1', help='Path to file 1 (.gp5)')
parser.add_argument('file2', help='Path to file 2 (.gp5)')
parser.add_argument('--track1', type=int, default=1, help='Track of file 1')
parser.add_argument('--track2', type=int, default=1, help='Track of file 2')
parser.add_argument("--compare_positions", type=str2bool, nargs='?', const=True, default=False,
                    help="Compare not only pitches but also string & fret positions.")
parser.add_argument('--outfile', help='Path to output file (.gp5)')
# parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output')

args = parser.parse_args()
assert os.path.isfile(args.file1), 'File 1 not found'
assert os.path.isfile(args.file2), 'File 2 not found'
assert args.track1 > 0, 'Invalid track number for track 1, should be > 0'
assert args.track2 > 0, 'Invalid track number for track 2, should be > 0'

filename1 = os.path.basename(args.file1).rstrip('.gp5')
filename2 = os.path.basename(args.file2).rstrip('.gp5')

if args.outfile is None:
    out_path = filename1 + ' VS ' + filename2 + '.gp5'
else:
    out_path = args.outfile

gp5file1 = GP5File(args.file1)
gp5file2 = GP5File(args.file2)

assert args.track1 <= gp5file1.nTracks, 'Invalid track number for track 1, should be <= number of tracks'
assert args.track2 <= gp5file2.nTracks, 'Invalid track number for track 2, should be <= number of tracks'
assert [gp5file1.tracks[args.track1 - 1].channel, gp5file2.tracks[args.track2 - 1].channel].count(10) != 1, \
    'Invalid tracks, cannot compare a melodic to a percussion track'

common_markers = meta_comparison(gp5file1, gp5file2)
measures, beats = compare(gp5file1, gp5file2, args.track1, args.track2, TUNING, common_markers, args.compare_positions)
tracks = [gp5file1.tracks[args.track1 - 1], copy(gp5file1.tracks[args.track1 - 1]), gp5file2.tracks[args.track2 - 1]]
tracks[0].name = 'common notes'
tracks[1].name = 'differences file 1'
tracks[2].name = 'differences file 2'

write_gp5(measures, tracks, beats, tempo=gp5file1.tempo, outfile=out_path)
