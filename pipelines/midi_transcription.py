import argparse
import os
import sys

module_path = os.path.abspath('..')
if module_path not in sys.path:
    sys.path.append(module_path)
from music_transcription.fileformat.midi2gp5 import convert_midi2gp5

SHORTEST_NOTES = {
    '1/1': 4.0, '1/2': 2.0, '1/4': 1.0, '1/8': 0.5, '1/16': 0.25, '1/32': 0.125, '1/64': 0.0625
}


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Transcribe MIDI file to notes and tabs (GP5)')
parser.add_argument('path_to_midi', help='Path to MIDI file')
parser.add_argument('--shortest_note', default='1/16', choices=SHORTEST_NOTES.keys(), help='Shortest possible note')
parser.add_argument("--force_drums", type=str2bool, nargs='?', const=True, default=False, help="Force drum track.")
parser.add_argument('--instrument_id', type=int, default=25, help='Instrument id for GP5 file')
parser.add_argument('--time_signature', default='4/4',
                    help='Time signature (4/4, 3/4, 6/8, etc.). If specified in the midi file, argument is ignored.')
parser.add_argument('--tempo', type=int, default=120,
                    help="Tempo of the piece in BPM. If specified in the midi file, argument is ignored.")
parser.add_argument('--path_to_gp5', help='Output path of GP5 file')
parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output')

args = parser.parse_args()
assert os.path.isfile(args.path_to_midi), 'MIDI file not found'
assert args.instrument_id > 0, 'Instrument ID is invalid, should be > 0'
assert args.time_signature.count("/") == 1, 'Invalid time signature. Should be numerator/denominator'
numerator, denominator = [int(x) for x in args.time_signature.split("/")]
assert args.tempo > 0, 'Invalid tempo'

filename = os.path.basename(args.path_to_midi).rstrip('.mid')

if args.path_to_gp5 is None:
    path_to_gp5 = filename + '.gp5'
else:
    path_to_gp5 = args.path_to_gp5

convert_midi2gp5(args.path_to_midi, path_to_gp5, shortest_note=SHORTEST_NOTES[args.shortest_note],
                 init_tempo=args.tempo, time_signature=(numerator, denominator), force_drums=args.force_drums,
                 default_instrument=args.instrument_id, verbose=args.verbose)
