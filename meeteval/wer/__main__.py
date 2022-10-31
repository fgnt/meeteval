"""Computes the Concatenated minimum Permutation Word Error Rate (cpWER)"""
import sys
import argparse
import dataclasses
import json
from pathlib import Path

from meeteval.cli.file_io.ctm import CTMGroup
from meeteval.cli.file_io.stm import STM
from meeteval.wer.wer import cp_word_error_rate, orc_word_error_rate, mimo_word_error_rate, combine_error_rates
from paderbox.utils.nested import nested_merge


def dump(obj, fd):
    """
    Dumps the obj to fd. Parses the suffix to find the file type.
    When a suffix is missing, use json.
    """
    suffix = Path(fd.name).suffix
    if suffix == '.json' or suffix == '':
        json.dump(obj, fd, indent=2, sort_keys=False)
    elif suffix == '.yaml':
        import yaml
        yaml.dump(obj, fd, sort_keys=False)
    else:
        raise NotImplemented(f'Unknown file ext: {suffix}')
    print(f'Wrote: {fd.name}', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="MeetEval cpWER", add_help=False)
    parser.add_argument(
        '--help', help='show this help message and exit',
        action='help',
        default=argparse.SUPPRESS,
    )
    parser.add_argument(
        '-r', help='Reference file in STM format',
        type=argparse.FileType('r'),
    )
    parser.add_argument(
        '-h', help='Hypothesis file in STM format',
        type=argparse.FileType('r'),
        # nargs='+',
        action='append'
    )
    parser.add_argument(
        '--orc', help='Activate ORC WER computation',
        default=False, action='store_true',
    )
    parser.add_argument(
        '--mimo', help='Activate MIMO WER computation',
        default=False, action='store_true',
    )
    parser.add_argument(
        '--cp', help='Activate cpWER computation',
        default=False, action='store_true',
    )
    parser.add_argument(
        '--json', help='Activate json print format (default: Human readable)',
        default=False, action='store_true',
    )
    parser.add_argument(
        '--yaml', help='Activate yaml print format (default: Human readable)',
        default=False, action='store_true',
    )
    parser.add_argument(
        '--output-details',
        help='Output file for detailed WER statistics for each example in JSON format',
        type=argparse.FileType('w'),
        default='details.json',
    )
    parser.add_argument(
        '--output-average',
        help='Output file for average WER statistics over the whole input in JSON format',
        type=argparse.FileType('w'),
        default='average.json',
    )
    args = parser.parse_args()

    wers = [wer for wer in ('orc', 'mimo', 'cp') if getattr(args, wer)]

    if len(wers) == 0:
        raise ValueError('Select at least one WER definition (--orc, --mimo or --cp)')

    # Load input files
    ref = STM.load(args.r)

    # Hypothesis can be an STM file or a collection of  CTM files. Detect
    # which one we have and load it
    if args.h[0].name.endswith('stm'):
        if len(args.h) > 1:
            raise ValueError()
        hyp = STM.load(args.h[0])
    elif args.h[0].name.endswith('ctm'):
        if any(not h.name.endswith('ctm') for h in args.h):
            raise ValueError()
        hyp = CTMGroup.load(args.h)
    else:
        raise ValueError()

    # Group by example IDs
    ref = ref.grouped_by_filename()
    hyp = hyp.grouped_by_filename()

    # Check that the input is valid
    if ref.keys() != hyp.keys():
        raise ValueError()

    # Compute cpWER for all examples
    details = {}
    for example_id in ref.keys():
        r = ref[example_id]
        h = hyp[example_id]
        result = {}
        if args.orc:
            result['orc'] = orc_word_error_rate(
                reference=r.utterance_transcripts(),
                hypothesis=[h_.merged_transcripts() for h_ in h.grouped_by_speaker_id()],
            )
        if args.cp:
            result['cp'] = cp_word_error_rate(
                reference=[r_.merged_transcripts() for r_ in r.grouped_by_speaker_id()],
                hypothesis=[h_.merged_transcripts() for h_ in h.grouped_by_speaker_id()],
            )
        if args.mimo:
            result['mimo'] = mimo_word_error_rate(
                reference=[r_.utterance_transcripts() for r_ in r.grouped_by_speaker_id()],
                hypothesis=[h_.merged_transcripts() for h_ in h.grouped_by_speaker_id()],
            )
        details[example_id] = result

    # Average
    average = {
        key: combine_error_rates(*[error_rates[key] for error_rates in details.values()])
        for key in wers
    }

    if args.json and args.yaml:
        raise RuntimeError(
            'You cannot request to print as --json and --yaml.\n'
            'Select one of them.')
    elif args.json:
        print(json.dumps({
            key: dataclasses.asdict(error_rate)
            for key, error_rate in average.items()
        }, sort_keys=False, indent=2))
    elif args.yaml:
        import yaml
        print(yaml.dump({
            key: dataclasses.asdict(error_rate)
            for key, error_rate in average.items()
        }, sort_keys=False))
    else:
        from pprint import pprint
        for k, v in average.items():
            print(f'{k}: ', end='')
            pprint(v)
        if not args.output_average and not args.output_details:
            print('For machine readable output use --json or --yaml.',
                  file=sys.stderr)
            print('Alternatively, use --output-average <file> or '
                  '--output-details <file> to write additionally to a file.',
                  file=sys.stderr)

    if args.output_average:
        dump({
            key: dataclasses.asdict(error_rate)
            for key, error_rate in average.items()
        }, args.output_average)

    # Save details as JSON/YAML
    if args.output_details:
        dump({
            example_id: {
                key: dataclasses.asdict(error_rate)
                for key, error_rate in error_rates.items()
            }
            for example_id, error_rates in details.items()
        }, args.output_details)


if __name__ == '__main__':
    main()
