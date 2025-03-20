import os
import sys
from stat import S_ISFIFO
from pathlib import Path
import argparse
import meeteval

# formats = {
#     'ctm': meeteval.io.ctm.CTM,
#     'stm': meeteval.io.stm.STM,
#     'seglst': meeteval.io.seglst.SegLST,
#     'rttm': meeteval.io.rttm.RTTM,
# }

# d = meeteval.io.load('').to_seglst()
# for s in d:
#     s['words'] = '<NA>'
# meeteval.io.STM.new(d).dump('.stm')


def is_piping_in():
    """
    Checks whether input is being piped into the command.
    """
    return S_ISFIFO(os.fstat(0).st_mode)


def convert(input_files, output_file, input_format, output_format, **kwargs):
    data = []

    for f in input_files:
        d = meeteval.io.load(f, format=input_format).to_seglst()
        
        extra = {}

        if isinstance(f, (str, Path)):
            extra['filestem'] = Path(f).stem

        if kwargs:
            for segment in d:
                for k, v in kwargs.items():
                    if isinstance(v, str):
                        v = v.format(**segment, **extra)
                    segment[k] = v
                    
        data.append(d)

    data = meeteval.io.SegLST.merge(*data)

    meeteval.io.dump(data, output_file, output_format)


def cli():
    from meeteval.wer.__main__ import SmartFormatter

    piping_in = is_piping_in()

    parser = argparse.ArgumentParser(
        formatter_class=SmartFormatter,
    )
    commands = parser.add_subparsers(
        title='Subcommands',
    )

    for reader in ['ctm', 'stm', 'seglst', 'rttm']:
        for writer in ['stm', 'seglst', 'rttm']: # 'ctm' is not supported as output format
            command_parser = commands.add_parser(
                f'{reader}2{writer}',
                help=f'Converts from {reader.upper()} to {writer.upper()}.',
                formatter_class=SmartFormatter,
                add_help=False,
            )
            command_parser.add_argument(
                '--help', 
                help='show this help message and exit',
                action='help',
                default=argparse.SUPPRESS,
            )
            command_parser.add_argument(
                '--force', '-f',
                help='Overwrite the output file if it exists.',
                action='store_true',
                default=False,
            )
            # When piping into the command, we do not want to allow 
            # additional input files
            command_parser.add_argument(
                'input_files', 
                nargs='+' if not piping_in else '*',
                help='The input files.',
            )
            command_parser.add_argument(
                'output_file', type=str,
                help='The output file. "-" means stdout.'
            )
            command_parser.set_defaults(
                input_format=reader,
                output_format=writer,
                parser=command_parser,
            )

            # Special cases
            if reader == 'ctm':
                command_parser.add_argument(
                    '--speaker',
                    type=str,
                    help='The speaker name to use for the CTM. Defaults to the file stem.',
                    required=piping_in,
                    default=None if piping_in else '{filestem}',
                )
            if reader == 'rttm' and writer in ('stm', 'seglst'):
                command_parser.add_argument(
                    '--words',
                    default='<NA>',
                )

    args = dict(vars(parser.parse_args()))
    if not args['force']:
        if os.path.exists(args['output_file']):
            args['arser'].error(
                f'Output file "{args["output_file"]}" already exists. '
                'Use --force / -f to overwrite.'
            )
    if piping_in:
        if args['input_files']:
            import subprocess
            args['parser'].error(f'Input files ({subprocess.list2cmdline(args["input_files"])}) are not allowed when piping into the command.')
        args['input_files'] = [sys.stdin]
    if args['output_file'] == '-':
        args['output_file'] = sys.stdout
    args.pop('parser')
    args.pop('force')
    convert(**args)
