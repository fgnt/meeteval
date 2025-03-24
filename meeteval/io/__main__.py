import os
import sys
from stat import S_ISFIFO
from pathlib import Path
import argparse
import meeteval


def _is_piping_in():
    """
    Checks whether input is being piped into the command.
    """
    return S_ISFIFO(os.fstat(0).st_mode)


def convert(input_files, output_file, input_format, output_format, parser, **kwargs):
    """
    Converts one or multiple `input_files` from `input_format` to 
    `output_format` and writes the result to `output_file`. All input files must 
    have the same format. The `kwargs` are used to fill missing keys in the 
    destination format. If a value in `kwargs` is a string, it is formatted with
    the segment information and the file stem of the input file in the key 
    `'filestem'`.
    """
    data = []

    for f in input_files:
        d = meeteval.io.load(f, format=input_format).to_seglst()
        
        extra = {}

        import stat
        if isinstance(f, (str, Path)) and stat.S_ISREG(os.stat(f).st_mode):
            extra['filestem'] = Path(f).stem
        else:
            for k, v in kwargs.items():
                if '{filestem}' in v:
                    parser.error(
                        f'"{{filestem}}" not available for key {k}="{v}" ' 
                        f'and file {f}: not a regular file. filestem '
                        f'is only meaningful for regular files.'
                    )

        if kwargs:
            for segment in d:
                for k, v in kwargs.items():
                    if isinstance(v, str):
                        try:
                            v = v.format(**segment, **extra)
                        except KeyError as e:
                            parser.error(
                                f'Key "{e.args[0]}" not found in segment '
                                f'({list(segment.keys())}) or extra keys '
                                f'({list(extra.keys())}) '
                                f'for file "{f}" and key "{k}"="{v}".'
                            )
                    segment[k] = v
                    
        data.append(d)

    data = meeteval.io.SegLST.merge(*data)

    meeteval.io.dump(data, output_file, output_format)


def cli():
    from meeteval.wer.__main__ import SmartFormatter

    piping_in = _is_piping_in()

    parser = argparse.ArgumentParser(
        formatter_class=SmartFormatter,
        description='Convert between different file formats supported by meeteval.\n\n'
                    'Use "meeteval-io SUBCOMMAND --help" for further usage details.'
    )
    commands = parser.add_subparsers(
        title='Subcommands',
    )

    for reader in ['ctm', 'stm', 'seglst', 'rttm']:
        for writer in ['stm', 'seglst', 'rttm']: # 'ctm' is not supported as output format
            command_parser = commands.add_parser(
                f'{reader}2{writer}',
                help=f'Converts from {reader.upper()} to {writer.upper()}.',
                description=f'Converts from {reader.upper()} to {writer.upper()}. '
                            f'Merges multiple input files.\n\n'
                            f'Piping into the command is supported for a single '
                            f'input file and "-" is required to pipe out: '
                            f'"cat file.stm | meeteval-io stm2rtm - | ..."',
                formatter_class=SmartFormatter,
                add_help=False,
            )
            command_parser.add_argument(
                '--help', 
                help='Show this help message and exit',
                action='help',
                default=argparse.SUPPRESS,
            )
            command_parser.add_argument(
                '--force', '-f',
                help='Overwrite the output file if it exists.',
                action='store_true',
                default=False,
            )
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
                    help='The speaker name to use for the CTM. Anything in '
                         'curly braces is replaced by the segment information, '
                         'for example "--speaker \'{session_id}\'". '
                         'Defaults to the file stem.',
                    required=piping_in,
                    default=None if piping_in else '{filestem}',
                )
            if reader == 'rttm' and writer in ('stm', 'seglst'):
                command_parser.add_argument(
                    '--words',
                    help='Placeholder for missing words. Anything in curly '
                         'braces is replaced by the segment information, '
                         'for example "{session_id}-{speaker}".',
                    default='<NA>',
                )

    args = dict(vars(parser.parse_args()))
    if not args['force']:
        if os.path.exists(args['output_file']):
            args['parser'].error(
                f'Output file "{args["output_file"]}" already exists. '
                'Use --force / -f to overwrite.'
            )
    if piping_in:
        if args['input_files']:
            import subprocess
            args['parser'].error(
                f'Input files ({subprocess.list2cmdline(args["input_files"])}) '
                f'are not allowed when piping into the command.'
            )
        args['input_files'] = [sys.stdin]
    if args['output_file'] == '-':
        args['output_file'] = sys.stdout
    args.pop('force')
    convert(**args)
