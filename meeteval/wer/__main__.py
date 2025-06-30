import argparse
import dataclasses
import json
import logging
import os
import re
import decimal
import textwrap
from pathlib import Path

import meeteval.io
from meeteval.io.smart import _open
from meeteval.wer.wer import combine_error_rates, ErrorRate
import sys
import meeteval.wer

from meeteval.wer.wer.time_constrained import pseudo_word_level_strategies


def _dump(obj, path: 'Path | str', default_suffix='.json'):
    """
    Dumps the `obj` to `path`. Parses the suffix to find the file type.
    When a suffix is missing, use json.

    >>> import tempfile
    >>> data = {
    ...     'a': 1,
    ...     'b': decimal.Decimal('3.9'),
    ...     'c': decimal.Decimal('100000000000.0173455555')
    ... }
    >>> def test(suffix):
    ...     with tempfile.TemporaryDirectory() as tmpdir:
    ...         _dump(data, f'{tmpdir}/test{suffix}')
    ...         print(Path(f'{tmpdir}/test{suffix}').read_text())

    >>> test('.json')
    {
      "a": 1,
      "b": 3.9,
      "c": 100000000000.0173455555
    }
    >>> test('.yaml')
    a: 1
    b: !!python/object/apply:decimal.Decimal
    - '3.9'
    c: !!python/object/apply:decimal.Decimal
    - '100000000000.0173455555'
    <BLANKLINE>

    """
    path = Path(path)
    with _open(path, 'w') as fd:
        suffix = path.suffix
        if suffix == '':
            suffix = default_suffix
        if suffix == '.json':
            # CB: Tried json, simplejson, orjson and ujson.
            #     Only simplejson was able to handle decimal.Decimal
            #     without changing the value.
            import simplejson
            simplejson.dump(obj, fd, indent=2, sort_keys=False)
        elif suffix == '.yaml':
            import yaml
            yaml.dump(obj, fd, sort_keys=False)
        else:
            raise NotImplementedError(f'Unknown file ext: {suffix}')

    if str(path) != '-':
        logging.info(f'Wrote: {path}')


def _load(path: Path):
    with _open(path, 'r') as fd:
        if path.suffix == '.json':
            return json.load(fd)
        elif path.suffix == '.yaml':
            import yaml
            return yaml.load(fd)
        else:
            raise NotImplementedError(f'Unknown file ext: {path.suffix}')


def _get_parent_stem(hypothesis_paths: 'list[Path]'):
    hypothesis_paths = [Path(p).resolve() for p in hypothesis_paths]

    if len(hypothesis_paths) == 1:
        parent, stem = hypothesis_paths[0].parent, hypothesis_paths[0].stem
    else:
        # Find the common parent and "stem" when we have multiple files
        parent = os.path.commonpath(hypothesis_paths)

        stems = [p.stem for p in hypothesis_paths]
        prefix = os.path.commonprefix(stems)
        postfix = os.path.commonprefix([s[::-1] for s in stems])[::-1]

        stem = f'{prefix}{postfix}'

    return parent, stem


def _save_results(
        per_reco,
        hypothesis_paths: 'list[Path]',
        per_reco_out: str,
        average_out: str,
        wer_name: str = 'WER',
):
    """Saves the results.
    """
    parent, stem = _get_parent_stem(hypothesis_paths)

    def to_str(example_id):
        if isinstance(example_id, str):
            return example_id
        elif isinstance(example_id, tuple):
            return '___'.join(example_id)
        else:
            raise TypeError(type(example_id), example_id)
        
    # Save details
    _dump({
        to_str(example_id): dataclasses.asdict(error_rate)
        for example_id, error_rate in per_reco.items()
    }, per_reco_out.format(parent=parent, stem=stem))

    # Compute and save average
    average = combine_error_rates(*per_reco.values())
    _dump(
        dataclasses.asdict(average),
        average_out.format(parent=parent, stem=stem),
    )
    if hasattr(average, 'scored_speaker_time'):
        error_time = average.missed_speaker_time + average.falarm_speaker_time + average.speaker_error_time
        logging.info(
            f'%{wer_name}: {average.error_rate:.2%} '
            f'[ {error_time:.2f}s / {average.scored_speaker_time:.2f}s, '
            f'{average.missed_speaker_time:.2f}s missed, '
            f'{average.falarm_speaker_time:.2f}s falarm, '
            f'{average.speaker_error_time:.2f}s spk error ]'
        )
    else:
        logging.info(
            f'%{wer_name}: {average.error_rate:.2%} '
            f'[ {average.errors} / {average.length}, '
            f'{average.insertions} ins, '
            f'{average.deletions} del, '
            f'{average.substitutions} sub ]'
        )
    return average


def sisower(
        reference, hypothesis,
        regex=None,
        normalizer=None,
        partial=False,
        average_out='{parent}/{stem}_wer.json',
        per_reco_out='{parent}/{stem}_wer_per_reco.json',
):
    """Computes the "standard" WER (SISO WER).
    
    Filenames / session_ids must be unique and there must be exactly one 
    hypothesis per reference.
    """
    results = meeteval.wer.sisower(
        reference, hypothesis,
        regex=regex,
        partial=partial,
        normalizer=normalizer,
    )
    _save_results(results, hypothesis, per_reco_out, average_out, wer_name='SISO-WER')


def orcwer(
        reference, hypothesis,
        average_out='{parent}/{stem}_orcwer.json',
        per_reco_out='{parent}/{stem}_orcwer_per_reco.json',
        regex=None,
        reference_sort='segment_if_available',
        hypothesis_sort='segment_if_available',
        uem=None,
        partial=False,
        normalizer=None,
):
    """Computes the Optimal Reference Combination Word Error Rate (ORC WER)"""
    results = meeteval.wer.orcwer(
        reference, hypothesis, regex=regex,
        reference_sort=reference_sort, hypothesis_sort=hypothesis_sort,
        uem=uem,
        partial=partial,
        normalizer=normalizer,
    )
    _save_results(results, hypothesis, per_reco_out, average_out, wer_name='ORC-WER')


def greedy_orcwer(
        reference, hypothesis,
        average_out='{parent}/{stem}_orcwer.json',
        per_reco_out='{parent}/{stem}_orcwer_per_reco.json',
        regex=None,
        reference_sort='segment_if_available',
        hypothesis_sort='segment_if_available',
        uem=None,
        partial=False,
        normalizer=None,
):
    """Computes the Optimal Reference Combination Word Error Rate (ORC WER)
    using a greedy algorithm. This algorithm is a lot faster than the optimal
    algorithm, but IS NOT GUARANTEED TO PRODUCE THE SAME RESULT!"""
    results = meeteval.wer.api.greedy_orcwer(
        reference, hypothesis, regex=regex,
        reference_sort=reference_sort, hypothesis_sort=hypothesis_sort,
        uem=uem,
        partial=partial,
        normalizer=normalizer,
    )
    _save_results(results, hypothesis, per_reco_out, average_out, wer_name='greedy ORC-WER')


def cpwer(
        reference, hypothesis,
        average_out='{parent}/{stem}_cpwer.json',
        per_reco_out='{parent}/{stem}_cpwer_per_reco.json',
        regex=None,
        reference_sort='segment_if_available',
        hypothesis_sort='segment_if_available',
        uem=None,
        normalizer=None,
        partial=False,
):
    """Computes the Concatenated minimum-Permutation Word Error Rate (cpWER)"""
    results = meeteval.wer.cpwer(
        reference, hypothesis, regex=regex,
        reference_sort=reference_sort, hypothesis_sort=hypothesis_sort,
        uem=uem, partial=partial, normalizer=normalizer,
    )
    _save_results(results, hypothesis, per_reco_out, average_out, wer_name='cpWER')


def mimower(
        reference, hypothesis,
        average_out='{parent}/{stem}_mimower.json',
        per_reco_out='{parent}/{stem}_mimower_per_reco.json',
        regex=None,
        reference_sort='segment_if_available',
        hypothesis_sort='segment_if_available',
        uem=None,
        normalizer=None,
        partial=False,
):
    """Computes the MIMO WER"""
    results = meeteval.wer.mimower(
        reference, hypothesis, regex=regex,
        reference_sort=reference_sort, hypothesis_sort=hypothesis_sort,
        uem=uem, partial=partial, normalizer=normalizer,
    )
    _save_results(results, hypothesis, per_reco_out, average_out, wer_name='MIMO-WER')


def tcmimower(
        reference, hypothesis,
        *,
        collar,
        average_out='{parent}/{stem}_tcmimower.json',
        per_reco_out='{parent}/{stem}_tcmimower_per_reco.json',
        hyp_pseudo_word_timing='character_based_points',
        ref_pseudo_word_timing='character_based',
        regex=None,
        hypothesis_sort='segment',
        reference_sort='segment',
        uem=None,
        normalizer=None,
        partial=False,
):
    """Computes the time-constrained MIMO WER (tcMIMO WER)"""
    results = meeteval.wer.tcmimower(
        reference, hypothesis, regex=regex,
        collar=collar,
        hyp_pseudo_word_timing=hyp_pseudo_word_timing,
        ref_pseudo_word_timing=ref_pseudo_word_timing,
        hypothesis_sort=hypothesis_sort,
        reference_sort=reference_sort,
        uem=uem, partial=partial,
        normalizer=normalizer,
    )
    _save_results(results, hypothesis, per_reco_out, average_out, wer_name='tcMIMO-WER')




def tcpwer(
        reference, hypothesis,
        *,
        collar,
        average_out='{parent}/{stem}_tcpwer.json',
        per_reco_out='{parent}/{stem}_tcpwer_per_reco.json',
        hyp_pseudo_word_timing='character_based_points',
        ref_pseudo_word_timing='character_based',
        regex=None,
        reference_sort='segment',
        hypothesis_sort='segment',
        uem=None,
        normalizer=None,
        partial=False,
):
    """Computes the time-constrained minimum permutation WER"""
    results = meeteval.wer.tcpwer(
        reference, hypothesis, regex=regex,
        ref_pseudo_word_timing=ref_pseudo_word_timing,
        hyp_pseudo_word_timing=hyp_pseudo_word_timing,
        collar=collar,
        reference_sort=reference_sort,
        hypothesis_sort=hypothesis_sort,
        uem=uem, normalizer=normalizer, partial=partial,
    )
    _save_results(results, hypothesis, per_reco_out, average_out, wer_name='tcpWER')


def tcorcwer(
        reference, hypothesis,
        *,
        collar,
        average_out='{parent}/{stem}_tcorcwer.json',
        per_reco_out='{parent}/{stem}_tcorcwer_per_reco.json',
        regex=None,
        hyp_pseudo_word_timing='character_based_points',
        ref_pseudo_word_timing='character_based',
        hypothesis_sort='segment',
        reference_sort='segment',
        uem=None,
        normalizer=None,
        partial=False,
):
    """Computes the time-constrained ORC WER (tcORC WER)"""
    results = meeteval.wer.tcorcwer(
        reference, hypothesis, regex=regex,
        collar=collar,
        hyp_pseudo_word_timing=hyp_pseudo_word_timing,
        ref_pseudo_word_timing=ref_pseudo_word_timing,
        hypothesis_sort=hypothesis_sort,
        reference_sort=reference_sort,
        uem=uem, partial=partial,
        normalizer=normalizer,
    )
    _save_results(results, hypothesis, per_reco_out, average_out, wer_name='tcORC-WER')


def greedy_tcorcwer(
        reference, hypothesis,
        *,
        collar,
        average_out='{parent}/{stem}_greedy_tcorcwer.json',
        per_reco_out='{parent}/{stem}_greedy_tcorcwer_per_reco.json',
        regex=None,
        hyp_pseudo_word_timing='character_based_points',
        ref_pseudo_word_timing='character_based',
        hypothesis_sort='segment',
        reference_sort='segment',
        uem=None,
        normalizer=None,
        partial=False,
):
    """Computes the time-constrained ORC WER (tcORC WER)"""
    results = meeteval.wer.greedy_tcorcwer(
        reference, hypothesis, regex=regex,
        collar=collar,
        hyp_pseudo_word_timing=hyp_pseudo_word_timing,
        ref_pseudo_word_timing=ref_pseudo_word_timing,
        hypothesis_sort=hypothesis_sort,
        reference_sort=reference_sort,
        uem=uem, partial=partial,
        normalizer=normalizer,
    )
    _save_results(results, hypothesis, per_reco_out, average_out, wer_name='greedy-tcORC-WER')


def greedy_dicpwer(
        reference, hypothesis,
        average_out='{parent}/{stem}_greedy_dicpwer.json',
        per_reco_out='{parent}/{stem}_greedy_dicpwer_per_reco.json',
        regex=None,
        reference_sort='segment_if_available',
        hypothesis_sort='segment_if_available',
        uem=None,
        partial=False,
        normalizer=None,
):
    """Computes the greedy DI-cpWER"""
    results = meeteval.wer.api.greedy_dicpwer(
        reference, hypothesis, regex=regex,
        reference_sort=reference_sort, hypothesis_sort=hypothesis_sort,
        uem=uem,
        partial=partial,
        normalizer=normalizer,
    )
    _save_results(results, hypothesis, per_reco_out, average_out, wer_name='greedy-DI-cpWER')


def greedy_ditcpwer(
        reference, hypothesis,
        *,
        collar,
        average_out='{parent}/{stem}_greedy_ditcpwer.json',
        per_reco_out='{parent}/{stem}_greedy_ditcpwer_per_reco.json',
        regex=None,
        hyp_pseudo_word_timing='character_based_points',
        ref_pseudo_word_timing='character_based',
        hypothesis_sort='segment',
        reference_sort='segment',
        uem=None,
        partial=False,
        normalizer=None,
):
    """Computes the time-constrained diarization-invariant cpWER (greedy DI-tcpWER)"""
    results = meeteval.wer.api.greedy_ditcpwer(
        reference, hypothesis, regex=regex,
        collar=collar,
        hyp_pseudo_word_timing=hyp_pseudo_word_timing,
        ref_pseudo_word_timing=ref_pseudo_word_timing,
        hypothesis_sort=hypothesis_sort,
        reference_sort=reference_sort,
        uem=uem, partial=partial,
        normalizer=normalizer,
    )
    _save_results(results, hypothesis, per_reco_out, average_out, wer_name='greedy-DI-tcpWER')


def _merge(
        files: 'list[str]',
        out: str = '-',
        average: bool = None,
        regex: str = None,
):
    # Load input files
    files = [Path(f) for f in files]
    data = [_load(f) for f in files]

    if regex is not None:
        regex = re.compile(regex)

    import meeteval
    ers = []

    for d in data:
        if 'errors' in d:  # Average file
            assert average is not False, average
            average = True  # A single average file forces to do an average
            ers.append([None, ErrorRate.from_dict(d)])
        else:
            for k, v in d.items():  # Details file
                if regex is not None and not regex.fullmatch(k):
                    continue
                if 'errors' in v:
                    ers.append([k, ErrorRate.from_dict(v)])

    if average:
        er = meeteval.wer.combine_error_rates(*[er for _, er in ers])
        out_data = dataclasses.asdict(er)
    else:
        out_data = {
            k: dataclasses.asdict(er)
            for k, er in ers
        }
        assert len(out_data) == len(ers), (len(out_data), len(ers), 'Duplicate filenames')

    _dump(out_data, Path(out), default_suffix=files[0].suffix)


def merge(files, out='-'):
    """Merges multiple (per-reco or averaged) files"""
    return _merge(files, out, average=None)


def average(files, out='-', regex=None):
    """Computes the average over one or multiple per-reco files"""
    return _merge(files, out, average=True, regex=regex)


def normalize(
        file,
        out,
        normalizer='lower,rm(.?!,)',
):
    """
    Normalize the words in the transcript with the normalizer `normalizer`.
    """
    from meeteval.wer.api import normalize
    meeteval.io.dump(normalize(file, normalizer=normalizer), out)

class SmartFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """
    https://stackoverflow.com/a/22157136/5766934
    """

    def _split_lines(self, text, width):
        import textwrap
        return [
            tt
            for i, t in enumerate(text.split('\n'))
            for tt in textwrap.wrap(t, width, subsequent_indent='  ' if i > 0 else '')
        ]
    
    def _fill_text(self, text: str, width: int, indent: str) -> str:
        """
        Extends `_fill_text` to work with multiple paragraphs, separated by \n\n.
        Does not wrap indented paragraphs.
        This function is used to format the (long) command descriptions at the top of 
        the help text.
        """
        import textwrap
        text = textwrap.dedent(text)
        paragraphs = text.split('\n\n')
        paragraphs = [
            p
            if p.startswith(' ') else
            textwrap.fill(p, width, initial_indent=indent, subsequent_indent=indent)
            for p in paragraphs
        ]
        return '\n\n'.join(paragraphs)

class CLI:
    def __init__(self):

        # Define argument parser and commands
        self.parser = argparse.ArgumentParser(
            formatter_class=SmartFormatter
        )
        self.parser.add_argument('--version', action='store_true',
                                 help='Show version')

        # Logging and verbosity
        logging.addLevelName(100,
                             'SILENT')  # Add a level that creates no output
        self.parser.add_argument(
            '--log-level', help='Log level', default='INFO',
            choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL', 'SILENT']
        )

        self.commands = self.parser.add_subparsers(
            title='Subcommands',
        )

    @staticmethod
    def positive_number(x: str):
        if x.isdigit():
            # Positive integer
            return int(x)

        x = float(x)
        if x < 0:
            raise ValueError(f'Number must be positive, but got {x}')

        return x

    @staticmethod
    def str_or_bool(x: str):
        """Convert common boolean strings to bool and pass through other
        strings"""
        if x in ('true', 'True'):
            return True
        elif x in ('false', 'False'):
            return False
        else:
            return x

    class extend_action(argparse.Action):
        def __call__(self, parser, namespace, values, option_string):
            """Extend action for argparse to allow multiple arguments.
            Equal to the "extend" action in argparse that was added in Python 3.8
            """
            if not isinstance(values, list):
                values = [values]
            current = getattr(namespace, self.dest, []) or []
            setattr(namespace, self.dest, current + values)

    def add_argument(self, command_parser, name, p, command_name):
        if name == 'reference':
            command_parser.add_argument(
                '-r', '--reference',
                help='Reference file(s) in SegLST, STM or CTM format',
                nargs='+', action=self.extend_action,
                required=True,
            )
        elif name == 'hypothesis':
            command_parser.add_argument(
                '-h', '--hypothesis',
                help='Hypothesis file(s) in SegLST, STM or CTM format',
                nargs='+', action=self.extend_action,
                required=True,
            )
        elif name == 'average_out':
            command_parser.add_argument(
                '--average-out',
                help='Output file for the average file. {stem} is replaced '
                     'with the stem of the (first) hypothesis file. '
                     '"-" is interpreted as stdout. For example: "-.yaml" '
                     'prints to stdout in yaml format.'
            )
        elif name == 'per_reco_out':
            command_parser.add_argument(
                '--per-reco-out',
                help='Output file for the per_reco file. {stem} is replaced '
                     'with the stem of the (first) hypothesis file. '
                     '"-" is interpreted as stdout. For example: "-.yaml" '
                     'prints to stdout in yaml format.'
            )
        elif name == 'out':
            command_parser.add_argument(
                '-o', '--out',
                required=False,
            )
        elif name == 'collar':
            command_parser.add_argument(
                '--collar', type=self.positive_number,
                help='Collar applied to the hypothesis timings (e.g., 5 for 5 seconds)',
                required=True,
            )
        elif name == 'regex':
            command_parser.add_argument(
                '--regex',
                help='A regex pattern to select only particular filenames.'
            )
        elif name == 'hyp_pseudo_word_timing':
            command_parser.add_argument(
                '--hyp-pseudo-word-timing',
                choices=pseudo_word_level_strategies.keys(),
                help='Specifies how word-level timings are '
                     'determined from segment-level timing '
                     'for the hypothesis.\n'
                     'Choices:\n'
                     '- equidistant_intervals: Divide segment-level timing into equally sized intervals\n'
                     '- equidistant_points: Place time points equally spaded int the segment-level intervals\n'
                     '- full_segment: Use the full segment for each word that belongs to that segment\n'
                     '- character_based: Estimate the word length based on the number of characters\n'
                     '- character_based_points: Estimates the word length based on the number of characters and '
                     'creates a point in the center of each word\n'
                     '- none: Do not estimate word-level timings but assume that the provided timings are already '
                     'given on a word level.'
            )
        elif name == 'ref_pseudo_word_timing':
            command_parser.add_argument(
                '--ref-pseudo-word-timing',
                choices=pseudo_word_level_strategies.keys(),
                help='Specifies how word-level timings are '
                     'determined from segment-level timing '
                     'for the reference.\n'
                     'Choices:\n'
                     '- equidistant_intervals: Divide segment-level timing into equally sized intervals\n'
                     '- equidistant_points: Place time points equally spaded int the segment-level intervals\n'
                     '- full_segment: Use the full segment for each word that belongs to that segment.\n'
                     '- character_based: Estimate the word length based on the number of characters\n'
                     '- character_based_points: Estimates the word length based on the number of characters and '
                     'creates a point in the center of each word\n'
                     '- none: Do not estimate word-level timings but assume that the provided timings are already '
                     'given on a word level.'
            )
        elif name == 'reference_sort':
            command_parser.add_argument(
                '--reference-sort',
                choices=[True, False, 'word', 'segment'],
                type=self.str_or_bool,
                help='How to sort words/segments in the reference.\n'
                     'Choices:\n'
                     '- segment: Sort segments by start time and do not check word order\n'
                     '- False: Do not sort and do not check word order. Segment order is taken from input file '
                     'and sorting is up to the user\n'
                     '- True: Sort by segment start time and assert that the word-level timings are sorted by start '
                     'time. Only supported for time-constrained WERs\n'
                     '- word: sort words by start time. Only supported for time-constrained WERs\n'
                     '- segment_if_available: Same as segment, but only applies when timestamps are present in the '
                     'inputs. If no timestamps are present, falls back to False.'
            )
        elif name == 'hypothesis_sort':
            command_parser.add_argument(
                '--hypothesis-sort',
                choices=[True, False, 'word', 'segment'],
                type=self.str_or_bool,
                help='How to sort words/segments in the reference.\n'
                     'Choices:\n'
                     '- segment: Sort segments by start time and do not check word order\n'
                     '- False: Do not sort and do not check word order. Segment order is taken from input file '
                     'and sorting is up to the user\n'
                     '- True: Sort by segment start time and assert that the word-level timings are sorted by start '
                     'time. Only supported for time-constrained WERs\n'
                     '- word: sort words by start time. Only supported for time-constrained WERs\n'
                     '- segment_if_available: Same as segment, but only applies when timestamps are present in the '
                     'inputs. If no timestamps are present, falls back to False.'
            )
        elif name == 'uem':
            command_parser.add_argument(
                '--uem',
                help='UEM file that defines the scoring regions. Only supported when reference and hypothesis files '
                     'contain time-stamps.',
                nargs='+', action=self.extend_action,
            )
        elif name == 'normalizer':
            from meeteval.wer.normalizer import normalizers
            command_parser.add_argument(
                '--normalizer',
                help=textwrap.dedent(normalizers.__doc__),
                choices=[None, *normalizers.keys()],
            )
        elif name == 'partial':
            command_parser.add_argument(
                '--partial',
                action='store_true',
                help='Compute the metrics on the subset of sessions defined in the reference files.'
                     'Ignore any sessions present in the hypothesis but not in the reference.'
            )
        elif name == 'files':
            command_parser.add_argument('files', nargs='+')
        elif name == 'file':
            command_parser.add_argument('file')
        else:
            raise AssertionError("Error in command definition", name)

    def add_command(self, fn, command_name=None):
        if command_name is None:
            command_name = fn.__name__
        command_parser = self.commands.add_parser(
            command_name,
            # Use full docstring as description at the top of the help text (e.g., meeteval-wer cpwer --help)
            description=fn.__doc__,
            # Use first paragraph as short help text in the command list (e.g., meeteval-wer --help)
            help=fn.__doc__.split('\n\n')[0] if fn.__doc__ is not None else None,
            formatter_class=SmartFormatter, # Custom formatter for help and description texts
            add_help=False,
        )
        command_parser.add_argument(
            '--help', help='show this help message and exit',
            action='help',
            default=argparse.SUPPRESS,
        )

        # Get arguments from signature
        import inspect
        parameters = inspect.signature(fn).parameters

        for name, p in parameters.items():
            self.add_argument(command_parser, name, p, command_name)

        # Get defaults from signature
        command_parser.set_defaults(
            func=fn,
            **{
                name: p.default
                for name, p in parameters.items()
                if p.default is not inspect.Parameter.empty
            }
        )

    def run(self):
        # Parse arguments and find command to execute
        args = self.parser.parse_args()

        # Logging
        logging.basicConfig(level=args.log_level.upper(), format='%(levelname)s %(message)s', force=True)

        if hasattr(args, 'func'):
            kwargs = vars(args)
            fn = args.func
            # Pop also removes from args namespace
            kwargs.pop('func')
            kwargs.pop('version')
            kwargs.pop('log_level')
            return fn(**kwargs)

        if getattr(args, 'version', False):
            from meeteval import __version__
            print(__version__)
            return

        self.parser.print_help()


def cli():
    cli = CLI()

    cli.add_command(sisower)
    cli.add_command(sisower, 'wer') # Alias for backwards compatibility
    cli.add_command(cpwer)
    cli.add_command(orcwer)
    cli.add_command(greedy_orcwer)
    cli.add_command(mimower)
    cli.add_command(tcmimower)
    cli.add_command(tcpwer)
    cli.add_command(tcorcwer)
    cli.add_command(greedy_dicpwer)
    cli.add_command(greedy_ditcpwer)
    cli.add_command(greedy_tcorcwer)
    cli.add_command(merge)
    cli.add_command(average)
    cli.add_command(normalize)

    cli.run()


if __name__ == '__main__':
    cli()
