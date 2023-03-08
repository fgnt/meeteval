import argparse
import dataclasses
import itertools
import json
import io
import os
import glob
from pathlib import Path
from typing import List

from meeteval.io.ctm import CTMGroup
from meeteval.io.stm import STM
from meeteval.wer.wer import (
    cp_word_error_rate,
    orc_word_error_rate,
    mimo_word_error_rate,
    combine_error_rates,
    ErrorRate,
    CPErrorRate,
)
import sys


def _dump(obj, path: Path, default_suffix='.json'):
    """
    Dumps the `obj` to `path`. Parses the suffix to find the file type.
    When a suffix is missing, use json.
    """
    if path.stem == '-':
        from contextlib import nullcontext
        p = nullcontext(sys.stdout)
    else:
        p = path.open('w')
    with p as fd:
        suffix = path.suffix
        if suffix == '':
           suffix = default_suffix
        if suffix == '.json':
            json.dump(obj, fd, indent=2, sort_keys=False)
        elif suffix == '.yaml':
            import yaml
            yaml.dump(obj, fd, sort_keys=False)
        else:
            raise NotImplemented(f'Unknown file ext: {suffix}')

    if path.stem != '-':
        print(f'Wrote: {path}', file=sys.stderr)


def _load(path: Path):
    with path.open('r') as fd:
        if path.suffix == '.json':
            return json.load(fd)
        elif path.suffix == '.yaml':
            import yaml
            return yaml.load(fd)
        else:
            raise NotImplemented(f'Unknown file ext: {path.suffix}')


def _load_reference(reference: [Path, str, io.TextIOBase, tuple, list]):
    """Loads a reference transcription file. Currently only STM supported"""
    return STM.load(reference)


def _load_hypothesis(hypothesis: List[Path]):
    """Loads the hypothesis. Supports one STM file or multiple CTM files
    (one per channel)"""
    if len(hypothesis) > 1:
        # We have multiple, only supported for ctm files
        hypothesis = list(map(Path, hypothesis))
        suffix = {h.suffixes[-1] for h in hypothesis}
        assert len(suffix) == 1, suffix
        filename = suffix.pop()
    else:
        hypothesis = hypothesis[0]
        if isinstance(hypothesis, io.TextIOBase):
            filename = hypothesis.name
        else:
            filename = str(hypothesis)

    if filename.endswith('.ctm'):
        if isinstance(hypothesis, list):
            return CTMGroup.load(hypothesis)
        else:
            return CTMGroup.load([hypothesis])
    elif filename.endswith('.stm'):
        return STM.load(hypothesis)
    elif filename.startswith('/dev/fd/'):
        # This is a pipe, i.e. python -m ... <(cat ...)
        # For now, assume it is an STM file
        return STM.load(hypothesis)
    else:
        raise RuntimeError(hypothesis, filename)


def _load_texts(reference_paths, hypothesis_paths):
    """Load and validate reference and hypothesis texts.

    Validation checks that reference and hypothesis have the same example IDs.
    """
    # Normalize and glob (for backwards compatibility) the path input
    reference_paths = list(itertools.chain.from_iterable(reference_paths))
    hypothesis_paths = list(itertools.chain.from_iterable(hypothesis_paths))

    def _glob(pathname):
        match = list(glob.glob(pathname))
        assert match, (pathname, match)
        return match

    reference_paths = [Path(file) for r in reference_paths for file in _glob(r)]
    hypothesis_paths = [Path(file) for h in hypothesis_paths for file in _glob(h)]

    # Load input files
    reference = _load_reference(reference_paths)

    # Hypothesis can be an STM file or a collection of  CTM files. Detect
    # which one we have and load it
    hypothesis = _load_hypothesis(hypothesis_paths)

    # Group by example IDs
    reference = reference.grouped_by_filename()
    hypothesis = hypothesis.grouped_by_filename()

    # Check that the input is valid
    if reference.keys() != hypothesis.keys():
        raise RuntimeError(
            'Keys of reference and hypothesis differ\n'
            f'hypothesis - reference: e.g. {list(set(hypothesis.keys()) - set(reference.keys()))[:5]}\n'
            f'reference - hypothesis: e.g. {list(set(reference.keys()) - set(hypothesis.keys()))[:5]}'
        )

    return reference, reference_paths, hypothesis, hypothesis_paths


def _get_parent_stem(hypothesis_paths: List[Path]):
    hypothesis_paths = [p.resolve() for p in hypothesis_paths]

    if len(hypothesis_paths) == 1:
        parent, stem = hypothesis_paths[0].parent, hypothesis_paths[0].stem
    else:
        parent = os.path.commonpath(hypothesis_paths)

        stems = [p.stem for p in hypothesis_paths]
        prefix = os.path.commonprefix(stems)
        postfix = os.path.commonprefix([s[::-1] for s in stems])[::-1]

        stem = f'{prefix}{postfix}'

    return parent, stem


def _save_results(
        per_reco,
        hypothesis_paths: List[Path],
        wer_suffix: str,
        per_reco_out: str,
        average_out: str,
):
    """Saves the results.
    """

    parent, stem = _get_parent_stem(hypothesis_paths)

    average_output = Path(average_out.format(
        parent=f'{parent}/',
        wer=wer_suffix,
        stem=stem,
    ))

    per_reco_output = Path(per_reco_out.format(
        parent=f'{parent}/',
        wer=wer_suffix,
        stem=stem,
    ))

    # Save details as JSON
    _dump({
        example_id: dataclasses.asdict(error_rate)
        for example_id, error_rate in per_reco.items()
    }, per_reco_output)

    # Compute and save average
    average = combine_error_rates(*per_reco.values())
    _dump(dataclasses.asdict(average), average_output)


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('--version', action='store_true', help='Show version')

commands = parser.add_subparsers(title='Subcommands')

common_wer_parser = argparse.ArgumentParser(add_help=False)
common_wer_parser.add_argument(
    '-r', '--reference', help='Reference file in STM format',
    nargs='+',
    action='append',
    # type=argparse.FileType('r'),
)
common_wer_parser.add_argument(
    '-h', '--hypothesis', help='Hypothesis file in STM format',
    # type=argparse.FileType('r'),
    nargs='+',
    action='append',
)
common_wer_parser.add_argument(
    '--average-out',
    default='{parent}/{stem}_{wer}.json',
    help='Output file for the average file. {stem} is replaced with the stem '
         'of the (first) hypothesis file and {wer} is replaced with the wer'
         'type (equal to the command name). '
         '"-" is interpreted as stdout. For example: "-.yaml" prints to '
         'stdout in yaml format.'
)
common_wer_parser.add_argument(
    '--per-reco-out', default='{parent}/{stem}_{wer}_per_reco.json',
    help='Output file for the per_reco file. {stem} is replaced with the stem '
         'of the (first) hypothesis file and {wer} is replaced with the wer'
         'type (equal to the command name). '
)

common_merge_parser = argparse.ArgumentParser(add_help=False)
common_merge_parser.add_argument(
    '-o', '--out',
    required=False, default='-',
    # type=argparse.FileType('w')
)
common_merge_parser.add_argument(
    'files', nargs='+',
    # type=argparse.FileType('r')
)


def command(name, help, parent_parsers):
    def _command(fn):
        command_parser = commands.add_parser(
            name,
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            help=help,
            parents=parent_parsers,
        )
        command_parser.add_argument(
            '--help', help='show this help message and exit',
            action='help',
            default=argparse.SUPPRESS,
        )
        command_parser.set_defaults(func=fn)
        return fn

    return _command


@command(
    'orcwer',
    help='Computes the Optimal Reference Combination Word Error Rate (ORC WER)',
    parent_parsers=[common_wer_parser],
)
def orcwer(reference, hypothesis, average_out, per_reco_out):
    reference, _, hypothesis, hypothesis_paths = _load_texts(reference, hypothesis)
    _save_results({
        example_id: orc_word_error_rate(
            reference=reference[example_id].utterance_transcripts(),
            hypothesis={
                k: h.merged_transcripts()
                for k, h in hypothesis[example_id].grouped_by_speaker_id().items()
            },
        )
        for example_id in reference.keys()
    }, hypothesis_paths, 'orcwer', per_reco_out, average_out)


@command(
    'cpwer',
    help='Computes the Concatenated minimum-Permutation Word Error Rate '
         '(cpWER)',
    parent_parsers=[common_wer_parser],
)
def cpwer(reference, hypothesis, average_out, per_reco_out):
    reference, _, hypothesis, hypothesis_paths = _load_texts(reference, hypothesis)
    _save_results({
        example_id: cp_word_error_rate(
            reference={
                k: r.merged_transcripts()
                for k, r in reference[example_id].grouped_by_speaker_id().items()
            },
            hypothesis={
                k: h.merged_transcripts()
                for k, h in hypothesis[example_id].grouped_by_speaker_id().items()
            },
        )
        for example_id in reference.keys()
    }, hypothesis_paths, 'cpwer', per_reco_out, average_out)


@command(
    'mimower',
    help='Computes the MIMO WER',
    parent_parsers=[common_wer_parser],
)
def mimower(reference, hypothesis, average_out, per_reco_out):
    reference, _, hypothesis, hypothesis_paths = _load_texts(reference, hypothesis)
    _save_results({
        example_id: mimo_word_error_rate(
            reference={
                k: r.utterance_transcripts()
                for k, r in reference[example_id].grouped_by_speaker_id().items()
            },
            hypothesis={
                k: h.merged_transcripts()
                for k, h in hypothesis[example_id].grouped_by_speaker_id().items()
            },
        )
        for example_id in reference.keys()
    }, hypothesis_paths, 'mimower', per_reco_out, average_out)


def _merge(
        files: List[str],
        out: str = None,
        average: bool = None
):
    # Load input files
    files = [Path(f) for f in files]
    data = [_load(f) for f in files]

    import meeteval
    ers = []

    for d in data:
        if 'errors' in d:  # Average file
            assert average is not False, average
            average = True  # A single average file forces to do an average
            ers.append([None, ErrorRate.from_dict(d)])
        else:
            for k, v in d.items():  # Details file
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


@command(
    'merge',
    help='Merges multiple (per-reco or averaged) files',
    parent_parsers=[common_merge_parser]
)
def merge(files, out):
    return _merge(files, out, average=None)


@command(
    'average',
    help='Computes the average over one or multiple per-reco files',
    parent_parsers=[common_merge_parser]
)
def average(files, out):
    return _merge(files, out, average=True)


def cli():
    # Parse arguments and find command to execute
    args = parser.parse_args()

    if hasattr(args, 'func'):
        kwargs = vars(args)
        fn = args.func
        # Pop also removes from args namespace
        kwargs.pop('func')
        kwargs.pop('version')
        fn(**kwargs)
        return

    if getattr(args, 'version', False):
        from meeteval import __version__
        print(__version__)
        return

    parser.print_help()


if __name__ == '__main__':
    cli()
