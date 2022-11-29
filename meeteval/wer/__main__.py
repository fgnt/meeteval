import dataclasses
import json
import io
import os
import glob
from functools import wraps
from pathlib import Path
from typing import Tuple, List

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

import click


def _dump(obj, path: Path):
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
        if path.suffix == '.json' or path.suffix == '':
            json.dump(obj, fd, indent=2, sort_keys=False)
        elif path.suffix == '.yaml':
            import yaml
            yaml.dump(obj, fd, sort_keys=False)
        else:
            raise NotImplemented(f'Unknown file ext: {path.suffix}')

    if path.stem != '-':
        print(f'Wrote: {path}', file=sys.stderr)
    else:
        print()


def _load(path: Path):
    with path.open('r') as fd:
        if path.suffix == '.json':
            return json.load(fd)
        elif path.suffix == '.yaml':
            import yaml
            return yaml.load(fd)
        else:
            raise NotImplemented(f'Unknown file ext: {path.suffix}')


def _load_reference(reference: Path):
    """Loads a reference transcription file. Currently only STM supported"""
    return STM.load(reference)


def _load_hypothesis(hypothesis: Tuple[Path, ...]):
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

    # print(type(hypothesis), repr(hypothesis), hypothesis, filename)

    if filename.endswith('.ctm'):
        return CTMGroup.load([hypothesis])
    elif filename.endswith('.stm'):
        return STM.load(hypothesis)
    elif filename.startswith('/dev/fd/'):
        # This is a pipe, i.e. python -m ... <(cat ...)
        # For now, assume it is an STM file
        return STM.load(hypothesis)
    else:
        raise RuntimeError(hypothesis, filename)


def _load_texts(reference, hypothesis):
    """Load and validate reference and hypothesis texts.

    Validation checks that reference and hypothesis have the same example IDs.
    """
    # Load input files
    reference = _load_reference(reference)

    # Hypothesis can be an STM file or a collection of  CTM files. Detect
    # which one we have and load it
    hypothesis = _load_hypothesis(hypothesis)

    # Group by example IDs
    reference = reference.grouped_by_filename()
    hypothesis = hypothesis.grouped_by_filename()

    # Check that the input is valid
    if reference.keys() != hypothesis.keys():
        if len(reference.keys()) != len(hypothesis.keys()):
            raise RuntimeError(
                f'Length of reference ({len(reference)}) and hypothesis ({len(hypothesis)}) should be equal.')
        else:
            raise RuntimeError(
                'Keys of reference and hypothesis differ\n'
                f'hypothesis - reference: e.g. {list(set(hypothesis.keys()) - set(reference.keys()))[:5]}'
                f'reference - hypothesis: e.g. {list(set(reference.keys()) - set(hypothesis.keys()))[:5]}'
            )

    return reference, hypothesis


def _get_parent_stem(hypothesis_paths):

    hypothesis_paths = [
        Path(p.name).resolve()
        if isinstance(p, io.TextIOBase) else
        Path(p).resolve()
        for p in hypothesis_paths
    ]

    if len(hypothesis_paths) == 1:
        parent, stem = hypothesis_paths[0].parent, hypothesis_paths[0].stem
    else:
        parent = os.path.commonpath(hypothesis_paths)

        stems = [p.stem for p in hypothesis_paths]
        prefix = os.path.commonprefix(stems)
        postfix = os.path.commonprefix([s[::-1] for s in stems])[::-1]

        print('parent', parent)
        print('prefix', prefix)
        print('postfix', postfix)
        print('hypothesis_paths', hypothesis_paths)
        stem = f'{prefix}{postfix}'

    parent = str(parent).replace('*', '')
    stem = stem.replace('*', '')

    return parent, stem


def _save_results(
        per_reco,
        hypothesis_paths: Path,
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
    _average(per_reco.values(), average_output)


def _print_human_readable(d):
    from pprint import pprint
    for k, v in d.items():
        print(f'{k}: ', end='')
        pprint(v)


def _average(error_rates: List[ErrorRate], out: Path, print_human_readable=False):
    average = combine_error_rates(*error_rates)
    if print_human_readable:
        _print_human_readable(dataclasses.asdict(average))
    _dump(dataclasses.asdict(average), out)


@click.group(help='MeetEval meeting evaluation tool')
def cli():
    pass


average_out_option = click.option(
    '--average-out', default='{parent}/{stem}_{wer}.json',
    help='Output file for the average file. {stem} is replaced with the stem '
         'of the (first) hypothesis file and {wer} is replaced with the wer'
         'type (equal to the command name). '
         '"-" is interpreted as stdout. For example: "-.yaml" prints to '
         'stdout in yaml format.'
)


def wer_command(help=None):
    def _wer_command(f):
        @cli.command(help=help)
        @click.option(
            '--reference', '-r', required=True, multiple=True,
            # type=click.File('r'),
            type=click.Path(exists=False, dir_okay=False),
            help='The reference file. Currently only support STM file(s). '
                 'To use glob you have to escape the string, '
                 'e.g. -r "eval*.stm"'
        )
        @click.option(
            '--hypothesis', '-h', multiple=True, required=True,
            # type=click.File('r'),
            type=click.Path(exists=False, dir_okay=False),
            help='The hypothesis file(s). Support either STM file(s) or'
                 'multiple CTM files (one per hypothesis output)'
        )
        @click.option(
            '--per-reco-out', default='{parent}/{stem}_{wer}_per_reco.json',
            help='Output file for the per_reco file. {stem} is replaced with the stem '
                 'of the (first) hypothesis file and {wer} is replaced with the wer'
                 'type (equal to the command name). '

        )
        @average_out_option
        @wraps(f)
        def f_command(reference: 'list[str]', hypothesis: 'list[str]', per_reco_out, average_out, **kwargs):
            def _glob(pathname):
                match = list(glob.glob(pathname))
                assert match, (pathname, match)
                return match

            reference_path = reference
            hypothesis_path = hypothesis

            reference = [
                file for r in reference for file in _glob(r)]
            hypothesis = [
                file for h in hypothesis for file in _glob(h)]

            reference, hypothesis = _load_texts(reference, hypothesis)

            # Compute cpWER for all examples
            per_reco = {}
            for example_id in reference.keys():
                per_reco[example_id] = f(
                    reference=(reference[example_id]),
                    hypothesis=(hypothesis[example_id]),
                    **kwargs
                )
            _save_results(per_reco, hypothesis_path, f_command.name, per_reco_out, average_out)

        return f_command

    return _wer_command


@wer_command(
    help='Computes the Concatenated minimum-Permutation Word Error Rate '
         '(cpWER)'
)
def cpwer(reference, hypothesis):
    return cp_word_error_rate(
        reference={k: r_.merged_transcripts() for k, r_ in reference.grouped_by_speaker_id().items()},
        hypothesis={k: h_.merged_transcripts() for k, h_ in hypothesis.grouped_by_speaker_id().items()},
    )


@wer_command(
    help='Computes the Optimal Reference Combination Word Error Rate (ORC WER)'
)
def orcwer(reference, hypothesis):
    return orc_word_error_rate(
        reference=reference.utterance_transcripts(),
        hypothesis=[h_.merged_transcripts() for h_ in hypothesis.grouped_by_speaker_id()],
    )


@wer_command(
    help='Computes the MIMO WER'
)
def mimower(reference, hypothesis):
    return mimo_word_error_rate(
        reference=[r_.utterance_transcripts() for r_ in reference.grouped_by_speaker_id()],
        hypothesis=[h_.merged_transcripts() for h_ in hypothesis.grouped_by_speaker_id()],
    )


@cli.command(
    help='Computes the average WER over a per-reco file'
)
@click.argument(
    'per_reco_file', required=True,
    type=click.Path(exists=True, dir_okay=False, allow_dash=True, path_type=Path),
)
@click.argument(
    'out', required=False,
    type=click.Path(writable=True, allow_dash=True, path_type=Path),
)
@click.option(
    '--wer-type', type=click.Choice(['wer', 'cpwer']), default='wer',
    help='WER type (as this cannot easily be inferred from the file). '
         'Specifies which information is tracked. Choose "cpwer" for cpWER and'
         '"wer" for other WERs.'
)
def average_per_reco(per_reco_file: Path, out: Path = None, wer_type='wer'):
    if out is None:
        out = per_reco_file.with_stem(per_reco_file.stem.replace('per_reco', ''))
        assert out != per_reco_file

    if wer_type == 'wer':
        def error_rate(v):
            return ErrorRate(v['errors'], v['length'])
    elif wer_type == 'cpwer':
        def error_rate(v):
            return CPErrorRate(**v)
    else:
        # Can never happen (error should be caught by click)
        RuntimeError()

    with per_reco_file.open('r') as f:
        per_reco = {
            key: error_rate(v)
            for key, v in json.load(f).items()
        }
    _average(per_reco.values(), out)


@cli.command(
    help='Compute the average of multiple average files'
)
@click.argument(
    'files', required=True,
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False, allow_dash=True, path_type=Path),
)
@click.argument(
    'out', required=True,
    type=click.Path(writable=True, allow_dash=True, path_type=Path),
)
def average(files, out, wer_type='cpwer'):
    if wer_type == 'wer':
        def error_rate(v):
            return ErrorRate(v['errors'], v['length'])
    elif wer_type == 'cpwer':
        def error_rate(v):
            v.pop('error_rate', None)
            return CPErrorRate(**v)
    else:
        # Can never happen (error should be caught by click)
        raise RuntimeError()

    files = [error_rate(_load(f)) for f in files]
    _average(files, out)


@cli.command(
    help='Merges multiple per-reco files'
)
@click.argument('average_files', required=True, nargs=-1, type=click.Path(dir_okay=False, path_type=Path))
@click.argument('out', required=True, type=click.Path(dir_okay=False, path_type=Path, writable=True, allow_dash=True))
def merge_per_reco(average_files, out):
    files = [_load(f) for f in average_files]

    # Make sure that example IDs do not overlap
    seen_ids = set()
    for file in files:
        example_ids = set(file.keys())
        if seen_ids & example_ids:
            # Overlap detected
            raise RuntimeError(
                f'Found duplicate example IDs: '
                f'{seen_ids & example_ids}'
            )
    merged = {}
    for file in files:
        merged.update(file)
    _dump(merged, out)


if __name__ == '__main__':
    cli()
