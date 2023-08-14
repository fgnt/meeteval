import argparse
import dataclasses
import glob
import json
import os
import re
from pathlib import Path
from typing import List, Tuple

from meeteval.io.ctm import CTMGroup
from meeteval.io.keyed_text import KeyedText
from meeteval.io.stm import STM
from meeteval.wer.wer import combine_error_rates, ErrorRate
import sys


def _dump(obj, path: 'Path | str', default_suffix='.json'):
    """
    Dumps the `obj` to `path`. Parses the suffix to find the file type.
    When a suffix is missing, use json.
    """
    path = Path(path)
    if path.stem == '-':
        from contextlib import nullcontext
        p = nullcontext(sys.stdout)
    else:
        try:
            p = path.open('w')
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f'Couldn\'t open the output file. Consider explicitly setting '
                f'the output files, especially when piping into this tool.'
            ) from e
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
            raise NotImplementedError(f'Unknown file ext: {suffix}')

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
            raise NotImplementedError(f'Unknown file ext: {path.suffix}')


def _load_reference(reference: 'Path | List[Path]'):
    """Loads a reference transcription file. Currently only STM supported"""
    return STM.load(reference)


def _load_hypothesis(hypothesis: List[Path]):
    """Loads the hypothesis. Supports one STM file or multiple CTM files
    (one per channel)"""
    if len(hypothesis) > 1:
        # We have multiple, only supported for ctm files
        suffix = {h.suffixes[-1] for h in hypothesis}
        assert len(suffix) == 1, suffix
        filename = suffix.pop()
    else:
        hypothesis = hypothesis[0]
        filename = str(hypothesis)

    if filename.endswith('.ctm'):
        if isinstance(hypothesis, list):
            return CTMGroup.load(hypothesis).to_stm()
        else:
            return CTMGroup.load([hypothesis]).to_stm()
    elif filename.endswith('.stm'):
        return STM.load(hypothesis)
    elif filename.startswith('/dev/fd/') or filename.startswith('/proc/self/fd/'):
        # This is a pipe, i.e. python -m ... <(cat ...)
        # For now, assume it is an STM file
        return STM.load(hypothesis)
    else:
        raise RuntimeError(hypothesis, filename)


def _load_texts(reference_paths: List[str], hypothesis_paths: List[str], regex) -> Tuple[STM, List[Path], STM, List[Path]]:
    """Load and validate reference and hypothesis texts.

    Validation checks that reference and hypothesis have the same example IDs.
    """

    # Normalize and glob (for backwards compatibility) the path input
    def _glob(pathname):
        match = list(glob.glob(pathname))
        # Forward pathname if not matched to get the correct error message
        return match or [pathname]

    reference_paths = [Path(file) for r in reference_paths for file in _glob(r)]
    hypothesis_paths = [Path(file) for h in hypothesis_paths for file in _glob(h)]

    # Load input files
    reference = _load_reference(reference_paths)

    # Hypothesis can be an STM file or a collection of  CTM files. Detect
    # which one we have and load it
    hypothesis = _load_hypothesis(hypothesis_paths)

    # Filter lines with regex based on filename
    if regex:
        r = re.compile(regex)

        def filter(stm):
            filenames = stm.filenames()
            filtered_filenames = [f for f in filenames if r.fullmatch(f)]
            assert filtered_filenames, (regex, filenames, 'Found nothing')
            return stm.filter(lambda l: l.filename in filtered_filenames)

        reference = filter(reference)
        hypothesis = filter(hypothesis)

    return reference, reference_paths, hypothesis, hypothesis_paths


def _get_parent_stem(hypothesis_paths: List[Path]):
    hypothesis_paths = [p.resolve() for p in hypothesis_paths]

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
        hypothesis_paths: List[Path],
        per_reco_out: str,
        average_out: str,
):
    """Saves the results.
    """
    parent, stem = _get_parent_stem(hypothesis_paths)

    # Save details
    _dump({
        example_id: dataclasses.asdict(error_rate)
        for example_id, error_rate in per_reco.items()
    }, per_reco_out.format(parent=f'{parent}/', stem=stem))

    # Compute and save average
    average = combine_error_rates(*per_reco.values())
    _dump(
        dataclasses.asdict(average),
        average_out.format(parent=f'{parent}/', stem=stem),
    )


def wer(
        reference, hypothesis,
        average_out='{parent}/{stem}_wer.json',
        per_reco_out='{parent}/{stem}_wer_per_reco.json',
):
    """Computes the "standard" WER (SISO WER). Only support kaldi-style text files"""
    reference_paths = [Path(r) for r in reference]
    hypothesis_paths = [Path(h) for h in hypothesis]
    if (
            any(r.suffix != '' for r in reference_paths) or
            any(h.suffix != '' for h in hypothesis_paths)
    ):
        raise ValueError(f'Only (kaldi-style) text files are supported, i.e., files without an extension '
                         f'(not dot allowed in the file name).\n'
                         f'Got: {reference_paths} for reference and {hypothesis_paths} for hypothesis.')
    reference = KeyedText.load(reference)
    hypothesis = KeyedText.load(hypothesis)
    from meeteval.wer.wer.siso import siso_word_error_rate_keyed_text
    results = siso_word_error_rate_keyed_text(reference, hypothesis)
    _save_results(results, hypothesis_paths, per_reco_out, average_out)


def orcwer(
        reference, hypothesis,
        average_out='{parent}/{stem}_orcwer.json',
        per_reco_out='{parent}/{stem}_orcwer_per_reco.json',
        regex=None,
):
    """Computes the Optimal Reference Combination Word Error Rate (ORC WER)"""
    from meeteval.wer.wer.orc import orc_word_error_rate_stm
    reference, _, hypothesis, hypothesis_paths = _load_texts(
        reference, hypothesis, regex=regex)
    results = orc_word_error_rate_stm(reference, hypothesis)
    _save_results(results, hypothesis_paths, per_reco_out, average_out)


def cpwer(
        reference, hypothesis,
        average_out='{parent}/{stem}_cpwer.json',
        per_reco_out='{parent}/{stem}_cpwer_per_reco.json',
        regex=None,
):
    """Computes the Concatenated minimum-Permutation Word Error Rate (cpWER)"""
    from meeteval.wer.wer.cp import cp_word_error_rate_stm
    reference, _, hypothesis, hypothesis_paths = _load_texts(
        reference, hypothesis, regex)
    results = cp_word_error_rate_stm(reference, hypothesis)
    _save_results(results, hypothesis_paths, per_reco_out, average_out)


def mimower(
        reference, hypothesis,
        average_out='{parent}/{stem}_mimower.json',
        per_reco_out='{parent}/{stem}_mimower_per_reco.json',
        regex=None,
):
    """Computes the MIMO WER"""
    from meeteval.wer.wer.mimo import mimo_word_error_rate_stm
    reference, _, hypothesis, hypothesis_paths = _load_texts(
        reference, hypothesis, regex=regex)
    results = mimo_word_error_rate_stm(reference, hypothesis)
    _save_results(results, hypothesis_paths, per_reco_out, average_out)


def tcpwer(
        reference, hypothesis,
        average_out='{parent}/{stem}_tcpwer.json',
        per_reco_out='{parent}/{stem}_tcpwer_per_reco.json',
        collar=0,
        hyp_pseudo_word_timing='character_based',
        ref_pseudo_word_timing='character_based',
        regex=None,
        verbose=False,
        allow_hypothesis_speaker_self_overlap=False,
        reference_overlap_correction=False,
):
    """Computes the time-constrained minimum permutation WER"""
    from meeteval.wer.wer.time_constrained import tcp_word_error_rate_stm
    reference, _, hypothesis, hypothesis_paths = _load_texts(
        reference, hypothesis, regex=regex)
    results = tcp_word_error_rate_stm(
        reference, hypothesis,
        reference_pseudo_word_level_timing=ref_pseudo_word_timing,
        hypothesis_pseudo_word_level_timing=hyp_pseudo_word_timing,
        collar=collar,
        allow_hypothesis_speaker_self_overlap=allow_hypothesis_speaker_self_overlap,
        reference_overlap_correction=reference_overlap_correction
    )
    _save_results(results, hypothesis_paths, per_reco_out, average_out)


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


def merge(files, out):
    """Merges multiple (per-reco or averaged) files"""
    return _merge(files, out, average=None)


def average(files, out):
    """Computes the average over one or multiple per-reco files"""
    return _merge(files, out, average=True)


def cli():
    # Define argument parser and commands
    parser = argparse.ArgumentParser()
    parser.add_argument('--version', action='store_true', help='Show version')
    commands = parser.add_subparsers(title='Subcommands')

    def positive_number(x: str):
        if x.isdigit():
            # Positive integer
            return int(x)

        x = float(x)
        if x < 0:
            raise ValueError(f'Number must be positive, but got {x}')

        return x

    def add_command(fn):
        command_parser = commands.add_parser(
            fn.__name__,
            add_help=False,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
            help=fn.__doc__,
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
            if name == 'reference':
                command_parser.add_argument(
                    '-r', '--reference',
                    help='Reference file(s) in STM or CTM format',
                    nargs='+', action='append',
                )
            elif name == 'hypothesis':
                command_parser.add_argument(
                    '-h', '--hypothesis',
                    help='Hypothesis file(s) in STM or CTM format',
                    nargs='+', action='append',
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
                    required=False, default='-',
                )
            elif name == 'collar':
                command_parser.add_argument(
                    '--collar', type=positive_number,
                    help='Collar applied to the hypothesis timings'
                )
            elif name == 'regex':
                command_parser.add_argument(
                    '--regex',
                    help='A regex pattern to select only particular filenames.'
                )
            elif name == 'hyp_pseudo_word_timing':
                command_parser.add_argument(
                    '--hyp-pseudo-word-timing', choices=[
                        'equidistant_intervals',
                        'equidistant_points',
                        'full_segment',
                        'character_based',
                        'none',
                    ],
                    help='Specifies how word-level timings are '
                         'determined from segment-level timing '
                         'for the hypothesis. Choices: '
                         'equidistant_intervals: Divide segment-level timing into equally sized intervals; '
                         'equidistant_points: Place time points equally spaded int the segment-level intervals; '
                         'full_segment: Use the full segment for each word that belongs to that segment;'
                         'character_based: Estimate the word length based on the number of characters; '
                         'none: Do not estimate word-level timings but assume that the provided timings are already '
                         'given on a word level.'
                )
            elif name == 'ref_pseudo_word_timing':
                command_parser.add_argument(
                    '--ref-pseudo-word-timing', choices=[
                        'equidistant_intervals',
                        'equidistant_points',
                        'full_segment',
                        'character_based',
                        'none',
                    ],
                    help='Specifies how word-level timings are '
                         'determined from segment-level timing '
                         'for the reference. Choices: '
                         'equidistant_intervals: Divide segment-level timing into equally sized intervals; '
                         'equidistant_points: Place time points equally spaded int the segment-level intervals; '
                         'full_segment: Use the full segment for each word that belongs to that segment. '
                         'character_based: Estimate the word length based on the number of characters; '
                         'none: Do not estimate word-level timings but assume that the provided timings are already '
                         'given on a word level.'
                )
            elif name == 'allow_hypothesis_speaker_self_overlap':
                command_parser.add_argument(
                    '--allow-hypothesis-speaker-self-overlap', action='store_true',
                    help='Allow speaker self-overlap in the hypothesis. '
                         'This can change the order of words, so it is not recommended to use this option. '
                         'You may set it when you are too lazy to fix your system or you want to get a preview '
                         'of the WER, but a valid recognition system should in general never produce self-overlap.'
                         'It is not guaranteed that the returned WER is correct if this option is set!'
                )
            elif name == 'reference_overlap_correction':
                command_parser.add_argument(
                    '--reference-overlap-correction', action='store_true',
                    help='Correct small overlaps in the reference by shifting the '
                         'start and end times of the overlapping segments to the center point of the overlap.'
                )
            elif name == 'files':
                command_parser.add_argument('files', nargs='+')
            elif name == 'verbose':
                command_parser.add_argument('--verbose', action='store_true')
            else:
                raise AssertionError("Error in command definition", name)

        # Get defaults from signature
        command_parser.set_defaults(
            func=fn,
            **{
                name: p.default
                for name, p in parameters.items()
                if p.default is not inspect.Parameter.empty
            }
        )

    add_command(wer)
    add_command(cpwer)
    add_command(orcwer)
    add_command(mimower)
    add_command(tcpwer)
    add_command(merge)
    add_command(average)

    # Parse arguments and find command to execute
    args = parser.parse_args()

    if hasattr(args, 'func'):
        kwargs = vars(args)
        fn = args.func
        # Pop also removes from args namespace
        kwargs.pop('func')
        kwargs.pop('version')
        if 'reference' in kwargs:
            kwargs['reference'] = [
                r for reference in kwargs['reference'] for r in reference
            ]
        if 'hypothesis' in kwargs:
            kwargs['hypothesis'] = [
                h for hypothesis in kwargs['hypothesis'] for h in hypothesis
            ]
        return fn(**kwargs)

    if getattr(args, 'version', False):
        from meeteval import __version__
        print(__version__)
        return

    parser.print_help()


if __name__ == '__main__':
    cli()
