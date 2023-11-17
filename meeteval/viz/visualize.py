import logging

logging.basicConfig(level=logging.ERROR)
import collections
import contextlib
import dataclasses
import functools
import itertools
import operator
import shutil
import uuid
from pathlib import Path
from typing import Dict, List, Mapping

from cached_property import cached_property

from meeteval.io.stm import STM, STMLine
from meeteval.wer.wer.time_constrained import TimeMarkedTranscript, get_pseudo_word_level_timings

from meeteval.io.tidy import convert_to_tidy, keys, groupby, Tidy, tidy_args


def dumps_json(
        obj, *, indent=2, sort_keys=True, **kwargs):
    import io
    fd = io.StringIO()
    dump_json(
        obj,
        path=fd,
        indent=indent,
        create_path=False,
        sort_keys=sort_keys,
        **kwargs,
    )
    return fd.getvalue()


def dump_json(
        obj, path, *, indent=2, create_path=True, sort_keys=False, **kwargs):
    """
    Numpy types will be converted to the equivalent Python type for dumping the
    object.

    :param obj: Arbitrary object that is JSON serializable,
        where Numpy is allowed.
    :param path: String or ``pathlib.Path`` object.
    :param indent: See ``json.dump()``.
    :param kwargs: See ``json.dump()``.

    """
    import io
    import json

    if isinstance(path, io.IOBase):
        json.dump(obj, path, indent=indent,
                  sort_keys=sort_keys, **kwargs)
    elif isinstance(path, (str, Path)):
        path = Path(path).expanduser()

        if create_path:
            path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('w') as f:
            json.dump(obj, f, indent=indent,
                      sort_keys=sort_keys, **kwargs)
    else:
        raise TypeError(path)


html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <script src="https://d3js.org/d3.v7.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
</head>
<body>
<div>{settings}</div>
<div id="my_dataviz"></div>
<script src="visualize.js"></script>
<script>
    d3.json("data.json").then(d => alignment_visualization(d, "#my_dataviz"));
</script>
</body>
</html>
"""


def get_wer(ref: Tidy, hyp: Tidy, assignment_type, collar=5):
    if assignment_type == 'cp':
        from meeteval.wer.wer.cp import cp_word_error_rate
        wer = cp_word_error_rate(ref, hyp)
    elif assignment_type in ('tcp', 'ditcp'):
        from meeteval.wer.wer.time_constrained import time_constrained_minimum_permutation_word_error_rate
        # The visualization looks wrong if we don't sort segments
        wer = time_constrained_minimum_permutation_word_error_rate(
            ref, hyp,
            collar=collar,
            reference_sort='segment',
            hypothesis_sort='segment',
            reference_pseudo_word_level_timing='character_based',
            hypothesis_pseudo_word_level_timing='character_based_points',
        )
    else:
        raise ValueError(assignment_type)
    return wer


def apply_assignment(assignment, d: Tidy):
    # Both ref and hyp key can be missing or None
    # This can happen when the filter function excludes a speaker completely
    # TODO: Find a good way to name these and adjust apply_cp_assignment accordingly
    assignment = dict(
        ((b, a if a is not None else f'[{b}]')
        for a, b in assignment)
    )
    d = [
        {**w, keys.SPEAKER: assignment.get(w[keys.SPEAKER], f'[{w[keys.SPEAKER]}]')}
        for w in d
    ]
    return d


def get_diarization_invariant_alignment(ref: Tidy, hyp: Tidy, collar=5):
    from meet_eval.dicpwer.dicp import greedy_di_tcp_error_rate
    words, _ = get_words_and_alignment(ref, hyp, 'tcp', collar=collar)

    wer = greedy_di_tcp_error_rate(
        list(groupby(ref, 'speaker').values()),
        [[[vv] for vv in v] for v in groupby(ref, 'speaker').values()],
        collar=collar
    )

    hyp = wer.apply_assignment(sorted(hyp, key=lambda x: x['start_time']))
    hyp = [
        {**l, 'speaker2': k, }
        for k, v in hyp.items()
        for l in v
    ]

    _, alignment = get_words_and_alignment(ref, hyp, 'tcp', collar=collar)
    return words, alignment


def get_words_and_alignment(ref: Tidy, hyp: Tidy, alignment_type, collar=5, ignore='<CC>'):
    if alignment_type == 'cp':
        from meeteval.wer.wer.time_constrained import align
        # Set the collar large enough that all words overlap with all other words
        min_time = min(map(lambda x: x['start_time'], ref + hyp))
        max_time = max(map(lambda x: x['end_time'], ref + hyp))
        align = functools.partial(
            align, collar=max_time - min_time + 1, style='index',
            reference_pseudo_word_level_timing='none',
            hypothesis_pseudo_word_level_timing='none',
            # Disable sort: We pass words that are already ordered correctly
            reference_sort=False,
            hypothesis_sort=False,
        )
    elif alignment_type == 'tcp':
        from meeteval.wer.wer.time_constrained import align
        align = functools.partial(
            align, collar=collar, style='index',
            reference_pseudo_word_level_timing='none',
            hypothesis_pseudo_word_level_timing='none',
            # Disable sort: We pass words that are already ordered correctly
            reference_sort=False,
            hypothesis_sort=False,
        )
    elif alignment_type == 'ditcp':
        return get_diarization_invariant_alignment(ref, hyp, collar=collar)
    else:
        raise ValueError(alignment_type)

    # Get word-level timings
    from meeteval.wer.wer.time_constrained import get_pseudo_word_level_timings

    # Compute alignment
    words = []

    ref = groupby(sorted(ref, key=lambda x: x[keys.START_TIME]), keys.SPEAKER)
    hyp = groupby(sorted(hyp, key=lambda x: x[keys.START_TIME]), keys.SPEAKER)

    for k in set(ref.keys()) | set(hyp.keys()):
        ref_word = ref.get(k, [])
        hyp_word = hyp.get(k, [])

        # Estimate words from segments
        # ref_word = get_pseudo_word_level_timings(ref_, 'character_based')
        # hyp_word = get_pseudo_word_level_timings(hyp_, 'character_based')

        # Filter out empty words. Their boundaries will be drawn based on the utterance/segment boundaries
        ref_word = [w for w in ref_word if w[keys.WORDS] != '']
        hyp_word = [w for w in hyp_word if w[keys.WORDS] != '']

        # Ignore `ignore` tokens while aligning
        # ignored_words = [w for w in ref_word + hyp_word if w[keys.WORDS] == ignore]
        # words.extend([{**w, 'match_type': 'ignored'} for w in ignored_words])
        # ref_word = [w for w in ref_word if w[keys.WORDS] != ignore]
        # hyp_word = [w for w in hyp_word if w[keys.WORDS] != ignore]

        a = align(ref_word, hyp_word, reference_pseudo_word_level_timing='none',
                  hypothesis_pseudo_word_level_timing='none', style='tidy', collar=collar)

        for r, h in a:
            assert r is not None or h is not None
            if r is not None:
                r['word_index'] = len(words)
                if r[keys.WORDS] != '':
                    match_type = 'deletion' if h is None or h[keys.WORDS] == '' else (
                        'correct' if r[keys.WORDS] == h[keys.WORDS] else 'substitution'
                    )
                    r['match_type'] = match_type
                    if h is not None:
                        h['match_index'] = r['word_index']
                words.append(r)
            if h is not None:
                h['word_index'] = len(words)
                if h[keys.WORDS] != '':
                    match_type = 'insertion' if r is None or r[keys.WORDS] == '' else (
                        'correct' if r[keys.WORDS] == h[keys.WORDS] else 'substitution'
                    )
                    h['match_type'] = match_type
                    if r is not None:
                        r['match_index'] = h['word_index']
                words.append(h)
            if r is not None and h is not None:
                assert r[keys.START_TIME] <= h[keys.END_TIME] + collar, (r, h)
                assert h[keys.START_TIME] <= r[keys.END_TIME] + collar, (r, h)

    words = [{**w, 'center_time': (w[keys.START_TIME] + w[keys.END_TIME]) / 2} for w in words]
    return words


@tidy_args(2)
def get_visualization_data(ref: Tidy, hyp: Tidy, assignment='tcp', alignment_transform=None):
    if alignment_transform is None:
        alignment_transform = lambda x: x

    data = {
        'info': {
            'filename': ref[0]['session_id'],
            'speakers': list(set(map(lambda x: x['speaker'], ref))),
            'alignment_type': assignment,
            'length': max([e['end_time'] for e in ref + hyp]) - min([e['start_time'] for e in ref + hyp])
        }
    }

    # Add information about ref/hyp to each utterance
    ref = [{**s, 'source': 'reference'} for s in ref]
    hyp = [{**s, 'source': 'hypothesis'} for s in hyp]

    # Sort by begin time. Otherwise, the alignment will be unintuitive and likely not what the user wanted
    ref = sorted(ref, key=lambda x: x[keys.START_TIME])
    hyp = sorted(hyp, key=lambda x: x[keys.START_TIME])

    # Convert to words so that the transformation can be applied
    ref_word = get_pseudo_word_level_timings(ref, 'character_based')
    hyp_word = get_pseudo_word_level_timings(hyp, 'character_based')
    ref_word = [{**w, keys.WORDS: call_with_args(alignment_transform, w), 'original_words': w[keys.WORDS]} for w in
                ref_word]
    hyp_word = [{**w, keys.WORDS: call_with_args(alignment_transform, w), 'original_words': w[keys.WORDS]} for w in
                hyp_word]

    # Remove any words that are now empty
    ref_ignored_words = [w for w in ref_word if w[keys.WORDS] == '' or w[keys.WORDS] is None]
    hyp_ignored_words = [w for w in hyp_word if w[keys.WORDS] == '' or w[keys.WORDS] is None]
    ref_word = [w for w in ref_word if w[keys.WORDS] is not None and w[keys.WORDS] != '']
    hyp_word = [w for w in hyp_word if w[keys.WORDS] is not None and w[keys.WORDS] != '']

    # Get assignment using the word-level timestamps and filtered data
    wer = get_wer(ref_word, hyp_word, assignment, collar=5)
    hyp = apply_assignment(wer.assignment, hyp)
    hyp_word = apply_assignment(wer.assignment, hyp_word)
    hyp_ignored_words = apply_assignment(wer.assignment, hyp_ignored_words)

    # Get the alignment using the filtered data. Add ignored words for visualization
    words = get_words_and_alignment(ref_word, hyp_word, assignment)
    ignored_words = [{**w, 'match_type': 'ignored'} for w in hyp_ignored_words + ref_ignored_words]
    words = words + ignored_words

    # Map back to original_words TODO: remove "original_words" key
    words = [{**w, keys.WORDS: w['original_words'], 'transformed_words': w[keys.WORDS]} for w in words]
    data['words'] = words

    # Add utterances to data. Add total number of words to each utterance
    data['utterances'] = [{**l, 'total': len(l[keys.WORDS].split())} for l in ref + hyp]

    data['info']['wer'] = dataclasses.asdict(wer)
    return data


def call_with_args(fn, d):
    import inspect
    sig = inspect.signature(fn)
    parameters = sig.parameters
    if any([p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()]):
        kwargs = d
    else:
        kwargs = {
            name: d[name]
            for name, p in list(parameters.items())[1:]
            if p.kind in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY]
        }
    return fn(d[keys.WORDS], **kwargs)


class AlignmentVisualization:
    available_colormaps = ['default', 'diff', 'seaborn_muted']

    def __init__(
            self,
            ref: STM,
            hyp: STM,
            alignment='tcp',
            colormap='default',
            barplot_style='absolute',
            barplot_scale_exclude_total=False,
            num_minimaps=2,
            show_details=True,
            show_legend=True,
            highlight_regex=None,
            alignment_transform=None,
    ):
        self.ref = ref
        self.hyp = hyp
        self.alignment = alignment
        self.colormap = colormap
        self.barplot_style = barplot_style
        self.barplot_scale_exclude_total = barplot_scale_exclude_total
        self.num_minimaps = num_minimaps
        self.show_details = show_details
        self.show_legend = show_legend
        self.highlight_regex = highlight_regex
        self.alignment_transform = alignment_transform

    def _get_colormap(self):
        if isinstance(self.colormap, str):
            if not self.colormap in self.available_colormaps:
                raise ValueError(
                    f'Unknown colormap: {self.colormap}. Use one of {self.available_colormaps} or a custom mapping from '
                    f'match types ("correct", "insertion", "deletion" and "substitution") to (HTML) colors.'
                )
            return f'colormaps.{self.colormap}'
        else:
            colormap_keys = ['correct', 'substitution', 'insertion', 'deletion']
            if self.colormap.keys() != colormap_keys:
                raise ValueError(
                    f'Colormap defined the wrong keys: Need {colormap_keys} but found {list(self.colormap.keys())}'
                )
            return dumps_json(self.colormap, indent=None)

    @cached_property
    def data(self):
        return get_visualization_data(self.ref, self.hyp, self.alignment, alignment_transform=self.alignment_transform)

    def _repr_html_(self):
        return self.html()

    def html(self):
        """
        TODO: make this work without the manual patching ("replace") below

        Be aware that this writes _a lot of data_ in json format into the output cell. This can cause the browser to
        hang/crash and may produce large ipynb files.
        """
        # Generate data
        element_id = 'viz-' + str(uuid.uuid4())

        # Generate HTML and JS for data
        visualize_js = (Path(__file__).parent / 'visualize.js').read_text()
        css = (Path(__file__).parent / 'visualize.css').read_text()
        highlight_regex = f'"{self.highlight_regex}"' if self.highlight_regex else 'null'
        html = f'''
            <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/howler/2.2.4/howler.min.js"></script>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css" integrity="sha512-z3gLpd7yknf1YoNbCzqRKc4qyor8gaKU1qmn+CShxbuBusANI9QpRohGBreCFkKxLhei6S9CQXFEbbKuqLg0DA==" crossorigin="anonymous" referrerpolicy="no-referrer" />
            <style>
                {css}
            </style>
            <div style="margin: auto" class="meeteval-viz">
                <div id='{element_id}'></div>
            <div>
            <script>
                {visualize_js}

                function exec() {{
                    // Wait for d3 to load
                    if (typeof d3 !== 'undefined') alignment_visualization(
                        {dumps_json(self.data, indent=None, sort_keys=False)}, 
                        "#{element_id}",
                        {{
                            colors: {self._get_colormap()},
                            barplot: {{
                                style: "{self.barplot_style}",
                                scaleExcludeCorrect: {'true' if self.barplot_scale_exclude_total else 'false'}
                            }},
                            minimaps: {{
                                number: {self.num_minimaps}
                            }},
                            show_details: {'true' if self.show_details else 'false'},
                            show_legend: {'true' if self.show_legend else 'false'},
                            search_bar: {{
                                initial_query: {highlight_regex}
                            }}
                        }}
                    );
                    else setTimeout(exec, 100);
                }}
                exec();
                
            </script>
        '''
        return html

    def dump(self, filename):
        Path(filename).write_text(self.html())


def cli():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        '--help', help='show this help message and exit',
        action='help',
        default=argparse.SUPPRESS,
    )

    parser.add_argument('--reference', '-r', type=str, required=True)
    parser.add_argument('--hypothesis', '-h', type=str, required=True)
    parser.add_argument('--filename', type=str, required=True)  # TODO: name!
    parser.add_argument('--alignment', '-a', type=str, default='tcp', choices=['tcp', 'cp', 'ditcp'])
    parser.add_argument('--output-dir', '-o', type=str, default='.')
    parser.add_argument('--html-title', type=str, default='MeetEval Visualization {filename}')
    parser.add_argument('--no-browser', action='store_true')

    args = parser.parse_args()

    data = get_visualization_data(
        STM.load(args.reference).grouped_by_filename()[args.filename],
        STM.load(args.hypothesis).grouped_by_filename()[args.filename],
        assignment=args.alignment,
    )

    output_dir = Path(args.output_dir)
    dump_json(data, output_dir / 'data.json')
    Path(output_dir / 'index.html').write_text(
        html_template.format(title=args.html_title.format(filename=args.filename), settings=str(args)))
    shutil.copy(Path(__file__).parent / 'plot.js', output_dir / 'plot.js')
    shutil.copy(Path(__file__).parent / 'visualize.js', output_dir / 'visualize.js')

    import os
    os.system(f'cd {output_dir} && python -m http.server')
    #
    # if not args.no_browser:
    #     try:
    #         import webbrowser
    #     except:
    #         logging.warning('Could not open browser automatically. Please install the "webbrowser" Python module or open the file manually.')
    #     else:
    #         webbrowser.open(str(output_dir / 'index.html'))


if __name__ == '__main__':
    cli()
