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
from meeteval.wer.wer.time_constrained import TimeMarkedTranscript


def groupby(
        iterable,
        key = None,
):
    """
    A non-lazy variant of `itertools.groupby` with advanced features.

    Args:
        iterable: Iterable to group
        key: Determines by what to group. Can be:
            - `None`: Use the iterables elements as keys directly
            - `callable`: Gets called with every element and returns the group
                key
            - `str`, or `int`: Use `__getitem__` on elements in `iterable`
                to obtain the key
            - `Iterable`: Provides the keys. Has to have the same length as
                `iterable`.

    Examples:
        >>> groupby('ab'*3)
        {'a': ['a', 'a', 'a'], 'b': ['b', 'b', 'b']}
        >>> groupby(range(10), lambda x: x%2)
        {0: [0, 2, 4, 6, 8], 1: [1, 3, 5, 7, 9]}
        >>> groupby(({'a': x%2, 'b': x} for x in range(3)), 'a')
        {0: [{'a': 0, 'b': 0}, {'a': 0, 'b': 2}], 1: [{'a': 1, 'b': 1}]}
        >>> groupby(['abc', 'bd', 'abd', 'cdef', 'c'], 0)
        {'a': ['abc', 'abd'], 'b': ['bd'], 'c': ['cdef', 'c']}
        >>> groupby(range(10), list(range(5))*2)
        {0: [0, 5], 1: [1, 6], 2: [2, 7], 3: [3, 8], 4: [4, 9]}
        >>> groupby('abc', ['a'])
        Traceback (most recent call last):
            ...
        ValueError: zip() argument 2 is shorter than argument 1
        >>> groupby('abc', {})
        Traceback (most recent call last):
            ...
        TypeError: Invalid type for key: <class 'dict'>
    """
    if callable(key) or key is None:
        key_fn = key
    elif isinstance(key, (str, int)):
        key_fn = operator.itemgetter(key)
    elif not isinstance(key, Mapping):
        value_getter = operator.itemgetter(0)
        groups = collections.defaultdict(list)
        for key, group in itertools.groupby(zip(iterable, key, strict=True), operator.itemgetter(1)):
            groups[key].extend(map(value_getter, group))
        return dict(groups)
    else:
        raise TypeError(f'Invalid type for key: {type(key)}')

    groups = collections.defaultdict(list)
    for key, group in itertools.groupby(iterable, key_fn):
        groups[key].extend(group)
    return dict(groups)

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
        obj, path, *, indent=2, create_path=True, sort_keys=True, **kwargs):
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


def solve_assignment(ref: STM, hyp: STM, assignment_type, collar=5):
    if assignment_type == 'cp':
        from meeteval.wer.wer.cp import cp_word_error_rate
        wer = cp_word_error_rate(
            {k: [s['words'] for s in v] for k, v in groupby(sorted(ref, key=lambda x: x['start_time']), 'speaker').items()},
            {k: [s['words'] for s in v] for k, v in groupby(sorted(hyp, key=lambda x: x['start_time']), 'speaker').items()},
        )
    elif assignment_type in ('tcp', 'ditcp'):
        from meeteval.wer.wer.time_constrained import time_constrained_minimum_permutation_word_error_rate
        # The visualization looks wrong if we don't sort segments
        wer = time_constrained_minimum_permutation_word_error_rate(
            groupby(ref, 'speaker'),
            groupby(hyp, 'speaker'),
            collar=collar,
            reference_sort='segment',
            hypothesis_sort='segment',
            reference_pseudo_word_level_timing='character_based',
            hypothesis_pseudo_word_level_timing='character_based_points',
        )
    else:
        raise ValueError(assignment_type)

    _, hyp = wer.apply_assignment(
        groupby(ref, 'speaker'),
        groupby(hyp, 'speaker'),
    )
    hyp = [
        {**l, 'speaker': k}
        for k, v in hyp.items()
        for l in v
    ]

    return ref, hyp, wer


@dataclasses.dataclass(frozen=True)
class STMLine2Spk(STMLine):
    speaker2_id: str


def get_diarization_invariant_alignment(ref: List[Dict], hyp: List[Dict], collar=5):
    from meet_eval.dicpwer.dicp import greedy_di_tcp_error_rate
    words, _ = get_words_and_alignment(ref, hyp, 'tcp', collar=collar)

    wer = greedy_di_tcp_error_rate(
        list(groupby(ref, 'speaker').values()),
        [[[vv] for vv in v]for v in groupby(ref, 'speaker').values()],
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


def get_words_and_alignment(ref: List[Dict], hyp: List[Dict], alignment_type, collar=5):
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
    alignment = []

    ref = groupby(sorted(ref, key=lambda x: x['start_time']), 'speaker')
    hyp = groupby(sorted(hyp, key=lambda x: x['start_time']), 'speaker')

    for k in set(ref.keys()) | set(hyp.keys()):
        ref_ = ref.get(k, [])
        hyp_ = hyp.get(k, [])

        # Ignore timings here, we just need this for the speaker ID
        hyp_words = [
            {**l, 'words': w}
            for l in hyp_
            for w in l['words'].split()
        ]
        ref_word_timings = get_pseudo_word_level_timings(TimeMarkedTranscript.create(ref_), 'character_based')
        hyp_word_timings = get_pseudo_word_level_timings(TimeMarkedTranscript.create(hyp_), 'character_based')

        a = align(ref_word_timings, hyp_word_timings)
        for r, h in a:
            assert r is not None or h is not None
            if r is not None:
                match_type = 'deletion' if h is None else (
                    'correct' if ref_word_timings.transcript[r] == hyp_word_timings.transcript[h] else 'substitution'
                )
                words.append({
                    'transcript': ref_word_timings.transcript[r],
                    'begin_time': ref_word_timings.timings[r][0],
                    'end_time': ref_word_timings.timings[r][1],
                    'speaker_id': k,
                    'source': 'reference',
                    'match_type': match_type
                })
            if h is not None:
                match_type = 'insertion' if r is None else (
                    'correct' if ref_word_timings.transcript[r] == hyp_word_timings.transcript[h] else 'substitution'
                )
                words.append({
                    'transcript': hyp_word_timings.transcript[h],
                    'begin_time': hyp_word_timings.timings[h][0],
                    'end_time': hyp_word_timings.timings[h][1],
                    'speaker_id': k,
                    'source': 'hypothesis',
                    'match_type': match_type
                })

            if r is None:
                alignment.append({
                    'ref_speaker_id': k,
                    'hyp_speaker_id': hyp_words[h].get('speaker2', k),
                    'match_type': match_type,
                    'hyp_center_time': (hyp_word_timings.timings[h][-1] + hyp_word_timings.timings[h][0]) / 2
                })
            elif h is None:
                alignment.append({
                    'ref_speaker_id': k,
                    'hyp_speaker_id': k,
                    'match_type': match_type,
                    'ref_center_time': (ref_word_timings.timings[r][-1] + ref_word_timings.timings[r][0]) / 2
                })
            else:
                alignment.append({
                    'ref_speaker_id': k,
                    'hyp_speaker_id': hyp_words[h].get('speaker2', k),
                    'match_type': match_type,
                    'ref_center_time': (ref_word_timings.timings[r][-1] + ref_word_timings.timings[r][0]) / 2,
                    'hyp_center_time': (hyp_word_timings.timings[h][-1] + hyp_word_timings.timings[h][0]) / 2,
                })

            # TODO: Do we need this?
            # # Find corresponding utterance / segment
            # ref_utterance = None
            # hyp_segment = None
            # for u in utterances:
            #     if u['speaker_id'] != k:
            #         continue
            #     if h is not None:
            #         if u['source'] == 'hypothesis' and u['begin_time'] <= hyp_.timings[h][1] and u['end_time'] >= \
            #                 hyp_.timings[h][0]:
            #             hyp_segment = u
            #     if r is not None:
            #         if u['source'] == 'reference' and u['begin_time'] <= ref_.timings[r][1] and u['end_time'] >= \
            #                 ref_.timings[r][0]:
            #             ref_utterance = u
            # if match_type == 'deletion':
            #     ref_utterance['deletions'] += 1
            # elif match_type == 'insertion':
            #     hyp_segment['insertions'] += 1
            # elif match_type == 'substitution':
            #     ref_utterance['substitutions'] += 1
            #     hyp_segment['substitutions'] += 1
            # else:
            #     ref_utterance['correct'] += 1
            #     hyp_segment['correct'] += 1
    alignment.sort(key=lambda x: x['ref_center_time'] if 'ref_center_time' in x else x['hyp_center_time'])
    return words, alignment


def get_visualization_data(ref: List[Dict], hyp: List[Dict], assignment='tcp'):
    if isinstance(ref, STM):
        ref = ref.segments()
    if isinstance(hyp, STM):
        hyp = hyp.segments()
    data = {
        'info': {
            'filename': ref[0]['session_id'],
            'speakers': list(set(map(lambda x: x['speaker'], ref))),
            'alignment_type': assignment,
            'length': max([e['end_time'] for e in ref + hyp]) - min([e['start_time'] for e in ref + hyp])
        }
    }

    ref, hyp, wer = solve_assignment(ref, hyp, assignment)

    hyp_utterances = [
        {
            'transcript': l['words'],
            'begin_time': l['start_time'],
            'end_time': l['end_time'],
            'speaker_id': l['speaker'],
            'source': 'hypothesis',
            # 'correct': 0,
            # 'substitutions': 0,
            # 'insertions': 0,
            'total': len(l['words'].split()),
            **({'audio': l['audio']} if 'audio' in l else {})
        }
        for l in hyp
    ]
    ref_utterances = [
        {
            'transcript': l['words'],
            'begin_time': l['start_time'],
            'end_time': l['end_time'],
            'speaker_id': l['speaker'],
            'source': 'reference',
            # 'correct': 0,
            # 'substitutions': 0,
            # 'deletions': 0,
            'total': len(l['words'].split()),
            **({'audio': l['audio']} if 'audio' in l else {})
        }
        for l in ref
    ]
    data['utterances'] = hyp_utterances + ref_utterances

    # Word level
    data['words'], data['alignment'] = get_words_and_alignment(ref, hyp, assignment)

    data['info']['wer'] = dataclasses.asdict(wer)
    return data

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
        return get_visualization_data(self.ref, self.hyp, self.alignment)

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
                    console.log("Waiting for d3 to load");
                    // Wait for d3 to load
                    if (typeof d3 !== 'undefined') alignment_visualization(
                        {dumps_json(self.data, indent=None)}, 
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
                            show_legend: {'true' if self.show_legend else 'false'}
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
