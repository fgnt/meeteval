import os
import pprint

from meeteval.wer.wer.utils import check_single_filename
import urllib.request
import meeteval
from meeteval.wer import ErrorRate

import dataclasses
import functools
import uuid
from pathlib import Path

try:
    from functools import cached_property
except ImportError:
    # Fallback for Python 3.7 and lower, since cached_property was added in
    # Python 3.8.
    from cached_property import cached_property  # Python 3.7

from meeteval.io.stm import STM
from meeteval.wer.wer.time_constrained import get_pseudo_word_level_timings

from meeteval.io.seglst import asseglst, SegLST


def dumps_json(
        obj, *, indent=2, sort_keys=True,
        float_round=None,
        **kwargs,
):
    import io
    fd = io.StringIO()
    dump_json(
        obj,
        path=fd,
        indent=indent,
        create_path=False,
        sort_keys=sort_keys,
        float_round=float_round,
        **kwargs,
    )
    return fd.getvalue()


def dump_json(
        obj, path, *, indent=2, create_path=True, sort_keys=False,
        float_round=None,
        **kwargs,
):
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
    import simplejson

    if float_round is not None:
        import decimal

        def nested_round(obj):
            if isinstance(obj, (tuple, list)):
                return [nested_round(e) for e in obj]
            elif isinstance(obj, dict):
                return {nested_round(k): nested_round(v) for k, v in
                        obj.items()}
            elif isinstance(obj, (float, decimal.Decimal)):
                return round(obj, float_round)
            elif type(obj) in [int, str, type(None)]:
                return obj
            else:
                raise TypeError(type(obj))
        obj = nested_round(obj)

    if isinstance(path, io.IOBase):
        simplejson.dump(obj, path, indent=indent,
                        sort_keys=sort_keys, **kwargs)
    elif isinstance(path, (str, Path)):
        path = Path(path).expanduser()

        if create_path:
            path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('w') as f:
            simplejson.dump(obj, f, indent=indent,
                            sort_keys=sort_keys,
                            for_json=True,
                            **kwargs)
    else:
        raise TypeError(path)



def get_alignment(data, alignment_type, collar=5):
    # Extract hyps and ref from data. They have been merged earlier for easier processing
    hyp = data.filter(lambda s: s['source'] == 'hypothesis')
    ref = data.filter(lambda s: s['source'] == 'reference')

    if alignment_type == 'levenshtein':
        from meeteval.wer.wer.time_constrained import align
        # Set the collar large enough that all words overlap with all other words
        min_time = min(map(lambda x: x['start_time'], data))
        max_time = max(map(lambda x: x['end_time'], data))
        align = functools.partial(
            align,
            collar=max_time - min_time + 1,
            reference_pseudo_word_level_timing='none',
            hypothesis_pseudo_word_level_timing='none',
            # Disable sort: We pass words that are already ordered correctly
            reference_sort=False,
            hypothesis_sort=False,
            style='seglst',
        )
    elif alignment_type == 'time_constrained':
        from meeteval.wer.wer.time_constrained import align
        align = functools.partial(
            align,
            collar=collar,
            reference_pseudo_word_level_timing='none',
            hypothesis_pseudo_word_level_timing='none',
            # Disable sort: We pass words that are already ordered correctly
            reference_sort=False,
            hypothesis_sort=False,
            style='seglst',
        )
    else:
        raise NotImplementedError(alignment_type)

    # Compute alignment and extract words
    ref = ref.groupby('speaker')
    hyp = hyp.groupby('speaker')

    for k in set(ref.keys()) | set(hyp.keys()):
        a = align(
            ref.get(k, SegLST([])),
            hyp.get(k, SegLST([])).map(
                lambda s: {
                    **s,
                    'start_time': (s['end_time'] + s['start_time']) / 2,
                    'end_time': (s['end_time'] + s['start_time']) / 2,
                }
            )
        )

        # Add a list of matches to each word. This assumes that `align` keeps the
        # identity of the segments
        for r, h in a:
            assert r is not None or h is not None

            # Get the original words so that inplace updates have an effect
            r = data[r['word_index']] if r else None
            h = data[h['word_index']] if h else None

            # Find match type
            # Add matching: The reference can match with multiple hypothesis streams,
            # so r has a list of indices while h has a list of only a single index
            if r is None:  # Insertion
                h['matches'] = [(None, 'insertion')]
            elif h is None:  # Deletion
                r.setdefault('matches', []).append((None, 'deletion'))
            elif r['words'] == h['words']:  # Correct
                h.setdefault('matches', []).append((r['word_index'], 'correct'))
                r.setdefault('matches', []).append((h['word_index'], 'correct'))
            else:  # Substitution
                h.setdefault('matches', []).append((r['word_index'], 'substitution'))
                r.setdefault('matches', []).append((h['word_index'], 'substitution'))


def get_error_rate(ref, hyp, assignment):
    """
    Computes the word error rate and applies the assignment to the reference and hypothesis.
    """
    if assignment == 'cp':
        wer = meeteval.wer.wer.cp.cp_word_error_rate(ref, hyp)
    elif assignment == 'tcp':
        wer = meeteval.wer.wer.time_constrained.time_constrained_minimum_permutation_word_error_rate(
            ref, hyp,
            collar=5,
            reference_sort='segment',
            hypothesis_sort='segment',
            reference_pseudo_word_level_timing='character_based',
            hypothesis_pseudo_word_level_timing='character_based_points',
        )
    elif assignment == 'tcorc':
        wer = meeteval.wer.wer.time_constrained_orc_wer(
            ref, hyp,
            collar=5,
            reference_sort='segment',
            hypothesis_sort='segment',
            reference_pseudo_word_level_timing='character_based',
            hypothesis_pseudo_word_level_timing='character_based_points',
        )
    elif assignment == 'greedy_tcorc':
        wer = meeteval.wer.wer.greedy_time_constrained_orc_wer(
            ref, hyp,
            collar=5,
            reference_sort='segment',
            hypothesis_sort='segment',
            reference_pseudo_word_level_timing='character_based',
            hypothesis_pseudo_word_level_timing='character_based_points',
        )
    elif assignment == 'greedy_ditcp':
        wer = meeteval.wer.wer.greedy_di_tcp_word_error_rate(
            ref, hyp,
            collar=5,
            reference_sort='segment',
            hypothesis_sort='segment',
            reference_pseudo_word_level_timing='character_based',
            hypothesis_pseudo_word_level_timing='character_based_points',
        )
    elif assignment == 'orc':
        wer = meeteval.wer.wer.orc.orc_word_error_rate(ref, hyp)
    elif assignment == 'greedy_orc':
        wer = meeteval.wer.wer.greedy_orc_word_error_rate(ref, hyp)
    elif assignment == 'greedy_dicp':
        wer = meeteval.wer.wer.greedy_di_cp_word_error_rate(ref, hyp)
    else:
        raise ValueError(assignment)
    
    return wer


def add_overlap_shift(utterances: SegLST):
    """
    Adds the keys "overlap_shift" and "overlap_width" to each utterance. These
    values are used to determine the width and horizontal position of each
    utterance in the visualization such that they do not overlap visually, even if they
    overlap temporally.
    """
    utterances = utterances.sorted('start_time')

    # Compute the latest seen end time for each utterance up to that point.
    # This makes the search for overlapping utterances faster.
    latest_seen_end_times = []
    for u in utterances:
        latest_seen_end_times.append(max(u['end_time'], latest_seen_end_times[-1] if latest_seen_end_times else 0))

    for utterance in utterances:

        # Find any other overlapping utterances
        overlaps = []
        for other_utterance, end_time in zip(utterances[:utterance['utterance_index']][::-1], latest_seen_end_times[::-1]):
            if other_utterance['end_time'] > utterance['start_time']:
                if other_utterance['source'] == utterance['source'] and other_utterance['speaker'] == utterance['speaker']:
                    overlaps.append(other_utterance['utterance_index'])
                    other_utterance['utterance_overlaps'].append(utterance['utterance_index'])
            elif end_time < utterance['start_time']:
                break
        
        # Compute shifts from the overlaps such that the utterances don't overlap
        # This is a greedy approach that works well for most cases
        utterance['utterance_overlaps'] = overlaps
        if overlaps:
            shifts = [
                utterances[o]['overlap_shift']
                for o in overlaps
            ]
            for shift in range(len(shifts) + 1):
                if shift not in shifts:
                    break
            utterance['overlap_shift'] = shift
        else:
            utterance['overlap_shift'] = 0

    # Compute the width for each (sub)column and assign it to the utterance
    # This should result in the largest possible width for each utterance
    # such that no two utterances overlap
    for utterance in utterances:
        utterance['num_columns'] = max([utterances[o]['overlap_shift'] for o in utterance['utterance_overlaps']] + [utterance['overlap_shift']]) + 1 

    for utterance in  utterances.sorted(lambda x: -x['num_columns']):
        num_columns =  max([utterances[o]['num_columns'] for o in utterance['utterance_overlaps']] + [utterance['num_columns']])

        width = 1 / num_columns
        utterance['overlap_width'] = width



def get_visualization_data(
        ref: SegLST, 
        hyp: SegLST, 
        assignment='tcp', 
        alignment_transform=None,
        precomputed_error_rate=None,
        system_name=None,
    ):
    """
    Generates the data structure as required by the visualization frontend.

    Solves the stream assignment problem and computes the alignment between the reference and hypothesis.
    Then, computes additional useful information for display in the visualization.

    Args:
        precomputed_error_rate: A precomputed `ErrorRate` object with a `apply_assignment` method. If given, 
            the alignment will be applied to the reference and hypothesis.
            Note that this assigment should match the `assignment` parameter. If not, the visualization will be incorrect.
    """
    ref = asseglst(ref)
    hyp = asseglst(hyp)
    check_single_filename(ref, hyp)

    data = {
        'info': {
            'session_id': ref[0]['session_id'],
            'system_name': system_name,
            'alignment_type': assignment,
            'end_time': max([e['end_time'] for e in hyp + ref]),
            'sart_time': min([e['start_time'] for e in hyp + ref]),
            'length': max([e['end_time'] for e in hyp + ref]) - min([e['start_time'] for e in hyp + ref]),
        }
    }

    # Add original speaker/stream label
    ref = ref.map(lambda s: {**s, 'stream': s['speaker']})
    hyp = hyp.map(lambda s: {**s, 'stream': s['speaker']})

    # Get and apply stream assignment
    if precomputed_error_rate is not None:
        wer = precomputed_error_rate
    else:
        wer = get_error_rate(ref, hyp, assignment)
    ref, hyp = wer.apply_assignment(ref, hyp)
    align_type = 'time_constrained' if assignment in ['tcp', 'tcorc'] else 'levenshtein'

    if alignment_transform is None:
        alignment_transform = lambda x: x

    # Add information about ref/hyp to each utterance
    ref = ref.map(lambda s: {**s, 'source': 'reference'})
    hyp = hyp.map(lambda s: {**s, 'source': 'hypothesis'})

    u = ref + hyp

    # Sort by begin time. Otherwise, the alignment will be unintuitive and likely not what the user wanted
    u = u.sorted('start_time')

    # Add utterance index
    for i, utterance in enumerate(u):
        utterance['utterance_index'] = i

    # Convert to words so that the transformation can be applied
    w = get_pseudo_word_level_timings(u, 'character_based')
    w = w.map(lambda w: {**w, 'words': call_with_args(alignment_transform, w), 'original_words': w['words']})

    # Remove any words that are now empty
    ignored_words = w.filter(lambda s: not s['words'])  # .map(lambda s: {**s, 'match_type': 'ignored'})
    w = w.filter(lambda s: s['words'])

    # Get the alignment using the filtered data. Add ignored words for visualization
    # Add running word index used by the alignment to refer to different words
    for i, word in enumerate(w):
        word['word_index'] = i
    get_alignment(w, align_type, collar=5)
    words = w + ignored_words

    # Map back to original_words
    # and pre-compute things that are required in the visualization
    words = words.map(lambda w: {
        **w,
        'words': w['original_words'],
        'transformed_words': w['words'],
        'duration': w['end_time'] - w['start_time'],
    })

     # Add info about the number of errors to each uttearnce
    for word in words:
        utterance = u[word['utterance_index']]
        utterance['errors'] = utterance.get('errors', 0) + (0 if 'matches' in word and word['matches'] and word['matches'][0][1] == 'correct' else 1)

    compress = True
    if compress:
        data['words'] = {k: words.T.get(k) for k in words.T.keys(all=True)}
        data['words'] = {
            k: data['words'][k]
            for k in [
                'words',
                'source',
                'matches',
                'speaker',
                'start_time',
                'duration',
                'utterance_index',
            ]
        }
        def compress(m):
            if not m:
                return m
            if isinstance(m, (tuple, list)):
                return [compress(e) for e in m]
            if isinstance(m, str):
                return {
                    'insertion': 'i',
                    'deletion': 'd',
                    'substitution': 's',
                    'correct': 'c',
                }[m]
            return m
        data['words']['matches'] = [compress(m) for m in data['words']['matches']]
        data['words']['source'] = [{'hypothesis': 'h', 'reference': 'r'}[s] for s in data['words']['source']]
    else:
        data['words'] = words.segments

    add_overlap_shift(u)

    # Add utterances to data. Add total number of words to each utterance
    data['utterances'] = [{**l, 'total': len(l['words'].split())} for l in u]

    data['info']['wer'] = dataclasses.asdict(wer)

    def wer_by_speaker(speaker):
        # Get all words from this speaker
        words_ = words.filter(lambda s: s['speaker'] == speaker)

        # Get all hypothesis words. From this we can find the number of insertions, substitutions and correct matches.
        # Ignore any words that are not matched (i.e., don't have a "matches" key)
        hyp_words = words_.filter(lambda s: s['source'] == 'hypothesis' and 'matches' in s)
        insertions = len(hyp_words.filter(lambda s: s['matches'][0][1] == 'insertion'))
        substitutions = len(hyp_words.filter(lambda s: s['matches'][0][1] == 'substitution'))
        # correct = len(hyp_words.filter(lambda s: s['matches'][0][1] == 'correct'))

        # Get all reference words. From this we can find the number of
        # deletions, substitutions and correct matches.
        # The number of deletions is the number of reference words that are not matched with a hypothesis word.
        ref_words = words_.filter(lambda s: s['source'] == 'reference' and 'matches' in s)
        deletions = len(ref_words.filter(
            lambda s: not [w for w, _ in s['matches'] if w is not None and words[w]['source'] == 'hypothesis']))

        return dataclasses.asdict(ErrorRate(
            errors=insertions + deletions + substitutions,
            length=len(ref_words),
            insertions=insertions,
            deletions=deletions,
            substitutions=substitutions,
            reference_self_overlap=None,
            hypothesis_self_overlap=None,
        ))

    data['info']['wer_by_speakers'] = {
        speaker: wer_by_speaker(speaker)
        for speaker in list((ref + hyp).unique('speaker'))
    }
    for k in [
        'errors', 'length', 'insertions', 'deletions', 'substitutions'
    ]:
        stats_of_by_speaker = sum(
            [wer_of_speaker[k] for _, wer_of_speaker in data['info']['wer_by_speakers'].items()],
            0
        )
        if stats_of_by_speaker != data['info']['wer'][k]:
            def indent(s, prefix):
                indent = ' ' * len(prefix)
                return prefix + s.replace('\n', '\n' + indent)
            wer_details = indent(pprint.pformat(data['info']['wer']), '  WER details: ')
            alignments_details = indent(pprint.pformat(data['info']['wer_by_speakers']), '  Alignment details: ')
            raise RuntimeError(
                f'Inconsistent WER statistics between WER and alignment calculation for {k!r}:\n'
                f'  {k} from WER: {data["info"]["wer"][k]}\n'
                f'  {k} from alignment calculation: {stats_of_by_speaker}\n'
                f'{wer_details}\n'
                f'{alignments_details}'
            )

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
    return fn(d['words'], **kwargs)


class AlignmentVisualization:
    available_colormaps = ['default', 'diff', 'seaborn_muted']

    def __init__(
            self,
            reference,
            hypothesis,
            alignment='tcp',
            colormap='default',
            barplot_style='absolute',
            barplot_scale_exclude_total=False,
            num_minimaps='auto',
            show_details=True,
            show_legend=True,
            highlight_regex=None,
            alignment_transform=None,
            markers=None,
            recording_file: 'str | Path | dict[str, str | Path]' = None,
            js_debug=False,  # If True, don't embed js (and css) code and use absolute paths
            sync_id=None,
            precomputed_error_rate=None,   # A precomputed assignment. Saves computation
            show_playhead=True,
            system_name=None,   # Name of the system to display in the visualization
    ):
        if isinstance(reference, (str, Path)):
            reference = meeteval.io.load(reference)
        if isinstance(hypothesis, (str, Path)):
            hypothesis = meeteval.io.load(hypothesis)
        self.reference = reference
        self.hypothesis = hypothesis
        self.alignment = alignment
        self.colormap = colormap
        self.barplot_style = barplot_style
        self.barplot_scale_exclude_total = barplot_scale_exclude_total
        self.num_minimaps = num_minimaps
        self.show_details = show_details
        self.show_legend = show_legend
        self.highlight_regex = highlight_regex
        self.alignment_transform = alignment_transform
        self.markers = markers
        self.js_debug = js_debug
        if recording_file:
            if not isinstance(recording_file, dict):
                recording_file = {"": os.fspath(recording_file)}
            for k, v in recording_file.items():
                assert os.path.exists(v), (k, v, recording_file)
        else:
            recording_file = {'': ''}
        self.recording_file = recording_file
        self.sync_id = sync_id
        self.precomputed_error_rate = precomputed_error_rate
        self.show_playhead = show_playhead
        self.system_name = system_name

    def _get_colormap(self):
        if isinstance(self.colormap, str):
            if not self.colormap in self.available_colormaps:
                raise ValueError(
                    f'Unknown colormap: {self.colormap}. Use one of '
                    f'{self.available_colormaps} or a custom mapping from '
                    f'match types ("correct", "insertion", "deletion" and '
                    f'"substitution") to (HTML) colors.'
                )
            return f'colormaps.{self.colormap}'
        else:
            colormap_keys = ['correct', 'substitution', 'insertion', 'deletion']
            if self.colormap.keys() != colormap_keys:
                raise ValueError(
                    f'Colormap defined the wrong keys: Need {colormap_keys} '
                    f'but found {list(self.colormap.keys())}'
                )
            return dumps_json(self.colormap, indent=None)

    @cached_property
    def data(self):
        d = get_visualization_data(
            self.reference, 
            self.hypothesis, 
            assignment=self.alignment,
            alignment_transform=self.alignment_transform,
            precomputed_error_rate=self.precomputed_error_rate,
            system_name=self.system_name,
        )
        d['markers'] = self.markers
        return d

    def _iypnb_html_(self):
        """
        Be aware that this writes _a lot of data_ in json format into the
        output cell. This can cause the browser to hang/crash and may produce
        large ipynb files.
        """
        return f'''
            <html>
            <style>
                /* Styles for notebook view */
                body {{
                    margin: 1px;
                    padding: 0;
                    overflow: hidden;
                }}
                
                .meeteval-viz {{
                    width: 100%;
                    height: 80vh; /* 80% of the window height roughly aligns with the visible height in a typical notebook setup */
                }}
            </style>
            {self.html(encode_url=False)}
            </html>
            '''

    def _ipython_display_(self):
        from IPython.display import HTML, display

        reference = meeteval.io.asseglst(self.reference)
        hypothesis = meeteval.io.asseglst(self.hypothesis)

        ref_session_ids = reference.unique('session_id')
        hyp_session_ids = hypothesis.unique('session_id')

        if self.js_debug:
            print('WARNING: js_debug is not supported in ipynb.')
            self.js_debug = False

        if len(ref_session_ids) > 1 or len(hyp_session_ids) > 1:
            session_ids = sorted(ref_session_ids & hyp_session_ids)
            assert len(session_ids) >= 1, (session_ids, ref_session_ids, hyp_session_ids)
            import ipywidgets

            r = reference.groupby('session_id')
            h = hypothesis.groupby('session_id')

            cache = {}
            def func(session_id, alignment):
                key = (session_id, alignment)
                if key not in cache:
                    try:
                        self.reference = r[session_id]
                        self.hypothesis = h[session_id]
                        self.alignment = alignment
                        cache[key] = self._iypnb_html_()
                        del self.data
                    finally:
                        self.reference = reference
                        self.hypothesis = hypothesis
                return HTML(cache[key])

            session_id = ipywidgets.Dropdown(
                options=session_ids, value=session_ids[0])
            alignment = ipywidgets.Dropdown(
                options=['tcp', 'cp'], value=self.alignment)
            ipywidgets.interact(func, session_id=session_id, alignment=alignment)
        else:
            display(HTML(self._iypnb_html_()))

    def html(self, encode_url=True):
        """
        Creates a visualization in HTML format.

        Note: This HTML contains script and link tags that load external
            libraries, so the visualization will not work offline!
            TODO: Add an option to embed the dependencies into the HTML file.
        """
        import platformdirs

        # Generate data
        element_id = 'viz-' + str(uuid.uuid4())

        # Generate HTML and JS for data
        visualize_js = Path(__file__).parent / 'visualize.js'
        if self.js_debug:
            visualize_js = f'''<script src="{visualize_js}" charset="utf-8"></script>'''
        else:
            visualize_js = f'<script>{visualize_js.read_text()}</script>'

        css = (Path(__file__).parent / 'visualize.css')
        if self.js_debug:
            css = f'<link rel="stylesheet" href="{css}"/>'
        else:
            css = f'<style>{css.read_text()}</style>'

        highlight_regex = f'"{self.highlight_regex}"' if self.highlight_regex else 'null'

        cdn = {
            'd3': 'https://cdn.jsdelivr.net/npm/d3@7',
            # 'font_awesome': 'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css',
        }

        cache_dir = Path(platformdirs.user_data_dir('meeteval'))
        def load_cdn(name, url):
            file = cache_dir / name
            try:
                return file.read_text()
            except FileNotFoundError:
                cache_dir.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, file)
                return file.read_text()

        d3 = f'<script>{load_cdn("d3.js", cdn["d3"])}</script>'
        # font_awesome = f'<style>{load_cdn("d3font_awesome.css", cdn["font_awesome"])}</style>'

        # Determine the number of minimaps based on the displayed time length. If it's large, show two minimaps.
        # Else, use one minimap.
        # Assuming an average word rate of 160 words per minute (0.375 s per word) and a full-hd screen with 1920px width,
        # 300 seconds correspond to roughly 2 pixel per word in the minimap on average. At this size, you can still see 
        # individual word errors.
        # Typical end times (`e = [max(g.T['end_time']) for g in meeteval.io.load('ref.seglst.json').groupby('session_id').values()]; (min(e), max(e))`):
        #   - libricss: (597, 615)
        #   - notsofar: (246, 531)
        #   - dipco: (1209, 2753)
        #   - chime6: (7159, 8902)
        if self.num_minimaps == 'auto':
            if self.data['info']['end_time'] > 300:
                num_minimaps = 2
            else:
                num_minimaps = 1
        else:
            num_minimaps = self.num_minimaps

        font_awesome = ''
        html = f'''
            {d3}
            {font_awesome}
            {css}
            <div class="meeteval-viz" id='{element_id}'><div>
            {visualize_js}
            <script>

                function exec() {{
                    // Wait for d3 to load
                    if (typeof d3 !== 'undefined') alignment_visualization(
                        {dumps_json(self.data, indent=1 if self.js_debug else None, sort_keys=False, separators=(',', ':'), float_round=4)},
                        "#{element_id}",
                        {{
                            colors: {self._get_colormap()},
                            barplot: {{
                                style: "{self.barplot_style}",
                                scaleExcludeCorrect: {'true' if self.barplot_scale_exclude_total else 'false'}
                            }},
                            minimaps: {{
                                number: {num_minimaps}
                            }},
                            show_details: {'true' if self.show_details else 'false'},
                            show_legend: {'true' if self.show_legend else 'false'},
                            search_bar: {{
                                initial_query: {highlight_regex}
                            }},
                            recording_file: {dumps_json(self.recording_file, default=os.fspath)},
                            match_width: 0.1,
                            syncID: {dumps_json(self.sync_id, default='null')},
                            audio_server: 'http://localhost:7777',
                            encodeURL: {'true' if encode_url else 'false'},
                            show_playhead: {'true' if self.show_playhead else 'false'},
                        }}
                    );
                    else setTimeout(exec, 100);
                }}
                exec();
                
            </script>
        '''
        return html

    def dump(self, filename):
        # For standalone HTML, we have to
        #   - disable zooming for mobile devices (viewport setting)
        #   - Scale the visualization to the full window size
        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)
        filename.write_text(
            f'''
            <!DOCTYPE html>
            <html lang="en">
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, user-scalable=no" />
            <style>
                /* Styles for full-screen view */
                html {{
                    height: 100%;
                    width: 100%;
                }}

                body {{
                    height: 100%;
                    width: 100%;
                    margin: 1px;
                    padding: 0;
                    /* Make sure that no scroll bars appear */
                    overflow: hidden;
                    font-family: Arial, Helvetica, sans-serif;
                }}
                
                .meeteval-viz {{
                    width: 100%;
                    height: 100%;
                }}
            </style>
            {self.html()}
            </html>
            '''
        )


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
    parser.add_argument(
        '--example-id', type=str, required=True,
        help="The example to visualize, in case there is more than one in the "
             "reference and hypothesis"
    )
    parser.add_argument(
        '--alignment', '-a', type=str, default='tcp',
        choices=['tcp', 'cp']
    )
    parser.add_argument(
        '--output-filename', '-o', type=str,
        default='{example_id}.html'
    )
    parser.add_argument(
        '--html-title', type=str,
        default='MeetEval Visualization {filename}'
    )

    args = parser.parse_args()

    reference = meeteval.io.asseglst(meeteval.io.load(args.reference))
    hypothesis = meeteval.io.asseglst(meeteval.io.load(args.hypothesis))

    if 'session_id' in reference.T.keys():
        session_ids = reference.unique('session_id').union(hypothesis.unique('session_id'))
        if len(session_ids) > 0:
            assert args.example_id is not None
            reference = reference.filter(lambda s: s['session_id'] == args.example_id)
            hypothesis = hypothesis.filter(lambda s: s['session_id'] == args.example_id)
        else:
            assert args.example_id is None or next(iter(session_ids)) == args.example_id, args.example_id
    else:
        assert args.example_id is None, args.example_id

    output_filename = Path(args.output_filename.format(
        example_id=next(iter(reference.unique('session_id')))
    ))

    AlignmentVisualization(
        reference,
        hypothesis,
        alignment=args.alignment,
    ).dump(output_filename)

    print(
        f'Wrote visualization to {output_filename}\n'
        f'To view it, open the file in a browser (e.g., xdg-open {output_filename})'
    )


if __name__ == '__main__':
    cli()
