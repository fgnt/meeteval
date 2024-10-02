from pathlib import Path
import collections
import itertools

import meeteval
from meeteval.viz.visualize import AlignmentVisualization
from meeteval.wer.api import _load_texts
import tqdm


def create_viz_folder(
        reference,
        hypothesiss,
        out,
        alignments='tcp',
        regex=None,
        normalizer=None,
        js_debug=False,
        per_reco_file=None,
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    if isinstance(alignments, str):
        alignments = alignments.split(',')

    if per_reco_file is not None:
        assert len(alignments) == len(per_reco_file), alignments

        error_rate_classes = {
            'tcp': meeteval.wer.CPErrorRate,
            'cp': meeteval.wer.CPErrorRate,
            'tcorc': meeteval.wer.OrcErrorRate,
            'orc': meeteval.wer.OrcErrorRate,
            'greedy_orc': meeteval.wer.OrcErrorRate,
            'greedy_tcorc': meeteval.wer.OrcErrorRate,
            'greedy_dicp': meeteval.wer.DICPErrorRate,
            'greedy_ditcp': meeteval.wer.DICPErrorRate,
        }

        def load_per_reco_file(alignment, f):
            from meeteval.wer.__main__ import _load

            error_rate_cls = error_rate_classes[alignment]

            return {
                session_id: error_rate_cls.from_dict(pr)
                for session_id, pr in _load(Path(f)).items()
            }
        per_reco = {
            alignment: load_per_reco_file(alignment, f)
            for alignment, f in zip(alignments, per_reco_file)
        }
    else:
        per_reco = collections.defaultdict(lambda: collections.defaultdict(lambda: None))

    avs = {}
    for (i, hypothesis), alignment in tqdm.tqdm(list(itertools.product(
            hypothesiss.items(),
            alignments,
    ))):

        r, h = _load_texts(
            reference, hypothesis, regex=regex,
            normalizer=normalizer,
        )

        r = r.sorted('start_time')
        h = h.sorted('start_time')

        r = r.groupby('session_id')
        h = h.groupby('session_id')

        session_ids = set(r.keys()) & set(h.keys())
        xor = set(r.keys()) ^ set(h.keys())
        if xor:
            print(f'Ignore {xor}, because they are not available in reference and hypothesis.')

        for session_id in tqdm.tqdm(session_ids):
            av = AlignmentVisualization(
                r[session_id],
                h[session_id],
                alignment=alignment,
                js_debug=js_debug,
                sync_id=1,
                precomputed_error_rate=per_reco[alignment][session_id],
            )   
            av.dump(out / f'{session_id}_{i}_{alignment}.html')
            avs.setdefault((i, alignment), {})[session_id] = av

    ###########################################################################

    from yattag import Doc
    from yattag import indent

    avs_T = collections.defaultdict(lambda: {k: None for k in avs.keys()})
    for i, v in avs.items():
        for session_id, av in v.items():
            avs_T[session_id][i] = av
    avs_T = dict(avs_T)

    ###########################################################################

    from yattag import Doc
    from yattag import indent

    def get_wer(v):
        error_rate = meeteval.wer.combine_error_rates(*[
            meeteval.wer.ErrorRate.from_dict(
                av.data['info']['wer'])
            for av in v.values()
        ]).error_rate
        return f'{error_rate * 100:.2f} %'

    doc, tag, text = Doc().tagtext()
    doc.asis('<!DOCTYPE html>')
    with tag('html'):
        with tag('head'):
            # When offline, the page will still work.
            # The sort will break and the table style rested.
            with tag('script', src='https://code.jquery.com/jquery-3.6.4.min.js'):
                pass
            with tag('script', src='https://cdnjs.cloudflare.com/ajax/libs/jquery.tablesorter/2.31.2/js/jquery.tablesorter.min.js'):
                pass
            doc.asis('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/jquery.tablesorter/2.31.2/css/theme.default.min.css">')
            with tag('style'):
                n = len(alignments)
                doc.asis(f'''
                    /* Center table */
                    body {{
                        width: fit-content;
                        margin: 0 auto;
                    }}
                    /* Make numbers monospace and right-aligned (aligns the decimal point) */
                    .number {{
                        font-family: monospace;
                        text-align: right;
                    }}
                    td {{
                        text-align: right;
                    }}
                    td:nth-child(1) {{
                        text-align: left;
                    }}
                    tbody tr:nth-child(odd) td {{
                      background-color: #f2f2f2;
                    }}
                    tbody td:nth-child({n}n+2), 
                    thead tr:nth-child(2) th:nth-child({n}n+2),
                    thead tr:nth-child(1) th{{
                        padding-left: 3em;
                    }}
                ''')
        with tag('body'):
            with tag('table', klass='tablesorter', id='myTable', style='width: auto;'):
                with tag('thead'):
                    with tag('tr'):
                        with tag('th', ('data-sorter', 'false')):
                            pass

                        for system, item in itertools.groupby(avs.items(), key=lambda x: x[0][0]):
                            with tag('th', ('data-sorter', "false"), colspan=len(list(item))):
                                doc.text(system)

                        if len(alignments) > 1 or len(hypothesiss) > 1:
                            with tag('th', ('data-sorter', "false"), colspan=2):
                                with tag('span', klass='synced-view'):
                                    pass

                    with tag('tr'):
                        with tag('th'):
                            doc.text('Session ID')

                        for (k, alignment), v in avs.items():
                            with tag('th'):
                                doc.asis(f'{alignment}WER<br>')

                                with tag('span', klass='number'):
                                    doc.text(get_wer(v))

                        if len(alignments) > 1 or len(hypothesiss) > 1:
                            with tag('th', ('data-sorter', "false"), colspan=2):
                                doc.text("Side-by-side views")

                with tag('tbody'):
                    for session_id, v in avs_T.items():
                        with tag('tr'):
                            with tag('td'):
                                doc.text(f'{session_id}')
                            for (i, alignment), av in v.items():
                                with tag('td'):
                                    with tag('span', klass='number'):
                                        wer = av.data['info']['wer']['error_rate']
                                        doc.text(f"{wer * 100:.2f} %")
                                    doc.text(' (')
                                    with tag('a', href=f'{session_id}_{i}_{alignment}.html'):
                                        doc.text('View')
                                    doc.text(')')

                            if len(v) > 1:
                                with tag('td'):
                                    tags = '&'.join(f'{session_id}_{i}_{a}.html' for i, a in v.keys())
                                    with tag('a', href=f'side_by_side_sync.html?{tags}'):
                                        doc.text('SydeBySide')
            doc.asis('''
<script>
    $(document).ready(function() {
        // Initialize tablesorter on the table
        $("#myTable")
        // Read the sorting information from the URL after the tablesorter has been initialized
        .bind("tablesorter-initialized", (e, t) => {
            const urlParams = new URLSearchParams(window.location.search);
            if (urlParams.has('sort')) {
                const idx = urlParams.get('sort');
                const order = urlParams.get('order', 'ascending');
                $("#myTable").trigger('sorton', [[[idx, order]]]);
            }
        })
        // Store sorting information in URL
        .bind("sortEnd", (e, t) => {
            const cols = $(t).find('[aria-sort][aria-sort!="none"]');
            if (cols.length > 0) {
                const col = cols[0];
                const url = new URL(window.location);
                url.searchParams.set('sort', col.cellIndex);
                url.searchParams.set('order', col.getAttribute('aria-sort'));
                window.history.replaceState({}, '', url);
            }
        })
        .tablesorter();
    });
</script>
            ''')

    import shutil
    shutil.copy(Path(__file__).parent / 'side_by_side_sync.html', out / 'side_by_side_sync.html')

    with open(out / "index.html", "w") as text_file:
        text_file.write(indent(doc.getvalue()))
    print(f'Open file://{(out / "index.html").absolute()}')


def html(
        reference,
        hypothesis,
        alignment='tcp',
        regex=None,
        normalizer=None,
        out='viz',
        js_debug=False,
        per_reco_file=None,
):
    def prepare(i: int, h: str):
        if ':' in h and not Path(h).exists():
            # inspired by tensorboard from the --logdir_spec argument.
            name, path = h.split(':', maxsplit=1)
            return name, path
        else:
            if len(hypothesis) > 1:
                return f'System {i}', h
            else:
                return f'System', h

    assert len(reference) == 1, (len(reference), 'At the moment only shared reference is supported.')

    hypothesis = dict([
        prepare(i, h)
        for i, h in enumerate(hypothesis)
    ])

    create_viz_folder(
        reference=reference,
        hypothesiss=hypothesis,
        out=out,
        alignments=alignment,
        regex=regex,
        normalizer=normalizer,
        js_debug=js_debug,
        per_reco_file=per_reco_file,
    )


def cli():
    from meeteval.wer.__main__ import CLI

    class VizCLI(CLI):

        def add_argument(self, command_parser, name, p):
            if name == 'alignment':
                command_parser.add_argument(
                    '--alignment',
                    choices=['tcp', 'cp', 'orc', 'greedy_orc', 'tcorc', 'greedy_tcorc', 'greedy_dicp', 'greedy_ditcp'],
                    nargs='+',
                    help='Specifies which assigment and alignment are used. If a time-constrained algorithm is '
                         'selected for the stream assignment, then a time-constrained alignment will be computed, '
                         'otherwise the "classical" alignment without a time constraint is used.\n'
                         'Multiple alignments can be specified to generate multiple visualizations with a single '
                         'merged overview table and side-by-side views.\n'
                         'Choices:\n'
                         '- cp: cpWER and "classical" alignment\n'
                         '- tcp: tcpWER and time-constrained alignment\n'
                         '- orc: ORC-WER and "classical" alignment.\n'
                         '- greedy_orc: greedy ORC-WER and "classical" alignment.\n'
                         '- tcorc: tcORC-WER and time-constrained alignment.\n'
                         '- greedy_tcorc: greedy tcORC-WER.\n'
                         '- greedy_dicp: greedy DI-cpWER and "classical" alignment.\n'
                         '- greedy_ditcp: greedy DI-tcpWER and time-constrained alignment.',
                )
            elif name == 'hypothesis':
                command_parser.add_argument(
                    '-h', '--hypothesis',
                    help='Hypothesis file(s) in SegLST, STM or CTM format. '
                         'Multiple files can be provided for a side by side view. '
                         'Optionally prefixed with system name, e.g. mysystem:/path/to/hyp.stm',
                    nargs='+', action=self.extend_action,
                    required=True,
                )
            elif name == 'js_debug':
                command_parser.add_argument(
                    '--js-debug',
                    action='store_true',
                    help='Add a debug flag to the HTML output to enable debugging in the browser.'
                )
            elif name == 'per_reco_file':
                command_parser.add_argument(
                    '--per-reco-file',
                    help='A precomputed per-reco file. Loads the WER and (stream) '
                         'assignment information from this file instead of computing it. '
                         'If supplied, the number of files must match the number of alignments specified '
                         'with --alignment.',
                    default=None,
                    nargs='+',
                )
            else:
                return super().add_argument(command_parser, name, p)

    cli = VizCLI()
    cli.add_command(html)
    cli.run()


if __name__ == '__main__':
    cli()
