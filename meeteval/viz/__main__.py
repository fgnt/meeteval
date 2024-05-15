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
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    avs = {}
    for (i, hypothesis), alignment in tqdm.tqdm(list(itertools.product(
            hypothesiss.items(),
            alignments.split(','),
    ))):

        r, h = _load_texts(
            reference, hypothesis, regex=regex,
            reference_sort='segment',
            hypothesis_sort='segment',
            normalizer=normalizer,
        )

        r = r.groupby('session_id')
        h = h.groupby('session_id')

        session_ids = set(r.keys()) & set(h.keys())
        xor = set(r.keys()) ^ set(h.keys())
        if xor:
            print(f'Ignore {xor}, because they are not available in reference and hypothesis.')

        for session_id in tqdm.tqdm(session_ids):
            av = AlignmentVisualization(r[session_id],
                                        h[session_id],
                                        alignment=alignment,
                                        js_debug=js_debug,
                                        sync_id=1)
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

    for session_id, v in avs_T.items():
        doc, tag, text = Doc().tagtext()
        doc.asis('<!DOCTYPE html>')

        # With 100 % there is a scroll bar -> use 99 %
        with tag('html', style="height: 99%; margin: 0;"):
            with tag('body', style="width: 100%; height: 100%; margin: 0; display: flex;"):
                for (i, alignment), av in v.items():
                    with tag('div', style='flex-grow: 1'):
                        with tag('iframe', src=f'{session_id}_{i}_{alignment}.html',
                                 title="right", width="100%",
                                 height="100%", style="border-width: 0"):
                            pass

        file = out / f"{session_id}.html"
        file.write_text(indent(doc.getvalue()))
        print(f'Wrote {file.absolute()}')

    ###########################################################################

    from yattag import Doc
    from yattag import indent

    def get_wer(v):
        error_rate = meeteval.wer.combine_error_rates(*[
            meeteval.wer.ErrorRate.from_dict(
                av.data['info']['wer']['hypothesis'])
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
                doc.asis('''
                    /* Center table */
                    body {
                        width: fit-content;
                        margin: 0 auto;
                    }
                    /* Make numbers monospace and right-aligned (aligns the decimal point) */
                    .number {
                        font-family: monospace;
                        text-align: right;
                    }
                    td {
                        text-align: right;
                    }
                    td:nth-child(1) {
                        text-align: left;
                    }
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

                        if ',' in alignments or len(hypothesiss) > 1:
                            with tag('th', ('data-sorter', "false"), colspan=2):
                                with tag('span', klass='synced-view'):
                                    pass

                    with tag('tr'):
                        with tag('th'):
                            doc.text('Session ID')

                        for (k, alignment), v in avs.items():
                            with tag('th'):
                                doc.text(f'{alignment}WER: ')
                                with tag('span', klass='number'):
                                    doc.text(get_wer(v))

                        if ',' in alignments or len(hypothesiss) > 1:
                            with tag('th', ('data-sorter', "false"), colspan=2):
                                doc.text("Side-by-side views")

                with tag('tbody'):
                    for session_id, v in avs_T.items():
                        with tag('tr'):
                            with tag('td'):
                                doc.text(f'{session_id}')
                            for (i, alignment), av in v.items():
                                with tag('td'):
                                    with tag('a',
                                             href=f'{session_id}_{i}_{alignment}.html'):
                                        doc.text('View')
                                with tag('td'):
                                    wer = av.data['info']['wer']['hypothesis']['error_rate']
                                    doc.text(f"{wer * 100:.2f} %")

                            if len(v) > 1:
                                with tag('td'):
                                    with tag('a', href=f'{session_id}.html'):
                                        doc.text('View SideBySide')
                                with tag('td'):
                                    tags = '&'.join(f'{session_id}_{i}_{a}' for i, a in v.keys())
                                    with tag('a', href=f'side_by_side_sync.html?{tags}'):
                                        doc.text('View SydeBySide Synced')
            doc.asis('''
<script>
    $(document).ready(function() {
        // Initialize tablesorter on the table
        $("#myTable").tablesorter();
    });
</script>
            ''')

    import shutil
    shutil.copy(Path(__file__).parent / 'side_by_side_sync.html', out / 'side_by_side_sync.html')

    with open(out / "index.html", "w") as text_file:
        text_file.write(indent(doc.getvalue()))
    print(f'Open {(out / "index.html").absolute()}')


def html(
        reference,
        hypothesis,
        alignment='tcp',
        regex=None,
        normalizer=None,
        out='viz',
        js_debug=False,
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
    )


def cli():
    from meeteval.wer.__main__ import CLI

    class VizCLI(CLI):

        def add_argument(self, command_parser, name, p):
            if name == 'alignment':
                command_parser.add_argument(
                    '--alignment',
                    choices=['tcp', 'cp', 'tcp,cp', 'cp,tcp', 'tcorc', 'orc'],
                    help='Specifies which alignment is used.\n'
                         '- cp: Find the permutation that minimizes the cpWER and use the "classical" alignment.\n'
                         '- tcp: Find the permutation that minimizes the tcpWER and use a time constraint alignment.'
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
            else:
                return super().add_argument(command_parser, name, p)

    cli = VizCLI()
    cli.add_command(html)
    cli.run()


if __name__ == '__main__':
    cli()
