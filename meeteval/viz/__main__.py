from pathlib import Path
import collections

import meeteval
from meeteval.viz.visualize import AlignmentVisualization
from meeteval.wer.api import _load_texts
import tqdm


def create_viz_folder(
        reference,
        hypothesiss,
        out,
        alignment='tcp',
        regex=None,
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    avs = {}
    for i, hypothesis in tqdm.tqdm(hypothesiss.items()):

        r, h = _load_texts(
            reference, hypothesis, regex=regex,
            reference_sort='segment',
            hypothesis_sort='segment',
        )

        r = r.groupby('session_id')
        h = h.groupby('session_id')

        session_ids = set(r.keys()) & set(h.keys())
        xor = set(r.keys()) ^ set(h.keys())
        if xor:
            print(f'Ignore {xor}, because they are not available in reference and hypothesis.')

        for session_id in session_ids:
            av = AlignmentVisualization(r[session_id],
                                        h[session_id],
                                        alignment=alignment)
            av.dump(out / f'{session_id}_{i}.html')
            avs.setdefault(i, {})[session_id] = av

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
                for i, av in v.items():
                    with tag('div', style='flex-grow: 1'):
                        with tag('iframe', src=f'{session_id}_{i}.html',
                                 title="right", width="100%",
                                 height="100%"):
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
            with tag('script', src='https://code.jquery.com/jquery-3.6.4.min.js'):
                pass
            with tag('script', src='https://cdnjs.cloudflare.com/ajax/libs/jquery.tablesorter/2.31.2/js/jquery.tablesorter.min.js'):
                pass
            doc.asis('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/jquery.tablesorter/2.31.2/css/theme.default.min.css">')
        with tag('body'):
            with tag('table', klass='tablesorter', id='myTable'):
                with tag('thead'), tag('tr'):
                    for s in [
                        'Session ID',
                        *[
                            col
                            for k, v in avs.items()
                            for col in [k, f'WER: {get_wer(v)}']
                        ]
                    ]:
                        with tag('th'):
                            doc.text(s)

                with tag('tbody'):
                    for session_id, v in avs_T.items():
                        with tag('tr'):
                            with tag('td'):
                                doc.text(f'{session_id}')
                            for i, av in v.items():
                                with tag('td'):
                                    with tag('a',
                                             href=f'{session_id}_{i}.html'):
                                        doc.text('viz')
                                with tag('td'):
                                    wer = av.data['info']['wer']['hypothesis']['error_rate']
                                    doc.text(f"{wer * 100:.2f} %")

                            with tag('td'):
                                with tag('a', href=f'{session_id}.html'):
                                    doc.text('SideBySide viz')
            doc.asis('''
<script>
    $(document).ready(function() {
        // Initialize tablesorter on the table
        $("#myTable").tablesorter();
    });
</script>
            ''')

    with open(out / "index.html", "w") as text_file:
        text_file.write(indent(doc.getvalue()))
    print(f'Open {(out / "index.html").absolute()}')


def html(
        reference,
        hypothesis,
        alignment='tcp',
        regex=None,
        out='viz',
):
    def prepare(i: int, h: str):
        if ':' in h and not Path(h).exists():
            # inspired by tensorboard from the --logdir_spec argument.
            name, path = h.split(':', maxsplit=1)
            return name, path
        else:
            return f'sys{i}', h

    assert len(reference) == 1, (len(reference), 'At the moment only shared reference is supported.')

    hypothesis = dict([
        prepare(i, h)
        for i, h in enumerate(hypothesis)
    ])

    create_viz_folder(
        reference=reference,
        hypothesiss=hypothesis,
        out=out,
        alignment=alignment,
        regex=regex,
    )


def cli():
    from meeteval.wer.__main__ import CLI

    class VizCLI(CLI):

        def add_argument(self, command_parser, name, p):
            if name == 'alignment':
                command_parser.add_argument(
                    '--alignment',
                    choices=['tcp', 'cp'],
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
            else:
                return super().add_argument(command_parser, name, p)

    cli = VizCLI()
    cli.add_command(html)
    cli.run()


if __name__ == '__main__':
    cli()
