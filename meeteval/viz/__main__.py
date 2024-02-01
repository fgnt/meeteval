from pathlib import Path
import collections

import meeteval
from meeteval.viz.visualize import AlignmentVisualization
import tqdm

def create_viz_folder(
        reference,
        hypothesiss,
        out,
        alignment='tcp',
):
    out = Path(out)
    if isinstance(reference, (str, Path)):
        reference = meeteval.io.load(reference).to_seglst()

    reference = reference.groupby('session_id')
    out.mkdir(parents=True, exist_ok=True)

    avs = {}
    for i, hypothesis in tqdm.tqdm(hypothesiss.items()):
        if isinstance(hypothesis, (str, Path)):
            hypothesis = meeteval.io.load(hypothesis).to_seglst()

        hypothesis = hypothesis.groupby('session_id')

        session_ids = set(reference.keys()) & set(hypothesis.keys())

        for session_id in session_ids:
            av = AlignmentVisualization(reference[session_id],
                                        hypothesis[session_id],
                                        alignment=alignment)
            av.dump(out / f'{session_id}_{i}.html')
            avs.setdefault(i, {})[session_id] = av

    # meeteval.wer.combine_error_rates([
    #     meeteval.wer.ErrorRate.from_dict(av.data['info']['wer'] for av in avs.values())
    #
    # ])

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
        with tag('html'):
            with tag('style'):
                doc.asis("""html, body {height: 100%; margin: 0;}""")
                doc.asis(
                    """.full-height {height: 100%; background: yellow;}""")
            with tag('body',
                     style="width: 100%; height: 100%; display: flex;"):
                for i, av in v.items():
                    with tag('div',
                             style=f"width: {100 // len(v)}%; height: 100%;"):
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

                    # meeteval.wer.combine_error_rates([
                    #         meeteval.wer.ErrorRate.from_dict(av.data['info']['wer'] for av in avs.values())
                    #
                    #     ])
                    def get_wer(v):
                        print(list(v.values())[0].data['info']['wer']['hypothesis'])
                        error_rate = meeteval.wer.combine_error_rates(*[
                            meeteval.wer.ErrorRate.from_dict(av.data['info']['wer']['hypothesis'])
                           for av in v.values()
                        ]).error_rate
                        return f'{error_rate*100:.2f} %'

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
                                    wer = av.data['info']['wer']['hypothesis'][
                                        'error_rate']
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


def main(ref, *, out, alignment='tcp', **kwargs):
    """
    """
    print('kwargs', kwargs)
    print('alignment', alignment)
    print('out', out)
    print('ref', ref)
    create_viz_folder(
        reference=ref,
        hypothesiss=kwargs,
        out=out,
        alignment=alignment,
    )


if __name__ == '__main__':
    import fire
    fire.Fire(main)
