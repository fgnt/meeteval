import meeteval.viz
from pathlib import Path
import pytest

example_files = (Path(__file__).parent.parent / 'example_files').absolute()


alignments = [
    'cp', 'tcp', 'orc', 'tcorc', 'greedy_tcorc',
    'greedy_orc', 'greedy_dicp', 'greedy_ditcp'
]


@pytest.mark.parametrize('alignment', alignments)
def test_viz_burn(alignment):
    """
    Tests if the code that generated the visualization produces an html file.
    Does not test if the visualization is correct.
    """
    ref = meeteval.io.asseglst(meeteval.io.load(example_files / 'hyp.stm')).groupby('session_id')
    hyp = meeteval.io.asseglst(meeteval.io.load(example_files / 'ref.stm')).groupby('session_id')

    for k in ref.keys():
        meeteval.viz.AlignmentVisualization(
            ref[k],
            hyp[k],
            alignment=alignment,
        ).dump(example_files / f'viz/test-{k}-{alignment}.html')
        assert (example_files / f'viz/test-{k}-{alignment}.html').exists()


@pytest.mark.parametrize('alignment', alignments)
def test_viz_precompute_wer(alignment):
    """
    Tests if the code that generated the visualization produces an html file
    when the WER (and assignment) are precomputed.
    Does not test if the visualization is correct.
    """
    ref = meeteval.io.asseglst(meeteval.io.load(example_files / 'hyp.stm')).groupby('session_id')
    hyp = meeteval.io.asseglst(meeteval.io.load(example_files / 'ref.stm')).groupby('session_id')

    for k in ref.keys():
        # Precompute WER
        wer = getattr(meeteval.wer.api, alignment + 'wer')(
            ref[k], hyp[k],
            **({'collar': 5} if 'tc' in alignment else {})
        )[k]

        # With precomputed WER
        meeteval.viz.AlignmentVisualization(
            ref[k],
            hyp[k],
            alignment=alignment,
            precomputed_error_rate=wer,
        ).dump(example_files / f'viz/test-{k}-{alignment}-precomputed.html')
        assert (example_files / f'viz/test-{k}-{alignment}-precomputed.html').exists()

        # Without precomputed WER
        meeteval.viz.AlignmentVisualization(
            ref[k],
            hyp[k],
            alignment=alignment,
        ).dump(example_files / f'viz/test-{k}-{alignment}.html')
        assert (example_files / f'viz/test-{k}-{alignment}.html').exists()

        # Test that file contents are identical. Only the (randomly generated)
        # visualization ID should differ.
        import re
        precomputed_text = (example_files / f'viz/test-{k}-{alignment}-precomputed.html').read_text()
        precomputed_text = re.sub(r'viz-[0-9a-f\-]*', '#viz-XXXX', precomputed_text)
        text = (example_files / f'viz/test-{k}-{alignment}.html').read_text()
        text = re.sub(r'viz-[0-9a-f\-]*', '#viz-XXXX', text)
        assert text == precomputed_text


def test_viz_index_html(tmp_path):
    (tmp_path / 'viz').mkdir()
    from meeteval.viz.__main__ import index_html

    ref = meeteval.io.asseglst(meeteval.io.load(example_files / 'hyp.stm')).groupby('session_id')
    hyp = meeteval.io.asseglst(meeteval.io.load(example_files / 'ref.stm')).groupby('session_id')

    for k in ref.keys():
        meeteval.viz.AlignmentVisualization(
            ref[k],
            hyp[k],
            alignment='cp',
            system_name='System1',
        ).dump(tmp_path / 'viz' / f'test_{k}_cp.html')

        meeteval.viz.AlignmentVisualization(
            ref[k],
            hyp[k],
            alignment='tcp',
            system_name='System1',
        ).dump(tmp_path / 'viz' / f'test_{k}_tcp.html')

    def get_index_html_data(path):
        import re
        import yaml
        content = path.read_text()
        data = re.search('data = (\[\n(.|\n)*\n\s*]);', content).groups(1)[0]
        data = yaml.safe_load(data) # JSON complains about trailing comma
        return data
    
    # Default case: Generate index file for folder
    index_html(
        list((tmp_path / 'viz').glob('*.html')),
        out=tmp_path / 'viz',
        copy=False,
    )
    data = get_index_html_data(tmp_path / 'viz' / 'index.html')
    assert len(data) == 2

    # Generate index file only for one metric
    index_html(
        list((tmp_path / 'viz').glob('*_tcp.html')),
        out=tmp_path / 'viz' / 'index_tcp.html',
        copy=False,
    )
    assert (tmp_path / 'viz' / 'index_tcp.html').exists()
    data = get_index_html_data(tmp_path / 'viz' / 'index_tcp.html')
    assert len(data) == 1

    # Test that the copy option copies the correct number of files
    index_html(
        list((tmp_path / 'viz').glob('*_cp.html')),
        out=tmp_path / 'viz_cp' / 'index_cp.html',
        copy=True,
    )
    assert (tmp_path / 'viz_cp' / 'index_cp.html').exists()
    assert len(list((tmp_path / 'viz_cp').iterdir())) == len(ref.keys()) + 2


@pytest.mark.parametrize('alignment', alignments)
def test_viz_speaker_mismatch(alignment):
    """
    Tests if the code that generated the visualization produces an html file.
    Does not test if the visualization is correct.
    """
    ref1 = meeteval.io.asseglst(meeteval.io.STM.parse(
        'rec1 0 spk1 10 20 Hello World\n'
    ))
    ref2 = meeteval.io.asseglst(meeteval.io.STM.parse(
        'rec1 0 spk1 10 20 Hello World\n'
        'rec1 0 spk2 20 30 Goodbye World\n'
    ))
    hyp1 = meeteval.io.asseglst(meeteval.io.STM.parse(
        'rec1 0 spk1 10 20 Hello World\n'
    ))
    hyp2 = meeteval.io.asseglst(meeteval.io.STM.parse(
        'rec1 0 spk1 10 20 Hello World\n'
        'rec1 0 spk2 20 30 Goodbye World\n'
    ))

    for i, (ref, hyp) in enumerate([
        (ref1, hyp2),
        (ref2, hyp1)
    ]):
        meeteval.viz.AlignmentVisualization(
            ref,
            hyp,
            alignment=alignment,
        ).dump(example_files / f'viz/test-mismatch-{i}-{alignment}.html')
        assert (example_files / f'viz/test-mismatch-{i}-{alignment}.html').exists()
