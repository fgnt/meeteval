import itertools
from pathlib import Path
import pytest
import subprocess
import meeteval

example_files = (Path(__file__).parent.parent / 'example_files').absolute()

test_files = {
    'ctm': 'hyp1.ctm',
    'stm': 'hyp.stm',
    'rttm': 'hyp.rttm',
    'seglst': 'hyp.seglst.json',
}

def run(cmd):
    cp = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        universal_newlines=True,
        cwd=example_files,
        executable='bash',  # echo "<(cat hyp.stm)" requires bash not sh.
    )

    if cp.returncode == 0:
        return cp
    else:
        if not isinstance(cmd, str):
            import shlex
            cmd = shlex.join(cmd)
        raise Exception(
            f'$ {cmd}'
            f'\n\nreturncode: {cp.returncode}'
            f'\n\nstdout:\n{cp.stdout}'
            f'\n\nstderr:\n{cp.stderr}'
        )


@pytest.mark.parametrize(
    'from_format, to_format',
    list(itertools.product(
        ['ctm', 'stm', 'rttm', 'seglst'],
        ['stm', 'rttm', 'seglst'],
    ))
)
def test_converter(from_format, to_format, tmp_path):
    run(f'meeteval-io {from_format}2{to_format} {example_files / test_files[from_format]} {tmp_path / test_files[to_format]}')
    assert (tmp_path / test_files[to_format]).exists()


def test_merge_ctm_filename(tmp_path):
    run(f'meeteval-io ctm2stm {example_files / "hyp1.ctm"} {example_files / "hyp2.ctm"} {tmp_path / "hyp.stm"}')
    assert (tmp_path / "hyp.stm").exists()
    meeteval.io.load(tmp_path / "hyp.stm").to_seglst().unique('speaker') == {'hyp1', 'hyp2'}


def test_merge_ctm_speaker_arg(tmp_path):
    run(f'meeteval-io ctm2stm --speaker spk-A {example_files / "hyp1.ctm"} {example_files / "hyp2.ctm"} {tmp_path / "hyp.stm"}')
    assert (tmp_path / "hyp.stm").exists()
    meeteval.io.load(tmp_path / "hyp.stm").to_seglst().unique('speaker') == {'spk-A'}

def test_piping(tmp_path):
    run(f'cat {example_files / "hyp.stm"} | meeteval-io stm2rttm {tmp_path / "hyp.rttm"}')
    run(f'cat {example_files / "hyp.stm"} | meeteval-io stm2rttm -')
    run(f'cat {example_files / "hyp.stm"} | meeteval-io stm2rttm - | echo')
    run(f'cat {example_files / "hyp.stm"} | meeteval-io stm2rttm - > {tmp_path / "hyp.rttm"}')
    run(f'meeteval-io stm2rttm <(cat {example_files / "hyp.stm"}) -')

def test_convert_correct(tmp_path):
    run(f'meeteval-io stm2rttm {example_files / "hyp.stm"} {tmp_path / "hyp.rttm"}')
    assert (tmp_path / "hyp.rttm").read_text() == (example_files / "hyp.rttm").read_text()

    # TODO: dump timestamps as str or int / float?
    # run(f'meeteval-io stm2seglst {example_files / "hyp.stm"} {tmp_path / "hyp.seglst.json"}')
    # assert (tmp_path / "hyp.seglst.json").read_text() == (example_files / "hyp.seglst.json").read_text()

    run(f'meeteval-io rttm2stm {example_files / "hyp.rttm"} {tmp_path / "hyp.stm"}')
    assert meeteval.io.load(tmp_path / "hyp.stm") == meeteval.io.load(example_files / "hyp.stm").map(lambda l: l.replace(transcript='<NA>'))

    run(f'meeteval-io seglst2stm {example_files / "hyp.seglst.json"} {tmp_path / "hyp.stm"} -f')
    assert meeteval.io.load(tmp_path / "hyp.stm") == meeteval.io.load(example_files / "hyp.stm")

def test_convert_file_exists(tmp_path):
    run(f'meeteval-io stm2rttm {example_files / "hyp.stm"} {tmp_path / "hyp.rttm"}')

    with pytest.raises(Exception, match='.*Output file .* already exists.* Use --force / -f to overwrite.'):
        run(f'meeteval-io stm2rttm {example_files / "hyp.stm"} {tmp_path / "hyp.rttm"}')

    run(f'meeteval-io stm2rttm --force {example_files / "hyp.stm"} {tmp_path / "hyp.rttm"}')
    run(f'meeteval-io stm2rttm -f {example_files / "hyp.stm"} {tmp_path / "hyp.rttm"}')

def test_ctm_piping():
    run(f'cat {example_files / "hyp1.ctm"} | meeteval-io ctm2stm --speaker spk-A - > /dev/null')
    with pytest.raises(Exception, match='.*the following arguments are required: --speaker.*'):
        run(f'cat {example_files / "hyp1.ctm"} | meeteval-io ctm2stm - > /dev/null')
    with pytest.raises(Exception, match='.*not a regular file.*'):
        run(f'meeteval-io ctm2stm <(cat {example_files / "hyp1.ctm"}) - > /dev/null')

def test_placeholder_replacement():
    """Tests that placeholders are replaced with information from the segments or the file stem."""
    out = run(f'meeteval-io rttm2stm {example_files / "hyp.rttm"} --words "<NA>" -').stdout
    assert all(l.transcript == '<NA>' for l in meeteval.io.STM.parse(out))

    out = run(f'meeteval-io rttm2stm {example_files / "hyp.rttm"} --words "{{filestem}}-{{session_id}}" -').stdout
    assert all(l.transcript == f'hyp-{l.filename}' for l in meeteval.io.STM.parse(out))

    out = run(f'meeteval-io ctm2seglst {example_files / "hyp1.ctm"} -').stdout
    assert all(l['speaker'] == 'hyp1' for l in meeteval.io.SegLST.parse(out))

    out = run(f'meeteval-io ctm2seglst {example_files / "hyp1.ctm"} --speaker "{{filestem}}-{{session_id}}-{{start_time}}-{{end_time}}-{{words}}" -').stdout
    assert all(l['speaker'] == f'hyp1-{l["session_id"]}-{l["start_time"]}-{l["end_time"]}-{l["words"]}' for l in meeteval.io.SegLST.parse(out))

    with pytest.raises(Exception, match='.*Key "argh" not found in segment.*'):
        run(f'meeteval-io ctm2seglst {example_files / "hyp1.ctm"} --speaker "{{argh}}" -')
