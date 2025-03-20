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


@pytest.fixture
def output_folder():
    output_folder = Path('converted').absolute()
    output_folder.mkdir(exist_ok=True)
    yield output_folder
    for f in output_folder.glob('*'):
        f.unlink()
    output_folder.rmdir()


@pytest.mark.parametrize(
    'from_format, to_format',
    list(itertools.product(
        ['ctm', 'stm', 'rttm', 'seglst'],
        ['stm', 'rttm', 'seglst'],
    ))
)
def test_converter(from_format, to_format, output_folder):
    run(f'meeteval-io {from_format}2{to_format} {example_files / test_files[from_format]} {output_folder / test_files[to_format]}')
    assert (output_folder / test_files[to_format]).exists()


def test_merge_ctm_filename(output_folder):
    run(f'meeteval-io ctm2stm {example_files / "hyp1.ctm"} {example_files / "hyp2.ctm"} {output_folder / "hyp.stm"}')
    assert (output_folder / "hyp.stm").exists()
    meeteval.io.load(output_folder / "hyp.stm").to_seglst().unique('speaker') == {'hyp1', 'hyp2'}


def test_merge_ctm_speaker_arg(output_folder):
    run(f'meeteval-io ctm2stm --speaker spk-A {example_files / "hyp1.ctm"} {example_files / "hyp2.ctm"} {output_folder / "hyp.stm"}')
    assert (output_folder / "hyp.stm").exists()
    meeteval.io.load(output_folder / "hyp.stm").to_seglst().unique('speaker') == {'spk-A'}

def test_piping(output_folder):
    run(f'cat {example_files / "hyp.stm"} | meeteval-io stm2rttm {output_folder / "hyp.rttm"}')
    run(f'cat {example_files / "hyp.stm"} | meeteval-io stm2rttm -')
    run(f'cat {example_files / "hyp.stm"} | meeteval-io stm2rttm - | echo')
    run(f'cat {example_files / "hyp.stm"} | meeteval-io stm2rttm - > {output_folder / "hyp.rttm"}')
    run(f'meeteval-io stm2rttm <(cat {example_files / "hyp.stm"}) -')

def test_convert_correct(output_folder):
    run(f'meeteval-io stm2rttm {example_files / "hyp.stm"} {output_folder / "hyp.rttm"}')
    assert (output_folder / "hyp.rttm").read_text() == (example_files / "hyp.rttm").read_text()

    # TODO: dump timestamps as str or int / float?
    # run(f'meeteval-io stm2seglst {example_files / "hyp.stm"} {output_folder / "hyp.seglst.json"}')
    # assert (output_folder / "hyp.seglst.json").read_text() == (example_files / "hyp.seglst.json").read_text()

    run(f'meeteval-io rttm2stm {example_files / "hyp.rttm"} {output_folder / "hyp.stm"}')
    assert meeteval.io.load(output_folder / "hyp.stm") == meeteval.io.load(example_files / "hyp.stm").map(lambda l: l.replace(transcript='<NA>'))

    run(f'meeteval-io seglst2stm {example_files / "hyp.seglst.json"} {output_folder / "hyp.stm"} -f')
    assert meeteval.io.load(output_folder / "hyp.stm") == meeteval.io.load(example_files / "hyp.stm")

def test_convert_file_exists(output_folder):
    run(f'meeteval-io stm2rttm {example_files / "hyp.stm"} {output_folder / "hyp.rttm"}')

    with pytest.raises(Exception, match='.*Output file .* already exists.* Use --force / -f to overwrite.'):
        run(f'meeteval-io stm2rttm {example_files / "hyp.stm"} {output_folder / "hyp.rttm"}')

    run(f'meeteval-io stm2rttm --force {example_files / "hyp.stm"} {output_folder / "hyp.rttm"}')
    run(f'meeteval-io stm2rttm -f {example_files / "hyp.stm"} {output_folder / "hyp.rttm"}')

def test_ctm_piping():
    run(f'cat {example_files / "hyp1.ctm"} | meeteval-io ctm2stm --speaker spk-A - > /dev/null')
    with pytest.raises(Exception, match='.*the following arguments are required: --speaker.*'):
        run(f'cat {example_files / "hyp1.ctm"} | meeteval-io ctm2stm - > /dev/null')
    run(f'meeteval-io ctm2stm <(cat {example_files / "hyp1.ctm"}) - > /dev/null')
