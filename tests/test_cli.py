import subprocess
from pathlib import Path
import shutil
import pytest

example_files = (Path(__file__).parent.parent / 'example_files').absolute()


def run(cmd, cwd=example_files):
    cp = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        universal_newlines=True,
        cwd=cwd,
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
    
def test_burn_orc():
    # Normal test with stm files
    run(f'python -m meeteval.wer orcwer -h hyp.stm -r ref.stm')
    # assert (example_files / 'hyp_orc.json').exists()
    # assert (example_files / 'hyp_orc_per_reco.json').exists()
    run(f'meeteval-wer orcwer -h hyp.stm -r ref.stm')

    # Multiple stm files
    run(f"python -m meeteval.wer orcwer -h hypA.stm -h hypB.stm -r refA.stm -r refB.stm")
    run(f"python -m meeteval.wer orcwer -h hyp.stm -h hypA.stm hypB.stm -r refA.stm refB.stm")

    # Test with glob (backwards compatibility). Note: '?' and '*' are escaped.
    # Be careful, that the glob only matches the desired files.
    # The 'hyp*.stm' will here match 'hyp.stm', 'hypA.stm' and 'hypB.stm'.
    run(f"python -m meeteval.wer orcwer -h 'hyp*.stm' -r 'ref*.stm'")
    run(f"python -m meeteval.wer orcwer -h 'hyp?.stm' -r 'ref?.stm'")

    # Test with shell glob
    run(f"python -m meeteval.wer orcwer -h hyp*.stm -r ref*.stm")
    run(f"python -m meeteval.wer orcwer -h hyp?.stm -r ref?.stm")

    # Test with ctm files
    run(f'python -m meeteval.wer orcwer -h hyp1.ctm -h hyp2.ctm -r ref.stm')
    run(f"python -m meeteval.wer orcwer -h 'hyp*.ctm' -r ref.stm")
    run(f'python -m meeteval.wer orcwer -h hyp1.ctm -r ref.stm')

    # Test output formats
    run(f"python -m meeteval.wer orcwer -h hyp*.stm -r ref*.stm --average-out average-out.json")
    # assert (example_files / 'average-out.json').exists()
    run("python -m meeteval.wer orcwer -h hyp*.stm -r ref*.stm --average-out '{parent}/{stem}-average-out.yaml'")
    # assert (example_files / 'hyp-average-out.yaml').exists()
    # Output to stdout. Specifying the format requires =
    run(f"python -m meeteval.wer orcwer -h hyp*.stm -r ref*.stm --average-out -")
    run(f"python -m meeteval.wer orcwer -h hyp*.stm -r ref*.stm --average-out=-.yaml")
    run(f"python -m meeteval.wer orcwer -h hyp*.stm -r ref*.stm --average-out=-.json")

    # Test with pipes. Makes "--average-out" file and "--per-reco-out" file
    # mandatory.
    run(f'python -m meeteval.wer orcwer -h <(cat hypA.stm hypB.stm) -r <(cat refA.stm refB.stm) --average-out hyp_orc.json --per-reco-out hyp_orc_per_reco.json')

    # Test with files in SegLST format
    run(f'python -m meeteval.wer orcwer -h hyp.seglst.json -r ref.seglst.json')

    # Test with regex
    run('python -m meeteval.wer orcwer -h hyp.stm -r ref.stm --regex ".*A"')
    run('python -m meeteval.wer orcwer -h hyp.seglst.json -r ref.seglst.json --regex ".*A"')

    # Test sort option. Only test the ones that are available for ORC
    run('python -m meeteval.wer orcwer -h hyp.stm -r ref.stm --reference-sort "segment" --hypothesis-sort "false"')


def test_burn_greedy_orc():
    # Normal test with stm files
    run(f'python -m meeteval.wer greedy_orcwer -h hyp.stm -r ref.stm')
    run(f'meeteval-wer greedy_orcwer -h hyp.stm -r ref.stm')

    # Test sort option. Only test the ones that are available for ORC
    run('python -m meeteval.wer greedy_orcwer -h hyp.stm -r ref.stm --reference-sort "segment" --hypothesis-sort "false"')


def test_burn_greedy_dicp():
    # Normal test with stm files
    run(f'python -m meeteval.wer greedy_dicpwer -h hyp.stm -r ref.stm')
    run(f'meeteval-wer greedy_dicpwer -h hyp.stm -r ref.stm')


def test_burn_greedy_ditcp():
    # Normal test with stm files
    run(f'python -m meeteval.wer greedy_ditcpwer -h hyp.stm -r ref.stm --collar 5')
    run(f'meeteval-wer greedy_ditcpwer -h hyp.stm -r ref.stm --collar 5')


def test_burn_mimo():
    run(f'python -m meeteval.wer mimower -h hyp.stm -r ref.stm')
    run(f"python -m meeteval.wer mimower -h 'hyp?.stm' -r 'ref?.stm'")
    run(f'python -m meeteval.wer mimower -h hyp.seglst.json -r ref.seglst.json')
    run('python -m meeteval.wer mimower -h hyp.stm -r ref.stm --reference-sort "segment" --hypothesis-sort "false"')


def test_burn_tcmimo():
    run(f'python -m meeteval.wer tcmimower -h hyp.stm -r ref.stm --collar 5')
    run(f'python -m meeteval.wer tcmimower -h hyp.stm -r ref.stm --hyp-pseudo-word-timing equidistant_points --collar 5')
    run(f'python -m meeteval.wer tcmimower -h hyp.seglst.json -r ref.seglst.json --collar 5')
    run(f'python -m meeteval.wer tcmimower -h hyp.stm -r ref.stm --hypothesis-sort true --collar 5')


def test_burn_cp():
    run(f'python -m meeteval.wer cpwer -h hyp.stm -r ref.stm')
    run(f"python -m meeteval.wer cpwer -h 'hyp?.stm' -r 'ref?.stm'")
    run(f'python -m meeteval.wer cpwer -h hyp.seglst.json -r ref.seglst.json')
    run('python -m meeteval.wer cpwer -h hyp.stm -r ref.stm --reference-sort "segment" --hypothesis-sort "false"')
    run(f'python -m meeteval.wer cpwer -h hyp.stm -r ref.stm --uem uem.uem')


def test_burn_tcp():
    # run(f'python -m meeteval.wer tcpwer -h hyp.stm -r ref.stm')  # Mar 2025: Disabled, because default collar=0 is too a too special case to be default
    run(f'python -m meeteval.wer tcpwer -h hyp.stm -r ref.stm --collar 5')
    run(f'python -m meeteval.wer tcpwer -h hyp.stm -r ref.stm --hyp-pseudo-word-timing equidistant_points --collar 5')
    run(f'python -m meeteval.wer tcpwer -h hyp.seglst.json -r ref.seglst.json --collar 5')
    run(f'python -m meeteval.wer tcpwer -h hyp.stm -r ref.stm --reference-sort word --hypothesis-sort true --collar 5')


def test_burn_tcorc():
    # run(f'python -m meeteval.wer tcorcwer -h hyp.stm -r ref.stm')  # Mar 2025: Disabled, because default collar=0 is too a too special case to be default
    run(f'python -m meeteval.wer tcorcwer -h hyp.stm -r ref.stm --collar 5')
    run(f'python -m meeteval.wer tcorcwer -h hyp.stm -r ref.stm --hyp-pseudo-word-timing equidistant_points --collar 5')
    run(f'python -m meeteval.wer tcorcwer -h hyp.seglst.json -r ref.seglst.json --collar 5')
    run(f'python -m meeteval.wer tcorcwer -h hyp.stm -r ref.stm --hypothesis-sort true --collar 5')


def test_burn_greedy_tcorc():
    # run(f'python -m meeteval.wer greedy_tcorcwer -h hyp.stm -r ref.stm')  # Mar 2025: Disabled, because default collar=0 is too a too special case to be default
    run(f'python -m meeteval.wer greedy_tcorcwer -h hyp.stm -r ref.stm --collar 5')
    run(f'python -m meeteval.wer greedy_tcorcwer -h hyp.stm -r ref.stm --hyp-pseudo-word-timing equidistant_points --collar 5')
    run(f'python -m meeteval.wer greedy_tcorcwer -h hyp.seglst.json -r ref.seglst.json --collar 5')
    run(f'python -m meeteval.wer greedy_tcorcwer -h hyp.stm -r ref.stm --hypothesis-sort true --collar 5')


def test_burn_md_eval_22():
    run(f'python -m meeteval.der md_eval_22 -h hyp.stm -r ref.stm')
    run(f'meeteval-der md_eval_22 -h hyp.stm -r ref.stm')
    run(f'python -m meeteval.der md_eval_22 -h hyp.stm -r ref.stm --collar 0.25')
    run(f'python -m meeteval.der md_eval_22 -h hyp.rttm -r ref.rttm')
    run(f'python -m meeteval.der md_eval_22 -h hyp.rttm -r ref.rttm --regions all')
    run(f'python -m meeteval.der md_eval_22 -h hyp.rttm -r ref.rttm --regions nooverlap')
    run(f'python -m meeteval.der md_eval_22 -h hyp.rttm -r ref.rttm --regex ".*A"')
    run(f'python -m meeteval.der md_eval_22 -h hyp.seglst.json -r ref.seglst.json')
    # examples for collar:
    #    0:    CHiME-6
    #    0.25: CHiME-7 DASR
    # ToDo: Table 2 of https://arxiv.org/pdf/2312.04324.pdf lists collars for
    #       datsets. Add them here.

def test_burn_dscore():
    run(f'python -m meeteval.der dscore -h hyp.stm -r ref.stm')
    run(f'meeteval-der dscore -h hyp.stm -r ref.stm')
    run(f'python -m meeteval.der dscore -h hyp.stm -r ref.stm --collar 0.25')
    run(f'python -m meeteval.der dscore -h hyp.rttm -r ref.rttm')
    run(f'python -m meeteval.der dscore -h hyp.rttm -r ref.rttm --regions all')
    run(f'python -m meeteval.der dscore -h hyp.rttm -r ref.rttm --regions nooverlap')
    run(f'python -m meeteval.der dscore -h hyp.rttm -r ref.rttm --regex ".*A"')
    run(f'python -m meeteval.der dscore -h hyp.seglst.json -r ref.seglst.json')

def test_burn_merge():
    run(f'python -m meeteval.wer cpwer -h hypA.stm -r refA.stm')  # create hypA_cpwer_per_reco.json and hypA_cpwer.json
    run(f'python -m meeteval.wer cpwer -h hypB.stm -r refB.stm')  # create hypB_cpwer_per_reco.json and hypB_cpwer.json

    # Merge average files to new average file (printed to stdout).
    run(f'python -m meeteval.wer merge hypA_cpwer.json hypB_cpwer.json')

    # Merge per_reco files to new per_reco file (printed to stdout).
    # See test_burn_average, how to create an average file from per_reco
    run(f'python -m meeteval.wer merge hypA_cpwer_per_reco.json hypB_cpwer_per_reco.json')

    # When input contains average and per_reco files, output will be a per_reco file (printed to stdout).
    run(f'python -m meeteval.wer merge hypA_cpwer_per_reco.json hypB_cpwer.json')

    # Write file (e.g. my_average[.yaml|.json]) instead of print.
    run(f'python -m meeteval.wer merge hypA_cpwer_per_reco.json hypB_cpwer.json --out my_average.json')


def test_burn_average():
    run(f'python -m meeteval.wer cpwer -h hypA.stm -r refA.stm')  # create hypA_cpwer_per_reco.json and hypA_cpwer.json
    run(f'python -m meeteval.wer cpwer -h hypB.stm -r refB.stm')  # create hypB_cpwer_per_reco.json and hypB_cpwer.json

    run(f'python -m meeteval.wer average hypA_cpwer_per_reco.json hypB_cpwer_per_reco.json')

    run(f'python -m meeteval.wer average hypA_cpwer_per_reco.json hypB_cpwer_per_reco.json')


def test_burn_siso():
    run(f'python -m meeteval.wer wer -h text_hyp -r text_ref')
    run(f'python -m meeteval.wer sisower -h text_hyp -r text_ref')
    run(f'python -m meeteval.wer wer -h hyp_siso.stm -r ref_siso.stm')
    run(f'python -m meeteval.wer wer -h text_hyp -r text_ref --normalizer "lower,rm(.?!,)"')


def test_viz_html():
    run(f'python -m meeteval.viz html -h hyp.stm -r ref.stm')
    run(f'python -m meeteval.viz html -h hyp.stm -r ref.stm --normalizer="lower,rm(.?!,)"')
    run(f'python -m meeteval.viz html -h hyp.stm -r ref.stm --alignment=tcp')
    run(f'python -m meeteval.viz html -h hyp.stm -r ref.stm --alignment=cp')
    run(f'python -m meeteval.viz html -h hyp.stm -r ref.stm --out=viz')
    run(f'python -m meeteval.viz html -h hyp.stm -r ref.stm --alignment cp tcorc')

    # Test loading a precomputed assignment
    run(f'python -m meeteval.wer cpwer -h hyp.stm -r ref.stm --per-reco-out hyp_cpwer_per_reco.json')
    run(f'python -m meeteval.wer tcorcwer -h hyp.stm -r ref.stm --per-reco-out hyp_tcorcwer_per_reco.json --collar 5')
    run(f'meeteval-viz html -h hyp.stm -r ref.stm --alignment cp tcorc --per-reco-file hyp_cpwer_per_reco.json hyp_tcorcwer_per_reco.json')

def test_viz_index_html(tmpdir):
    run(f'python -m meeteval.viz html -h hyp.stm -r ref.stm -o {tmpdir / "viz"}')
    run(f'python -m meeteval.viz index_html viz --out {tmpdir / "viz/index.html"}')
    run(f'python -m meeteval.viz index_html viz --out {tmpdir / "viz2"} --copy')
    run(f'python -m meeteval.viz index_html viz --out {tmpdir / "viz3.html"} --copy {tmpdir / "viz3"}')

def test_normalize(tmpdir):
    run(f'python -m meeteval.wer normalize hyp.stm -o - --normalizer="lower,rm(.?!,)"')
    run(f'python -m meeteval.wer normalize hyp.stm -o {tmpdir / "hyp_normalized.stm"} --normalizer="lower,rm([^a-z0-9 ])"')

    # Test that chaining normalizer and wer scripts is equal to using the normalizer option on the script
    chained = run('python -m meeteval.wer cpwer -r <(python -m meeteval.wer normalize ref.stm -o - --normalizer "lower,rm(.?!,)") -h <(python -m meeteval.wer normalize hyp.stm -o - --normalizer "lower,rm(.?!,)") --average-out - --per-reco-out -')
    option = run('python -m meeteval.wer cpwer -r ref.stm -h hyp.stm --average-out - --per-reco-out - --normalizer "lower,rm(.?!,)"')
    assert chained.stdout == option.stdout
    assert chained.stderr == option.stderr

def test_pipe_cli_commands():
    run('cat hyp.stm | python -m meeteval.wer normalize - -o - | python -m meeteval.io stm2seglst -o -')
