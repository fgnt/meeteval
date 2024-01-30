<h1 align="center">MeetEval</h1> 
<h3 align="center">A meeting transcription evaluation toolkit</h3>
<div align="center"><a href="#features">Features</a> | <a href="#installation">Installation</a> | <a href="#python-interface">Python Interface</a> | <a href="#command-line-interface">Command Line Interface</a> | <a href="#cite">Cite</a></div>
<a href="https://github.com/fgnt/meeteval/actions"><img src="https://github.com/fgnt/meeteval/actions/workflows/pytest.yml/badge.svg"/></a>


## Features
MeetEval supports the following metrics for meeting transcription evaluation:

- Standard WER for single utterances (Called SISO WER in MeetEval)
- Concatenated minimum-Permutation Word Error Rate (cpWER)
- Optimal Reference Combination Word Error Rate (ORC WER)
- Multi-speaker-input multi-stream-output Word Error Rate (MIMO WER)
- [Time-Constrained minimum-Permutation Word Error Rate (tcpWER)](/doc/tcpwer.md)
- Time-Constrained Optimal Reference Combination Word Error Rate (tcORC WER)

WIP: MeetEval supports visualization for cpWER and tcpWER

## Installation

You need to have [Cython](https://cython.org/) installed.

```shell
pip install cython
git clone https://github.com/fgnt/meeteval
pip install -e ./meeteval[cli]
```

The `[cli]` is optional, except when you want to use the command line
interface which uses `pyyaml`.

## Computing WERs

### Python interface

`MeetEval` provides a Python-based interface to compute WERs for pairs of reference and hypothesis.

```python
>>> from meeteval.wer import wer
>>> wer.siso_word_error_rate('The quick brown fox jumps over the lazy dog', 'The kwik browne focks jumps over the lay dock')
ErrorRate(errors=5, length=9, error_rate=0.5555555555555556)
>>> wer.orc_word_error_rate(['a b', 'c d', 'e'], ['a b e f', 'c d'])
OrcErrorRate(errors=1, length=5, error_rate=0.2, assignment=(0, 1, 0))
```

The results are wrapped in frozen `ErrorRate` objects.
This class bundles statistics (errors, total number of words) and potential auxiliary information (e.g., assignment for ORC WER) together with the WER.

To compute an "overall" WER over multiple examples, use the `combine_error_rates` function:

```python
>>> from meeteval.wer import wer
>>> wer1 = wer.siso_word_error_rate('The quick brown fox jumps over the lazy dog', 'The kwik browne focks jumps over the lay dock')
>>> wer1
ErrorRate(errors=5, length=9, error_rate=0.5555555555555556)
>>> wer2 = wer.siso_word_error_rate('Hello World', 'Goodbye')
>>> wer2
ErrorRate(errors=2, length=2, error_rate=1.0)
>>> wer.combine_error_rates(wer1, wer2)
ErrorRate(errors=7, length=11, error_rate=0.6363636363636364)
```

Note that the combined WER is _not_ the average over the error rates, but the error rate that results from combining the errors and lengths of all error rates.
`combine_error_rates` also discards any information that cannot be aggregated over multiple examples (such as the ORC WER assignment).

#### Aligning sequences

Sequences can be aligned, similar to `kaldialign.align`, using the tcpWER matching:
```python
>>> from meeteval.wer.wer.time_constrained import align
>>> align([{'words': 'a b', 'start_time': 0, 'end_time': 1}], [{'words': 'a c', 'start_time': 0, 'end_time': 1}, {'words': 'd', 'start_time': 2, 'end_time': 3}])
[('a', 'a'), ('b', 'c'), ('*', 'd')]
```

### Command-line interface

`MeetEval` supports [Segmental Time Mark](https://github.com/usnistgov/SCTK/blob/master/doc/infmts.htm#L75) (`STM`) files as input.
Each line in an `STM` file represents one "utterance" and is defined as

```STM
STM :== <filename> <channel> <speaker_id> <begin_time> <end_time> <transcript>
```
where
- `filename`: name of the recording
- `channel`: ignored by MeetEval
- `speaker_id`: ID of the speaker or system output stream/channel (not microphone channel)
- `begin_time`: in seconds, used to find the order of the utterances
- `end_time`: in seconds
- `transcript`: space-separated list of words

for example:
```
recording1 1 Alice 0 0 Hello Bob.
recording1 1 Bob 1 0 Hello Alice.
recording1 1 Alice 2 0 How are you?
...
recording2 1 Alice 0 0 Hello Carol.
...
```

An example `STM` file can be found in [the example_files](example_files/ref.stm).

We chose the `STM` format as the default because it contains all information required to compute the cpWER, ORC WER and MIMO WER.
Most metrics in `MeetEval` (all except tcpWER) currently do not support use of detailed timing information.
For those metrics, `begin_time` is only used to determine the correct utterance order and `end_time` is ignored.
The speaker-ID field in the hypothesis encodes the output channel for MIMO and ORC WER.
`MeetEval` does not support alternate transcripts (e.g., `"i've { um / uh / @ } as far as i'm concerned"`).

Once you created an `STM` file, the tool can be called like this:

```shell
python -m meeteval.wer [orcwer|mimower|cpwer|tcpwer] -h example_files/hyp.stm -r example_files/ref.stm
# or
meeteval-wer [orcwer|mimower|cpwer|tcpwer] -h example_files/hyp.stm -r example_files/ref.stm
```

The command `orcwer`, `mimower`, `cpwer` and `tcpwer` selects the WER definition to use.
By default, the hypothesis files is used to create the template for the average
(e.g. `hypothesis.json`) and per_reco `hypothesis_per_reco.json` file.
They can be changed with `--average-out` and `--per-reco-out`.
`.json` and `.yaml` are the supported suffixes.

More examples can be found in [tests/test_cli.py](tests/test_cli.py).

The tool also supports [time marked conversation input  files](https://github.com/usnistgov/SCTK/blob/f48376a203ab17f0d479995d87275db6772dcb4a/doc/infmts.htm#L285) (`CTM`)

```CTM
CTM :== <filename> <channel> <begin_time> <duration> <word> [<confidence>]
```

for the hypothesis (one file per speaker).
The time marks in the `CTM` file are only used to find the order of words.
Detailed timing information is not used.
You have to supply one `CTM` file for each system output channel using multiple `-h` arguments since `CTM` files don't encode speaker or system output channel information (the `channel` field has a different meaning: microphone).
For example:

```shell
meeteval-wer orcwer -h hyp1.ctm -h hyp2.ctm -r reference.stm
```

Note that the `LibriCSS` baseline recipe produces one `CTM` file which merges the speakers, so that it cannot be applied straight away. We recommend to use `STM` files.

## Visualization [WIP]

Preview: https://groups.uni-paderborn.de/nt/meeteval/viz.html

```python
import meeteval
from meeteval.viz.visualize import AlignmentVisualization

folder = r'https://raw.githubusercontent.com/fgnt/meeteval/main/'
av = AlignmentVisualization(
    meeteval.io.load(folder + 'example_files/ref.stm').groupby('filename')['recordingA'],
    meeteval.io.load(folder + 'example_files/hyp.stm').groupby('filename')['recordingA']
)
# display(av)  # Jupyter
# av.dump('viz.html')  # Create standalone HTML file
```


## Cite

The toolkit and the tcpWER were presented at the CHiME-2023 workshop (Computational Hearing in Multisource Environments) with the paper 
["MeetEval: A Toolkit for Computation of Word Error Rates for Meeting Transcription Systems"](https://arxiv.org/abs/2307.11394).

```bibtex
@InProceedings{MeetEval23,
  title={MeetEval: A Toolkit for Computation of Word Error Rates for Meeting Transcription Systems},
  author={von Neumann, Thilo and Boeddeker, Christoph and Delcroix, Marc and Haeb-Umbach, Reinhold},
  booktitle={CHiME-2023 Workshop, Dublin, England},
  year={2023}
}
```

The MIMO WER and efficient implementation of ORC WER are presented in the paper 
["On Word Error Rate Definitions and their Efficient Computation for Multi-Speaker Speech Recognition Systems"](https://ieeexplore.ieee.org/iel7/10094559/10094560/10094784.pdf).

```bibtex
@InProceedings{MIMO23,
  author       = {von Neumann, Thilo and Boeddeker, Christoph and Kinoshita, Keisuke and Delcroix, Marc and Haeb-Umbach, Reinhold},
  title        = {On Word Error Rate Definitions and their Efficient Computation for Multi-Speaker Speech Recognition Systems},
  booktitle    = {ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages        = {1--5},
  year         = {2023},
  organization = {IEEE}
}
```
