<h1 align="center">MeetEval</h1> 
<h3 align="center">A meeting transcription evaluation toolkit</h3>

<p align="center">:warning: This repository is under construction! :warning:</p>

<a href="https://github.com/fgnt/meeteval/actions"><img src="https://github.com/fgnt/meeteval/actions/workflows/pytest.yml/badge.svg"/></a>

## Installation

### Binaries

TODO!

### Development

You need to have [Cython](https://cython.org/) installed.
Then:

```shell
pip install cython
git clone git@github.com:fgnt/meeteval.git
pip install -e ./meeteval[cli]
```

The `[cli]` is optional, except when you want to use the command line
interface, that uses `click`.

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
>>> form meeteval.wer import wer
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

### Command-line interface

`MeetEval` supports [Segmental Time Mark](https://github.com/usnistgov/SCTK/blob/master/doc/infmts.htm#L75) (`STM`) files as input.
Each line in an `STM` file represents one "utterance" and is defined as

```STM
STM :== <filename> <channel> <speaker_id> <begin_time> <end_time> <transcript>
```
e.g.
```
recording1 0 Alice 0 0 Hello Bob.
recording1 0 Bob 1 0 Hello Alice.
recording1 0 Alice 2 0 How are you?
...
recording2 0 Alice 0 0 Hello Carol.
...
```
where
- `filename`: name of the recording
- `channel`: ignored
- `speaker_id`: ID of the speaker or system output stream/channel (not microphone channel)
- `begin_time`: in seconds, used to find the order of the utterances (can also be an int counter)
- `end_time`: in seconds (currently ignored)
- `transcript`: space-separated list of words

An example `STM` file can be found in [the example_files](example_files/ref.stm).

We chose the `STM` format as the default because it contains all information required to compute the cpWER, ORC WER and MIMO WER.
`MeetEval` currently does not support use of detailed timing information, so `begin_time` is only used to determine the correct utterance order and `end_time` is ignored.
This may change in future versions.
The speaker-ID field in the hypothesis encodes the output channel for MIMO and ORC WER.
Once you created an `STM` file, the tool can be called like this:

```shell
python -m meeteval.wer [orcwer|mimower|cpwer] -h hyp.stm -r ref.stm
# or
meeteval-wer [orcwer|mimower|cpwer] -h hypothesis.stm -r reference.stm
```

The command `orcwer`, `mimower` and `cpwer` allow selecting the WER definition to use.
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
meeteval-wer -h hyp1.ctm -h hyp2.ctm -r reference.stm --orc
```

Note that the `LibriCSS` baseline recipe produces one `CTM` file which merges the speakers, so that it cannot be applied straight away. We recommend to use `STM` files.

## Cite

The MIMO WER and efficient implementation of ORC WER are presented in the paper "On Word Error Rate Definitions and
their Efficient Computation for Multi-Speaker Speech Recognition Systems".

```bibtex
TODO!
```
