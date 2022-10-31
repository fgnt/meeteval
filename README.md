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
pip install -e ./meeteval
```

## Computing WERs

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
meeteval-wer -h hypothesis.stm -r reference.stm --orc --mimo --cp
```

The switches `--orc`, `--mimo` and `--cp` allow selecting the WER definition to use.
Multiple definitions can be selected simultaneously.

## Cite

The MIMO WER and efficient implementation of ORC WER are presented in the paper "On Word Error Rate Definitions and
their Efficient Computation for Multi-Speaker Speech Recognition Systems".

```bibtex
TODO!
```
