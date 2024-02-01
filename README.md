<h1 align="center">MeetEval</h1> 
<h3 align="center">A meeting transcription evaluation toolkit</h3>
<div align="center"><a href="#features">Features</a> | <a href="#installation">Installation</a> | <a href="#python-interface">Python Interface</a> | <a href="#command-line-interface">Command Line Interface</a> | <a href="#visualization-wip">Visualization (WIP)</a> | <a href="#cite">Cite</a></div>
<br>
<a href="https://github.com/fgnt/meeteval/actions"><img src="https://github.com/fgnt/meeteval/actions/workflows/pytest.yml/badge.svg"/></a>


## Features
MeetEval supports the following metrics for meeting transcription evaluation:

- **Standard WER** for single utterances (Called SISO WER in MeetEval)<br>
  `meeteval-wer wer -r ref -h hyp`
- **Concatenated minimum-Permutation Word Error Rate (cpWER)**<br>
  `meeteval-wer cpwer -r ref.stm -h hyp.stm`
- **Optimal Reference Combination Word Error Rate (ORC WER)**<br>
  `meeteval-wer orcwer -r ref.stm -h hyp.stm`
- **Multi-speaker-input multi-stream-output Word Error Rate (MIMO WER)**<br>
  `meeteval-wer mimower -r ref.stm -h hyp.stm`
- **Time-Constrained minimum-Permutation Word Error Rate (tcpWER)**<br>
  `meeteval-wer tcpwer -r ref.stm -h hyp.stm --collar 5`
- **Time-Constrained Optimal Reference Combination Word Error Rate (tcORC WER)**<br>
  `meeteval-wer tcorcwer -r ref.stm -h hyp.stm --collar 5`
- **Diarization Error Rate (DER)** by wrapping [mdeval](https://github.com/nryant/dscore/raw/master/scorelib/md-eval-22.pl)<br>
  `meeteval-der md_eval_22 -r ref.stm -h hyp.stm --collar .25`

Additionally (WIP), MeetEval contains a [visualization](#visualization-wip) tool for cpWER and tcpWER alignments that help spot errors in system outputs.

## Installation

### From PyPI

```shell
pip install meeteval
```

### From source

```shell
git clone https://github.com/fgnt/meeteval
pip install -e ./meeteval
```

## Command-line interface

`MeetEval` supports the following file formats as input:
 - [Segmental Time Mark](https://github.com/usnistgov/SCTK/blob/master/doc/infmts.htm#L75) (`STM`)
 - [Time Marked Conversation](https://github.com/usnistgov/SCTK/blob/master/doc/infmts.htm#L286) (`CTM`)
 - [SEGment-wise Long-form Speech Transcription annotation](#segment-wise-long-form-speech-transcription-annotation-seglst) (`SegLST`), the file format used in the [CHiME challenges](https://www.chimechallenge.org)
 - [Rich Transcription Time Marked](https://github.com/nryant/dscore?tab=readme-ov-file#rttm) (`RTTM`) files (only for Diarizaiton Error Rate)


> [!NOTE]
> `MeetEval` does not support alternate transcripts (e.g., `"i've { um / uh / @ } as far as i'm concerned"`).

The command-line interface is available as `meeteval-wer` or `python -m meeteval.wer` with the following signature:

```shell
python -m meeteval.wer [orcwer|mimower|cpwer|tcpwer|tcorcwer] -h example_files/hyp.stm -r example_files/ref.stm
# or
meeteval-wer [orcwer|mimower|cpwer|tcpwer|tcorcwer] -h example_files/hyp.stm -r example_files/ref.stm
```

You can add `--help` to any command to get more information about the available options.
The command name `orcwer`, `mimower`, `cpwer` and `tcpwer` selects the metric to use.
By default, the hypothesis files is used to create the template for the average
(e.g. `hypothesis.json`) and per_reco `hypothesis_per_reco.json` file.
They can be changed with `--average-out` and `--per-reco-out`.
`.json` and `.yaml` are the supported suffixes.

More examples can be found in [tests/test_cli.py](tests/test_cli.py).

### File Formats

#### SEGment-wise Long-form Speech Transcription annotation (SegLST)

The SegLST format was used in the [CHiME-7 challenge](https://www.chimechallenge.org/challenges/chime7/task1/index) and is the default format for `MeetEval`.

The SegLST format is stored in JSON format and contains a list of segments.
Each segment should have a minimum set of keys `"session_id"` and `"words"`.
Depending on the metric, additional keys may be required (`"speaker"`, `"start_time"`, `"end_time"`).

An example is shown below:
```py
[
    {
        "session_id": "recordingA", # Required
        "words": "The quick brown fox jumps over the lazy dog", # Required for WER metrics
        "speaker": "Alice", # Required for metrics that use speaker information (cpWER, ORC WER, MIMO WER)
        "start_time": 0,    # Required for time-constrained metrics (tcpWER, tcORC-WER, DER, ...)
        "end_time": 1,      # Required for time-constrained metrics (tcpWER, tcORC-WER, DER, ...)
        "audio_path": "path/to/recordingA.wav"  # Any additional keys can be included
    },
    ...
]
```
Another example can be found [here](example_files/hyp.seglst.json).

#### [Segmental Time Mark (STM)](https://github.com/usnistgov/SCTK/blob/master/doc/infmts.htm#L75)
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

An example `STM` file can be found in [here](example_files/ref.stm).

#### [Time Marked Conversation (CTM)](https://github.com/usnistgov/SCTK/blob/master/doc/infmts.htm#L286)
The CTM format is defined as

```CTM
CTM :== <filename> <channel> <begin_time> <duration> <word> [<confidence>]
```

for the hypothesis (one file per speaker).
You have to supply one `CTM` file for each system output channel using multiple `-h` arguments since `CTM` files don't encode speaker or system output channel information (the `channel` field has a different meaning: left or right microphone).
For example:

```shell
meeteval-wer orcwer -h hyp1.ctm -h hyp2.ctm -r reference.stm
```

> [!NOTE]
> Note that the `LibriCSS` baseline recipe produces one `CTM` file which merges the speakers, so that it cannot be applied straight away. We recommend to use `STM` or `SegLST` files.


## Python interface

For all metrics a [Low-level](#low-level-interface) and [high-level](#high-level-interface) interface is available.

> [!TIP]
> You want to use the [high-level](#high-level-interface) for computing metrics over a full dataset. <br>
> You want to use the [low-level](#low-level-interface) interface for computing metrics for single examples or when your data is represented as Python structures, e.g., nested lists of strings.

### Low-level interface

All WERs have a low-level interface in the `meeteval.wer` module that allows computing the WER for single examples.
The functions take the reference and hypothesis as input and return an `ErrorRate` object.
The `ErrorRate` bundles statistics (errors, total number of words) and potential auxiliary information (e.g., assignment for ORC WER) together with the WER.

```python
import meeteval

# SISO WER
wer = meeteval.wer.wer.siso.siso_word_error_rate(
    reference='The quick brown fox jumps over the lazy dog',
    hypothesis='The kwick brown fox jump over lazy '
)
print(wer)
# ErrorRate(error_rate=0.4444444444444444, errors=4, length=9, insertions=0, deletions=2, substitutions=2)

# cpWER
wer = meeteval.wer.wer.cp.cp_word_error_rate(
    reference=['The quick brown fox', 'jumps over the lazy dog'],
    hypothesis=['The kwick brown fox', 'jump over lazy ']
)
print(wer)
# CPErrorRate(error_rate=0.4444444444444444, errors=4, length=9, insertions=0, deletions=2, substitutions=2, missed_speaker=0, falarm_speaker=0, scored_speaker=2, assignment=((0, 0), (1, 1)))

# ORC-WER
wer = meeteval.wer.wer.orc.orc_word_error_rate(
    reference=['The quick brown fox', 'jumps over the lazy dog'],
    hypothesis=['The kwick brown fox', 'jump over lazy ']
)
print(wer)
# OrcErrorRate(error_rate=0.4444444444444444, errors=4, length=9, insertions=0, deletions=2, substitutions=2, assignment=(0, 1))
```

The input format can be a (list of) strings or an object representing a file format from `meeteval.io`:

```python
import meeteval
wer = meeteval.wer.wer.cp.cp_word_error_rate(
    reference = meeteval.io.STM.parse('recordingA 1 Alice 0 1 The quick brown fox jumps over the lazy dog'),
    hypothesis = meeteval.io.STM.parse('recordingA 1 spk-1 0 1 The kwick brown fox jump over lazy ')
)
print(wer)
# CPErrorRate(error_rate=0.4444444444444444, errors=4, length=9, insertions=0, deletions=2, substitutions=2, missed_speaker=0, falarm_speaker=0, scored_speaker=1, assignment=(('Alice', 'spk-1'),))
```

All low-level interfaces come with a single-example function (as show above) and a batch function that computes the WER for multiple examples at once.
The batch function is postfixed with `_multifile` and is similar to the high-level interface without fancy input format handling.
To compute the average over multiple `ErrorRate`s, use `meeteval.wer.combine_error_rates`.
Note that the combined WER is _not_ the average over the error rates, but the error rate that results from combining the errors and lengths of all error rates.
`combine_error_rates` also discards any information that cannot be aggregated over multiple examples (such as the ORC WER assignment).

For example with the cpWER:
```python
import meeteval.wer.wer.siso

wers = meeteval.wer.wer.cp.cp_word_error_rate_multifile(
    reference={
        'recordingA': {'speakerA': 'First example', 'speakerB': 'First example second speaker'}, 
        'recordingB': {'speakerA': 'Second example'},
    },
    hypothesis={
        'recordingA': ['First example with errors', 'First example second speaker'],
        'recordingB': ['Second example', 'Overestimated speaker'],
    }
)
print(wers)
# {
#   'recordingA': CPErrorRate(error_rate=0.3333333333333333, errors=2, length=6, insertions=2, deletions=0, substitutions=0, missed_speaker=0, falarm_speaker=0, scored_speaker=2, assignment=(('speakerA', 0), ('speakerB', 1))), 
#   'recordingB': CPErrorRate(error_rate=1.0, errors=2, length=2, insertions=2, deletions=0, substitutions=0, missed_speaker=0, falarm_speaker=1, scored_speaker=1, assignment=(('speakerA', 0), (None, 1)))
# }

# Use combine_error_rates to compute an "overall" WER over multiple examples
avg = meeteval.wer.combine_error_rates(wers)
print(avg)
# CPErrorRate(error_rate=0.5, errors=4, length=8, insertions=4, deletions=0, substitutions=0, missed_speaker=0, falarm_speaker=1, scored_speaker=3)
```

### High-level interface

All WERs have a high-level Python interface available directly in the `meeteval.wer` module that mirrors the [Command-line interface](#command-line-interface) and accepts the formats from `meeteval.io` as input.
All of these functions require the input format to contain a session-ID and output a dict mapping from session-ID to the result of that session

```python
import meeteval

# File Paths
wers = meeteval.wer.tcpwer('example_files/ref.stm', 'example_files/hyp.stm', collar=5)

# Loaded files
wers = meeteval.wer.tcpwer(meeteval.io.load('example_files/ref.stm'), meeteval.io.load('example_files/hyp.stm'), collar=5)

# Objects
wers = meeteval.wer.tcpwer(
    reference=meeteval.io.STM.parse('''
        recordingA 1 Alice 0 1 The quick brown fox jumps over the lazy dog
        recordingB 1 Bob 0 1 The quick brown fox jumps over the lazy dog
    '''),
    hypothesis=meeteval.io.STM.parse('''
        recordingA 1 spk-1 0 1 The kwick brown fox jump over lazy
        recordingB 1 spk-1 0 1 The kwick brown fox jump over lazy
    '''),
    collar=5,
)
print(wers)
# {
#     'recordingA': CPErrorRate(error_rate=0.4444444444444444, errors=4, length=9, insertions=0, deletions=2, substitutions=2, reference_self_overlap=SelfOverlap(overlap_rate=Decimal('0'), overlap_time=0, total_time=Decimal('1')), hypothesis_self_overlap=SelfOverlap(overlap_rate=Decimal('0'), overlap_time=0, total_time=Decimal('1')), missed_speaker=0, falarm_speaker=0, scored_speaker=1, assignment=(('Alice', 'spk-1'),)), 
#     'recordingB': CPErrorRate(error_rate=0.4444444444444444, errors=4, length=9, insertions=0, deletions=2, substitutions=2, reference_self_overlap=SelfOverlap(overlap_rate=Decimal('0'), overlap_time=0, total_time=Decimal('1')), hypothesis_self_overlap=SelfOverlap(overlap_rate=Decimal('0'), overlap_time=0, total_time=Decimal('1')), missed_speaker=0, falarm_speaker=0, scored_speaker=1, assignment=(('Bob', 'spk-1'),))
# }

avg = meeteval.wer.combine_error_rates(wers)
print(avg)
# CPErrorRate(error_rate=0.4444444444444444, errors=8, length=18, insertions=0, deletions=4, substitutions=4, reference_self_overlap=SelfOverlap(overlap_rate=Decimal('0'), overlap_time=0, total_time=Decimal('2')), hypothesis_self_overlap=SelfOverlap(overlap_rate=Decimal('0'), overlap_time=0, total_time=Decimal('2')), missed_speaker=0, falarm_speaker=0, scored_speaker=2)
```

### Aligning sequences

Sequences can be aligned, similar to `kaldialign.align`, using the tcpWER matching:
```python
import meeteval
meeteval.wer.wer.time_constrained.align([{'words': 'a b', 'start_time': 0, 'end_time': 1}], [{'words': 'a c', 'start_time': 0, 'end_time': 1}, {'words': 'd', 'start_time': 2, 'end_time': 3}])
# [('a', 'a'), ('b', 'c'), ('*', 'd')]
```

## Visualization [WIP]

> [!IMPORTANT]
> The visualization is under development! <br>
> Preview: https://groups.uni-paderborn.de/nt/meeteval/viz.html<br>
> Interactive notebook: https://fgnt.github.io/meeteval_jupyterlite/lab?path=Demo.ipynb

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
