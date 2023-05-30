# Limits on numbers of speakers

Typical use-cases for the WER definitions provided with MeetEval have relatively small numbers of speakers.
We thus limit the number of speakers that can be provided to MeetEval so that usage errors are caught early.

We chose limits that catch problematic cases early but do not interfere with common use-cases.
If you have a valid use-case that exceeds these limits, please open an issue at https://github.com/fgnt/meeteval/issues/new.

| Algorithm | Maximum number of speakers |
|-----------|----------------------------|
| MIMO WER  | 10                         |
| ORC WER   | 10                         |
| cpWER     | 20                         |

## A note on complexity

Some matching algorithms implemented in MeetEval are computationally demanding when the number of speakers in the reference or hypothesis is large.
Especially, the complexity of MIMO WER and ORC WER is exponential in the number of speakers.
Using the matching algorithms with too many speakers can thus lead to out-of-memory errors or an extraordinary long runtime.
