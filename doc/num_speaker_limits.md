# Limits on numbers of speakers

Some matching algorithms implemented in MeetEval are computationally demanding when the number of speakers in the reference or hypothesis is large.
Especially, the complexity of MIMO WER and ORC WER is exponential in the number of speakers.
We thus set a hard limit to the number of speakers to catch cases where the algorithm would run for a vey long time or run out of memory before the computation is actually started.

## Maximum values for numbers of speakers

The maximum chosen for a specific algorithm is arbitrary; we tried to choose values that catch problematic cases early but do not interfere with common use-cases.

### MIMO WER and ORC WER

The maximum is set to 10, which is in most cases already not computable.

### cpWER

We don't set a strict maximum on the number of speakers here, but a maximum on the difference in number of speakers in the reference and hypothesis.
This is to make sure that the matching doesn't run out of fallback speaker labels.
