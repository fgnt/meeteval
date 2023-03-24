# Time-Constrained minimum Permutation Word Error Rate

The Time-Constrained minimum Permutation Word Error Rate (tcpWER) is similar to the cpWER, but uses temporal information to prevent matching words that are far apart temporally.
The temporal constraint idea is similar (but not identical) to aslicte's `-time-prune` option.

## Goals of the tcpWER
The system should be forced to provide rough temporal annotations (diarization) and should be penalized when its results become implausible compared to the reference. 
This leads us to following properties:

- The system should group segments that it thinks belong to the same speaker together (similar to cpWER)
- The system should be penalized when it produces long segments by merging multiple reference segments
- It should not be penalized when the system provides more precise timing than the reference (e.g., by splitting in a pause)

## Pseudo-word-level annotations
To compute the matching, we need a temporal annotation (start and end time) for each word.
Often, detailed word-level temporal annotations are not available, either because annotating a reference is expensive or a system does not produce such detailed information.
We thus implement different strategies to infer "pseudo-word-level" timings from segment-level timings:

- `full_segment`: Copy the time annotation from the segment to every word within that segment
- `equidistant_intervals`: Divides the segment-level interval into number-of-words many equally sized intervals
- `euqidistant_points`: Places words as time-points (zero-length intervals) equally spaced in the segment

## Collar
We include a collar option both for the reference (`--ref-collar`) and the hypothesis (`--hyp-collar`) annotations.
It specifies by how much the system's (and pseudo-word-level annotation strategy's) prediction can differ from the ground truth annotation before it is counted as an error.
Due to the way we estimate pseudo-word-level annotations for the segment-level annotations, the collar has to be relatively large (compared to typical values for DER computation).
It should be chosen so that small diarization errors (e.g., merging two utterances of the same speaker uttered without a pause into a single segment) are not penalized but larger errors (merging utterances that are tens of seconds apart) is penalized.
This, of course, depends on the data, but we found values in the range of 2-5s to work well in some test on libri-CSS.

## Using tcpWER

The tcpWER currently only support STM files because they provide all necessary information on a segment level.
You can use any resolution for the begin and end times (e.g., seconds or samples), but make sure to adjust the collar accordingly (`5` or `80000` for 16kHz).
```shell
meeteval-wer tcpwer -h hyp.stm -r ref.stm --hyp-collar 5
```
