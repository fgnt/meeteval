import dataclasses
import functools
from meeteval.io import SegLST
from meeteval.wer.wer.error_rate import combine_error_rates
from meeteval.wer.wer.time_constrained import _time_constrained_siso_error_rate
from meeteval.wer.wer.mimo import MimoErrorRate, apply_mimo_assignment
from meeteval.wer.preprocess import preprocess


def time_constrained_mimo_word_error_rate(
        reference, 
        hypothesis,
        collar=0,
        reference_pseudo_word_level_timing='character_based',
        hypothesis_pseudo_word_level_timing='character_based_points',
        reference_sort='segment',
        hypothesis_sort='segment',
) -> 'MimoErrorRate':

    reference, hypothesis, ref_self_overlap, hyp_self_overlap = preprocess(
        reference, hypothesis,
        keep_keys=('words', 'segment_index', 'speaker', 'start_time', 'end_time'),
        reference_sort=reference_sort,
        hypothesis_sort=hypothesis_sort,
        reference_pseudo_word_level_timing=reference_pseudo_word_level_timing,
        hypothesis_pseudo_word_level_timing=hypothesis_pseudo_word_level_timing,
        segment_representation='word',
        segment_index='segment',
        remove_empty_segments=False,
        collar=collar,
    )
    error_rate = _tcmimower(reference, hypothesis)

    return dataclasses.replace(
        error_rate,
        reference_self_overlap=ref_self_overlap,
        hypothesis_self_overlap=hyp_self_overlap,
    )


def _cleanup_reference(reference: SegLST):
    """
    Remove empty segments and speakers from the reference and 
    track which speakers and segments got removed
    """
    empty_segments = reference.filter(lambda x: x['words'] == '')
    reference = reference.filter(lambda x: x['words'] != '')
    return reference, empty_segments


def _check_error_rate(reference: SegLST, hypothesis: SegLST, assignment, length, distance):
    reference, hypothesis = apply_mimo_assignment(assignment, reference, hypothesis)
    reference = reference.groupby('speaker')
    hypothesis = hypothesis.groupby('speaker')
    er = combine_error_rates([
        _time_constrained_siso_error_rate(
            reference.get(k, SegLST([])), 
            hypothesis.get(k, SegLST([])),
        )
        for k in set(hypothesis.keys()) | set(reference.keys())
    ])
    assert er.length == length, (length, er)
    assert er.errors == distance, (distance, er, assignment)
    return er

def _tcmimower_unchecked(reference: SegLST, hypothesis: SegLST, dummy_key: str):
    """
    The core function for the time-constrained MIMO-WER.

    This function does not check its input arguments and does not 
    handle edge cases. Specifically, it assumes:
    - reference and hypothesis are already preprocessed
    - Each reference segment contains exactly one word and has a segment_index assigned
    - reference and hypothesis are not empty
    - reference and hypothesis contain no empty segments
    - the number of streams in reference and hypothesis is small enough that the 
      assignment can be computed in reasonable time
    """
    reference_grouped = reference.groupby('speaker')
    hypothesis_grouped = hypothesis.groupby('speaker')

       # Safety check: The complexity explodes for large numbers of speakers
    if len(hypothesis_grouped) > 10:
        raise RuntimeError(
            f'Are you sure?\n'
            f'Found a total of {len(hypothesis_grouped)} speakers in the input.\n'
            f'This indicates a mistake in the input, or does your use-case '
            f'really require scoring with that many speakers?\n'
            f'See https://github.com/fgnt/meeteval/blob/main/doc/num_speaker_limits.md for details.'
        )
    if len(reference_grouped) > 10:
        raise RuntimeError(
            f'Are you sure?\n'
            f'Found a total of {len(reference_grouped)} speakers in the input.\n'
            f'This indicates a mistake in the input, or does your use-case '
            f'really require scoring with that many speakers?\n'
            f'See https://github.com/fgnt/meeteval/blob/main/doc/num_speaker_limits.md for details.'
        )

    # Compute MIMO distance
    from meeteval.wer.matching.cy_time_constrained_mimo_matching import time_constrained_mimo_levenshtein_distance

    distance, assignment = time_constrained_mimo_levenshtein_distance(
        [   # speakers
            [   # segments
                segment.T['words'] 
                for segment in speaker.groupby('segment_index').values()
            ] 
            for speaker in reference_grouped.values()
        ],
        [stream.T['words'] for stream in hypothesis_grouped.values()],
        [   # speakers
            [   # segments
                list(zip(segment.T['start_time'], segment.T['end_time'])) 
                for segment in speaker.groupby('segment_index').values()
            ] 
            for speaker in reference_grouped.values()
        ],
        [list(zip(stream.T['start_time'], stream.T['end_time'])) for stream in hypothesis_grouped.values()],
    )
    assert len(assignment) == len(reference.unique('segment_index')), (len(assignment), len(reference))

    # Translate the assignment from hypothesis index to stream id
    # Fill with a dummy stream if hypothesis or reference is empty
    reference_keys = list(reference_grouped.keys())
    hypothesis_keys = list(hypothesis_grouped.keys()) or [dummy_key]
    assignment = [(reference_keys[r], hypothesis_keys[h]) for r, h in assignment]

    # Consistency check
    er = _check_error_rate(
        reference, 
        hypothesis,
        assignment,
        length=sum([len(s['words']) if isinstance(s['words'], list) else 1 for s in reference]),
        distance=distance,
    )

    return er, assignment

def _tcmimower(reference, hypothesis):
    """
    The core function for the time-constrained MIMO-WER.

    """
    # Remove empty segments from the reference because they can increase
    # the computational complexity of the ORC algorithm, but they do not 
    # contribute to the distance

    reference_empty_segments = reference.filter(lambda x: x['words'] == '')
    reference_clean = reference.filter(lambda x: x['words'] != '')

    # hypothesis_empty_segments = reference.filter(lambda x: x['words'] == '')
    hypothesis_clean = hypothesis.filter(lambda x: x['words'] != '')

    # if len(reference_clean) == 0:
    #     raise NotImplementedError('Empty reference')
    # if len(hypothesis_clean) == 0:
    #     raise NotImplementedError('Empty hypothesis')

    er, assignment = _tcmimower_unchecked(reference_clean, hypothesis_clean, next(iter(hypothesis.unique('speaker'))))

    # Reconstruct the assignment with empty segments
    # and sort by original segment index
    if len(assignment) != len(reference.unique('segment_index')):
        dummy_key = sorted(hypothesis.unique('speaker'))[0]

        d = {
            speaker: [(i, len(segments) == 1 and segments[0]['words'] == '') for i, segments in v.groupby('segment_index').items()]
            for speaker, v in reference.groupby('speaker').items()
        }

        new_assignment = []
        while assignment:
            r = assignment[0][0]
            while d[r]:
                i, empty = d[r].pop(0)
                if empty:
                    new_assignment.append((r, dummy_key))
                else:
                    new_assignment.append(assignment.pop(0))
        
        for speaker, segments in d.items():
            for i, empty in segments:
                if empty:
                    new_assignment.append((speaker, dummy_key))
                else:
                    raise RuntimeError('Unassigned segment')

            # Pick the one with the smallest segment_index from:
            # - the 0-th entry in the assignment from reference_clean_grouped
            # - one from reference_empty_segments_grouped
            # missing_speaker, missing_indices = min(
            #     reference_empty_segments_grouped.items(), 
            #     key=lambda x: x[1][0]
            # ) if reference_empty_segments_grouped else (None, [None])
            # missing_index = missing_indices[0]

            # clean_index = reference_clean_grouped[assignment[0][0]][0] if assignment else None

            # if clean_index is None and missing_index is None:
            #     break

            # if (
            #     clean_index is not None and (missing_index is None or clean_index <= missing_index)
            #     or reference_clean_grouped.get(missing_speaker, [missing_index])[0] < missing_index
            # ):
            #     reference_clean_grouped[assignment[0][0]].pop(0)
            #     if not reference_clean_grouped[assignment[0][0]]:
            #         reference_clean_grouped.pop(assignment[0][0])
            #     new_assignment.append(assignment.pop(0))
            # else:
            #     new_assignment.append((missing_speaker, dummy_key))
            #     reference_empty_segments_grouped[missing_speaker].pop(0)
            #     if not reference_empty_segments_grouped[missing_speaker]:
            #         reference_empty_segments_grouped.pop(missing_speaker)
        assignment = new_assignment
        # for segment_index, segment in reference.groupby('segment_index').items():
        #     assert len(new_assignment) == segment_index, (len(new_assignment), segment_index)
        #     segment = segment[0]
        #     if assignment and assignment[0][0] == segment['speaker']:
        #         new_assignment.append(assignment.pop(0))
        #     else:
        #         new_assignment.append((segment['speaker'], dummy_key))
        # assert len(assignment) == 0, assignment
        # assignment = new_assignment

    assert len(assignment) == len(reference.unique('segment_index')), (len(assignment), len(reference.unique('segment_index')))

    return MimoErrorRate(
        er.errors, er.length,
        insertions=er.insertions,
        deletions=er.deletions,
        substitutions=er.substitutions,
        assignment=assignment,
        reference_self_overlap=None,
        hypothesis_self_overlap=None,
    )


def time_constrained_mimo_word_error_rate_multifile(
        reference,
        hypothesis,
        partial=False,
        collar=0,
        reference_sort='segment_if_available',
        hypothesis_sort='segment_if_available',
) -> 'dict[str, MimoErrorRate]':
    """
    Computes the time-constrained MIMO WER for each example in the reference 
    and hypothesis files.

    To compute the overall WER, use
    `sum(time_constrained_mimo_word_error_rate_multifile(r, h).values())`.
    """
    from meeteval.io.seglst import apply_multi_file
    return apply_multi_file(
        functools.partial(
            time_constrained_mimo_word_error_rate,
            collar=collar,
            reference_sort=reference_sort,
            hypothesis_sort=hypothesis_sort,
        ),
        reference, hypothesis, partial=partial
    )
