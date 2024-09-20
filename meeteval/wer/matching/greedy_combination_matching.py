import itertools
import functools
from typing import List, Iterable, Tuple

import numpy as np

from meeteval.io.seglst import SegLST
from meeteval.wer.matching.cy_greedy_combination_matching import cy_forward_col, cy_forward_col_time_constrained


def _apply_assignment(assignment, segments, n=None):
    if n is None:
        n = max(assignment) + 1
    else:
        n = max(max(assignment) + 1, n)
    h_ = [[] for _ in range(n)]
    for a, h in zip(assignment, segments):
        h_[a].append(h)
    return h_


def _greedy_correct_assignment(
        segments: List[np.ndarray],
        streams: List[np.ndarray],
        assignment: List[int],
        forward_column: callable = cy_forward_col,
):
    """
    Greedily modify the assignment (of the segments to the streams)
    to decrease the Levenshtein distance.

    Args:
        segments: A list of segments for which stream labels should be obtained
        streams: A list of streams to which the segments are assigned
        assignment: The initial assignment.
        forward_column: A function to compute a column update. Can be used
            to switch between different cost functions, e.g. with different
            substitution costs or with or without time constraint.

    >>> segments = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint)
    >>> streams = np.asarray([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.uint)
    >>> _greedy_correct_assignment(segments, streams, [0, 1, 2])
    ([0, 1, 2], 0)
    >>> _greedy_correct_assignment(segments, streams, [1, 0, 0])
    ([0, 1, 2], 0)
    >>> _greedy_correct_assignment(segments, streams[:2], [1, 0, 0])
    ([0, 1, 0], 3)
    """
    # Temporary variables
    cost_differences = np.zeros((len(streams),), dtype=int)
    costs_with_utterance = np.zeros((len(streams),), dtype=int)
    costs_without_utterance = np.zeros((len(streams),), dtype=int)
    updated_forward_columns = [None] * len(streams)

    # Iteratively update assignment greedy for at most 50 iterations.
    # The algorithm typically converges after a few iterations. The number 50
    # is chosen for safety.
    for num_updates in range(1, 50):
        assert all(a < len(streams) for a in assignment), (assignment, len(streams), num_updates)

        # Compute backward matrices for every stream_index
        def compute_backward_columns(r, h):
            backward_column = np.arange(len(h) + 1, dtype=np.uint)
            backward_columns = [backward_column[::-1]]

            for r_ in r[::-1]:
                backward_column = forward_column(backward_column, h[::-1], r_[::-1])
                backward_columns.insert(0, backward_column[::-1])
            return backward_columns

        backward_columns = [
            compute_backward_columns(r, h)
            for r, h in zip(_apply_assignment(assignment, segments, n=len(streams)), streams)
        ]

        # Keep track of the cost on the current stream_index so that we don't have to compute the min in every
        # iteration
        stream_costs = [bm[0][0] for bm in backward_columns]
        old_stream_costs = stream_costs.copy()
        backward_indices = [0] * len(streams)

        # forward_columns holds the active columns of the forward matrix
        forward_columns = [np.arange(len(h) + 1, dtype=np.uint) for h in streams]

        new_assignment = []
        for utterance_index, (current_assignment, current_segment) in enumerate(zip(assignment, segments)):
            for stream_index, (
                    current_forward_column, stream, current_backward_columns, current_backward_index
            ) in enumerate(zip(forward_columns, streams, backward_columns, backward_indices)):
                # Forward column for adding stream to stream_index `stream_index`. We always have to compute this update,
                # even if we assign stream to stream_index `stream_index` because we might need this column in a later
                # iteration
                updated_column = forward_column(
                    current_forward_column, stream, current_segment,
                )

                if stream_index == current_assignment:
                    # Keep current assignment
                    # Cost with is the current cost since the utterance is not moved
                    costs_with_utterance[stream_index] = stream_costs[stream_index]

                    # The cost of `stream_index` without the current utterance can be computed from the current forward column
                    # and the backward column after the current utterance
                    backward_column = current_backward_columns[current_backward_index + 1]
                    costs_without_utterance[stream_index] = int(np.min(current_forward_column + backward_column))

                    # The assertion is relatively expensive because it is called in the hot loop, thus disabled
                    # assert np.min(current_forward_column + backward_column) == np.min(column + current_backward_columns[:, bm_index])
                else:
                    # Switch assignment: remove from `a` and add to `stream_index`
                    # The cost of `stream_index` without `stream` is the current stream_index cost because it is moved to this stream_index
                    costs_without_utterance[stream_index] = stream_costs[stream_index]

                    # The cost of `stream_index` with `stream` is computed from the current backward column and the updated
                    # forward column
                    backward_column = current_backward_columns[current_backward_index]
                    costs_with_utterance[stream_index] = int(np.min(updated_column + backward_column))

                # The final assignment is found by minimizing the cost difference that results from moving the
                # utterance to another stream_index. We track the different costs here to save computation for the
                # later update
                cost_differences[stream_index] = costs_with_utterance[stream_index] - costs_without_utterance[
                    stream_index]
                updated_forward_columns[stream_index] = updated_column

            # Find best stream_index for `stream` and update running variables
            best = int(np.argmin(cost_differences))

            # If best != current_assignment but the cost doesn't change, then keep the assignment.
            # This makes sure that an optimal assignment is not changed
            if cost_differences[best] == cost_differences[current_assignment]:
                best = current_assignment

            forward_columns[best] = updated_forward_columns[best]
            backward_indices[current_assignment] += 1
            new_assignment.append(best)

            # Update the cached stream_index costs. This is only required when we change the assignment
            # for the current segment
            if best != current_assignment:
                stream_costs[best] = costs_with_utterance[best]
                stream_costs[current_assignment] = costs_without_utterance[current_assignment]

        # Break when the assignment didn't change
        if any(a != na for a, na in zip(assignment, new_assignment)):
            assignment = new_assignment
        else:
            break

        # We know that the cost cannot increase, so check that we do correct
        assert sum(old_stream_costs) >= sum(stream_costs), (old_stream_costs, stream_costs)

    return assignment, int(np.sum(stream_costs))


def initialize_assignment(
        segments: SegLST,
        streams: SegLST,
        initialization: str = 'cp',
):
    """
    Args:
        segments: A SegLST object containing the segments that are assigned to
            the streams
        streams: A SegLST object containing the streams.
        initialization: How the initial stream labels are obtained before the
            greedy algorithm starts. Can be one of:
            - `'random'`: Assigns random initial stream labels to each segment
            - `'constant'`: Assigns a constant stream label of 0 to each segment
            - `'cp'`: Uses the cpWER solver to obtain the initial stream labels. Assumes
                that the `'speaker'` key is present in both `a` and `b`.

    >>> segments = SegLST([{'words': 'a', 'speaker': 'A', 'segment_index': 0}, {'words': 'b', 'speaker': 'A', 'segment_index': 0}, {'words': 'c', 'speaker': 'B', 'segment_index': 1}])
    >>> streams = SegLST([{'words': 'c', 'speaker': 'A'}, {'words': 'a b', 'speaker': 'B'}])
    >>> initialize_assignment(segments, streams, initialization='cp')
    [1, 0]
    >>> initialize_assignment(segments, streams, initialization='constant')
    [0, 0]
    >>> segments = SegLST([{'words': 'a', 'speaker': 'A', 'segment_index': 0}, {'words': 'b', 'speaker': 'C', 'segment_index': 0}, {'words': 'c', 'speaker': 'B', 'segment_index': 1}])
    >>> initialize_assignment(segments, streams, initialization='cp')
    [0, 0]
    """
    if initialization in ('cp', 'tcp'):
        # Special case when no streams are present
        if len(streams) == 0:
            return initialize_assignment(segments, streams, 'constant')

        # Special case when no segments are present
        if len(segments) == 0:
            return []

        # Compute cpWER to get a good starting point
        from meeteval.wer.wer.cp import _minimum_permutation_assignment
        from meeteval.wer.wer.siso import siso_levenshtein_distance
        from meeteval.wer.wer.time_constrained import time_constrained_siso_levenshtein_distance
        if isinstance(streams, SegLST):
            streams = streams.groupby('speaker')
        speaker_grouped_segments = segments.groupby('speaker')
        assignment, _, cost_matrix = _minimum_permutation_assignment(
            speaker_grouped_segments,
            streams,
            distance_fn=siso_levenshtein_distance if initialization == 'cp' else time_constrained_siso_levenshtein_distance,
        )
        # Use integers for the assignment labels.
        counter = iter(itertools.count())
        int_stream_labels = {k: next(counter) for k in streams.keys()}
        counter = iter(itertools.count())
        int_segment_labels = {k: next(counter) for k in speaker_grouped_segments.keys()}
        # Map unmatched segments to the stream that produced the lowest distance.
        # The distances are taken from the cost_matrix, where only the first
        # `len(streams)` entries are used. The remaining entries represent
        # matching against an empty (dummy) stream.
        assignment = {
            segment_label: int_stream_labels[stream_label]
            if stream_label in int_stream_labels
            else int(np.argmin(cost_matrix[int_segment_labels[segment_label], :len(streams)]))
            for i, (segment_label, stream_label) in enumerate(assignment)
        }
        # Translate assignment to segments
        assignment = [
            assignment[k[0]['speaker']]
            for k in segments.groupby('segment_index').values()
        ]
    elif initialization == 'random':
        assignment = np.random.choice(len(streams), len(set(segments.T['segment_index'])))
    elif initialization == 'constant':
        assignment = [0] * len(set(segments.T['segment_index']))
    else:
        raise ValueError(f'Unknown initialization: {initialization}')
    return assignment


def greedy_combination_matching(
        segments: List[Iterable[int]],
        streams: List[Iterable[int]],
        initial_assignment: List[int],
        *,
        distancetype: str = '21',   # '21', '2', '1'
):
    """
    Segments in `segments` are assigned to streams in `streams`.

    Args:
        segments: A list of segments for which stream labels should be obtained
        streams: A list of streams to which the segments are assigned
        initial_assignment: The initial assignment of the segments to the streams.
            Can be obtained with `initialize_assignment`.
        distancetype: The type of distance to use. Can be one of:
            - `'1'`: Use insertion cost of 1 (like in Levenshtein distance)
            - `'2'`: Use insertion cost of 2 (cost of insertion + deletion)
            - `'21'`: Start with '2' until converged and then use '1' until converged
    """
    if len(segments) == 0:
        return sum([len(s) for s in streams]), []
    if len(streams) == 0:
        return sum([len(s) for s in segments]), [0] * len(segments)

    segments = [np.asarray(s, dtype=np.uint) for s in segments]
    streams = [np.asarray(s, dtype=np.uint) for s in streams]

    assert len(initial_assignment) == len(segments), (len(initial_assignment), len(segments), initial_assignment)

    # Correct assignment
    assignment = initial_assignment
    assert distancetype in ('1', '2', '21'), distancetype
    for d in distancetype:
        assignment, distance = _greedy_correct_assignment(
            segments, streams, assignment,
            functools.partial(cy_forward_col, cost_substitution=int(d))
        )

    return distance, assignment


def greedy_time_constrained_combination_matching(
        segments: List[Iterable[Tuple[int, float, float]]],
        streams: List[Iterable[Tuple[int, float, float]]],
        initial_assignment: List[int],
        *,
        distancetype: str = '21',   # '21', '2', '1'
):
    """
    Segments in `segments` are assigned to streams in `streams`.

    Args:
        segments: A list of segments for which stream labels should be obtained
        streams: A list of streams to which the segments are assigned
        initial_assignment: The initial assignment of the segments to the streams.
            Can be obtained with `initialize_assignment`.
        distancetype: The type of distance to use. Can be one of:
            - `'1'`: Use insertion cost of 1 (like in Levenshtein distance)
            - `'2'`: Use insertion cost of 2 (cost of insertion + deletion)
            - `'21'`: Start with '2' until converged and then use '1' until converged

    >>> greedy_time_constrained_combination_matching(
    ...     [[(0, 0, 1), (1, 1, 2)]],
    ...     [[(0, 0, 1), (1, 1, 2)]],
    ...     [0]
    ... )
    (0, [0])

    >>> greedy_time_constrained_combination_matching(
    ...     [[(0, 0, 1)], [(1, 1, 2)]],
    ...     [[(0, 0, 1)], [(1, 1, 2)]],
    ...     [0, 0]
    ... )
    (0, [0, 1])
    """
    if len(segments) == 0:
        return sum([len(s) for s in streams]), []
    if len(streams) == 0:
        return sum([len(s) for s in segments]), [0] * len(segments)

    assert len(initial_assignment) == len(segments), (len(initial_assignment), len(segments), initial_assignment)

    # Correct assignment
    assignment = initial_assignment
    assert distancetype in ('1', '2', '21'), distancetype
    for d in distancetype:
        def forward_col(column, a, b):
            return cy_forward_col_time_constrained(
                column,
                a=np.asarray([t for t, _, _ in a], dtype=np.uint),
                b=np.asarray([t for t, _, _ in b], dtype=np.uint),
                a_begin=np.asarray([t for _, t, _ in a], dtype=np.float64),
                a_end=np.asarray([t for _, _, t in a], dtype=np.float64),
                b_begin=np.asarray([t for _, t, _ in b], dtype=np.float64),
                b_end=np.asarray([t for _, _, t in b], dtype=np.float64),
                cost_substitution=int(d)
            )
        assignment, distance = _greedy_correct_assignment(
            segments, streams, assignment,
            forward_col
        )

    return distance, assignment
