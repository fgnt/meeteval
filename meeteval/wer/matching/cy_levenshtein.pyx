# distutils: language = c++
#cython: language_level=3

import numpy as np
from libcpp.vector cimport vector
from libcpp.pair cimport pair

ctypedef unsigned int uint

cdef extern from "levenshtein.h":
    uint levenshtein_distance_(
            vector[uint] reference,
            vector[uint] hypothesis,
    )

    uint levenshtein_distance_custom_cost_(
            vector[uint] reference,
            vector[uint] hypothesis,
            uint cost_del,
            uint cost_ins,
            uint cost_sub,
            uint cost_cor,
    )

    uint time_constrained_levenshtein_distance_[T](
            vector[uint] reference,
            vector[uint] hypothesis,
            vector[pair[T, T]] reference_timing,
            vector[pair[T, T]] hypothesis_timing,
            uint cost_del,
            uint cost_ins,
            uint cost_sub,
            uint cost_cor,
    )

    uint time_constrained_levenshtein_distance_v2_[T](
            vector[uint] reference,
            vector[uint] hypothesis,
            vector[pair[T, T]] reference_timing,
            vector[pair[T, T]] hypothesis_timing,
            uint cost_del,
            uint cost_ins,
            uint cost_sub,
            uint cost_cor,
    )

    uint time_constrained_levenshtein_distance_unoptimized_[T](
            vector[uint] reference,
            vector[uint] hypothesis,
            vector[pair[T, T]] reference_timing,
            vector[pair[T, T]] hypothesis_timing,
            uint cost_del,
            uint cost_ins,
            uint cost_sub,
            uint cost_cor,
    )

    struct LevenshteinStatistics:
        uint insertions
        uint deletions
        uint substitutions
        uint correct
        uint total
        vector[pair[uint, uint]] alignment

    LevenshteinStatistics time_constrained_levenshtein_distance_with_alignment_[T](
            vector[uint] reference,
            vector[uint] hypothesis,
            vector[pair[T, T]] reference_timing,
            vector[pair[T, T]] hypothesis_timing,
            uint cost_del,
            uint cost_ins,
            uint cost_sub,
            uint cost_cor,
            uint eps
    )


def obj2vec(*args):
    # Taken from kaldialign https://github.com/pzelasko/kaldialign/blob/17d2b228ec575aa4f45ff2a191fb4716e83db01e/kaldialign/__init__.py#L6-L15
    sym2int = {v: i for i, v in enumerate({a_ for a in args for a_ in a})}
    return [[sym2int[a_] for a_ in a] for a in args]


def _validate_costs(cost_del, cost_ins, cost_sub, cost_cor):
    if not (isinstance(cost_del, int) and isinstance(cost_ins, int) and isinstance(cost_sub, int) and isinstance(
            cost_cor, int)):
        raise ValueError(
            f'Only unsigned integer costs are supported, but found cost_del={cost_del}, '
            f'cost_ins={cost_ins}, cost_sub={cost_sub}, cost_cor={cost_cor}'
        )

def levenshtein_distance(
        reference,
        hypothesis,
        cost_del=1,
        cost_ins=1,
        cost_sub=1,
        cost_cor=0,
):
    reference, hypothesis = obj2vec(reference, hypothesis)
    if cost_del == 1 and cost_ins == 1 and cost_sub == 1 and cost_cor == 0:
        # This is the fast case where we can use the standard optimized algorithm
        return levenshtein_distance_(reference, hypothesis)

    _validate_costs(cost_del, cost_ins, cost_sub, cost_cor)

    return levenshtein_distance_custom_cost_(
        reference, hypothesis, cost_del, cost_ins, cost_sub, cost_cor
    )

cdef _validate_inputs(reference, hypothesis, vector[pair[double, double]] reference_timing, vector[pair[double, double]] hypothesis_timing):
    if len(reference) != len(reference_timing):
        raise ValueError(
            f'reference and reference_timing have mismatching lengths '
            f'{len(reference)} != {len(reference_timing)}'
        )
    if len(hypothesis) != len(hypothesis_timing):
        raise ValueError(
            f'hypothesis and hypothesis_timing have mismatching lengths '
            f'{len(hypothesis)} != {len(hypothesis_timing)}'
        )

    # reference_timing = np.array(reference_timing)
    # hypothesis_timing = np.array(hypothesis_timing)

    # assert len(reference) == 0 or reference_timing.shape == (len(reference), 2), (
    # reference_timing.shape, len(reference))
    # assert len(hypothesis) == 0 or hypothesis_timing.shape == (len(hypothesis), 2), (
    # hypothesis_timing.shape, len(hypothesis))
    # assert len(reference) == 0 or len(hypothesis) == 0 or reference_timing.dtype == hypothesis_timing.dtype, (
    # reference_timing.dtype, hypothesis_timing.dtype)

    cdef double last_start = 0
    for t in reference_timing:
        if t.second < t.first:
            raise ValueError(
                f'The end time of an interval must not be smaller than its begin time, but the reference violates this')
        if t.first < last_start:
            raise ValueError(
                f'The start times of the annotations must be increasing, which they are not for the reference. '
                f'This might be caused by overlapping segments, see the (potential) previous warning.'
            )
        last_start = t.first

    last_start = 0
    for t in hypothesis_timing:
        if t.second < t.first:
            raise ValueError(
                f'The end time of an interval must not be smaller than its begin time, but the hypothesis violates this')
        if t.first < last_start:
            raise ValueError(
                f'The start times of the annotations must be increasing, which they are not for the hypothesis. '
                f'This might be caused by overlapping segments, see the (potential) previous warning.'
            )
        last_start = t.first

    return reference, hypothesis, reference_timing, hypothesis_timing

def time_constrained_levenshtein_distance(
        reference,  # list[int]
        hypothesis,  # list[int]
        reference_timing,  # list[tuple[int, int]]
        hypothesis_timing,  # list[tuple[int, int]]
        cost_del: uint = 1,
        cost_ins: uint = 1,
        cost_sub: uint = 1,
        cost_cor: uint = 0,
):
    _validate_costs(cost_del, cost_ins, cost_sub, cost_cor)
    # reference_timing, hypothesis_timing = times_to_int(reference_timing, hypothesis_timing)
    reference, hypothesis, reference_timing, hypothesis_timing = _validate_inputs(
        reference, hypothesis, reference_timing, hypothesis_timing
    )


    if len(reference) == 0:
        return len(hypothesis) * cost_ins
    if len(hypothesis) == 0:
        return len(reference) * cost_del
    reference, hypothesis = obj2vec(reference, hypothesis)

    args = (reference, hypothesis,
            reference_timing,
            hypothesis_timing,
            cost_del,
            cost_ins,
            cost_sub,
            cost_cor,)
    if np.issubdtype(reference_timing.dtype, np.signedinteger):
        return time_constrained_levenshtein_distance_[int](*args)
    elif np.issubdtype(reference_timing.dtype, np.unsignedinteger):
        return time_constrained_levenshtein_distance_[uint](*args)
    elif np.issubdtype(reference_timing.dtype, np.floating):
        return time_constrained_levenshtein_distance_[double](*args)
    else:
        raise TypeError(reference_timing.dtype)

def time_constrained_levenshtein_distance_unoptimized(
        reference,  # list[int]
        hypothesis,  # list[int]
        reference_timing,  # list[tuple[int, int]]
        hypothesis_timing,  # list[tuple[int, int]]
        cost_del=1,
        cost_ins=1,
        cost_sub=1,
        cost_cor=0,
):
    """
    The time-constrained levenshtein distance without time-pruning optimization. This mainly exists so we
    can test the optimized implementation against this one.
    """
    _validate_costs(cost_del, cost_ins, cost_sub, cost_cor)
    reference, hypothesis, reference_timing, hypothesis_timing = _validate_inputs(
        reference, hypothesis, reference_timing, hypothesis_timing
    )
    if len(reference) == 0:
        return len(hypothesis) * cost_ins
    if len(hypothesis) == 0:
        return len(reference) * cost_del
    reference, hypothesis = obj2vec(reference, hypothesis)

    args = (reference, hypothesis,
            reference_timing,
            hypothesis_timing,
            cost_del,
            cost_ins,
            cost_sub,
            cost_cor,)
    if np.issubdtype(reference_timing.dtype, np.signedinteger):
        return time_constrained_levenshtein_distance_unoptimized_[int](*args)
    elif np.issubdtype(reference_timing.dtype, np.unsignedinteger):
        return time_constrained_levenshtein_distance_unoptimized_[uint](*args)
    elif np.issubdtype(reference_timing.dtype, np.floating):
        return time_constrained_levenshtein_distance_unoptimized_[double](*args)
    else:
        raise TypeError(reference_timing.dtype)

def time_constrained_levenshtein_distance_with_alignment(
        reference,  # list[int]
        hypothesis,  # list[int]
        reference_timing,  # list[tuple[int, int]]
        hypothesis_timing,  # list[tuple[int, int]]
        cost_del=1,
        cost_ins=1,
        cost_sub=1,
        cost_cor=0,
        eps='*',
):
    reference, hypothesis, reference_timing, hypothesis_timing = _validate_inputs(reference, hypothesis,
                                                                                  reference_timing, hypothesis_timing)
    if len(reference) == 0:
        return {
            'total': len(hypothesis),
            'alignment': [(eps, h) for h in hypothesis],
            'correct': 0,
            'insertions': len(hypothesis) * cost_ins,
            'deletions': 0,
            'substitutions': 0,
        }
    if len(hypothesis) == 0:
        return {
            'total': len(reference),
            'alignment': [(r, eps) for r in reference],
            'correct': 0,
            'insertions': 0,
            'deletions': len(reference) * cost_del,
            'substitutions': 0,
        }

    assert eps not in reference and eps not in hypothesis, (eps, reference, hypothesis)
    map2int = False
    if len(reference) and isinstance(reference[0], int) or len(hypothesis) and isinstance(hypothesis[0], int):
        assert isinstance(eps, int), eps
    else:
        map2int = True
        int2sym = dict(enumerate(sorted(set(reference) | set(hypothesis) | set([eps]))))
        sym2int = {v: k for k, v in int2sym.items()}
        reference = [sym2int[a_] for a_ in reference]
        hypothesis = [sym2int[b_] for b_ in hypothesis]
        eps = sym2int[eps]

    statistics = time_constrained_levenshtein_distance_with_alignment_[double](
        reference, hypothesis,
        reference_timing,
        hypothesis_timing,
        cost_del,
        cost_ins,
        cost_sub,
        cost_cor,
        eps
    )

    # Cython for some reason doesn't allow *args here
    # if np.issubdtype(reference_timing.dtype, np.signedinteger):
    #     statistics = time_constrained_levenshtein_distance_with_alignment_[int](
    #         reference, hypothesis,
    #         reference_timing,
    #         hypothesis_timing,
    #         cost_del,
    #         cost_ins,
    #         cost_sub,
    #         cost_cor,
    #         eps
    #     )
    # elif np.issubdtype(reference_timing.dtype, np.unsignedinteger):
    #     statistics = time_constrained_levenshtein_distance_with_alignment_[uint](
    #         reference, hypothesis,
    #         reference_timing,
    #         hypothesis_timing,
    #         cost_del,
    #         cost_ins,
    #         cost_sub,
    #         cost_cor,
    #         eps
    #     )
    # elif np.issubdtype(reference_timing.dtype, np.floating):
    #     statistics = time_constrained_levenshtein_distance_with_alignment_[double](
    #         reference, hypothesis,
    #         reference_timing,
    #         hypothesis_timing,
    #         cost_del,
    #         cost_ins,
    #         cost_sub,
    #         cost_cor,
    #         eps
    #     )
    # else:
    #     raise TypeError(reference_timing.dtype)

    if map2int:
        statistics['alignment'] = [
            (int2sym[e[0]], int2sym[e[1]])
            for e in statistics['alignment']
        ]

    return statistics

import numpy as np

# from libcpp.vector cimport vector
# from libcpp.vector cimport vector

# def overlap(r_t, h_t):
#     return r_t[0] < h_t[1] and r_t[1] > h_t[0]


# def times_to_int(a, b):
#     int2sym = dict(enumerate(sorted({
#         begin_or_end
#         for list_ in [a, b]
#         for begin, end in list_
#         for begin_or_end in [begin, end]
#     })))
#     sym2int = {v: k for k, v in int2sym.items()}
#     a = [(sym2int[begin], sym2int[end]) for begin, end in a]
#     b = [(sym2int[begin], sym2int[end]) for begin, end in b]
#     return a, b

cimport cython
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)
def time_constrained_levenshtein_distance_v2(
    reference_, hypothesis_,
    reference_timing_, hypothesis_timing_,
):
    cdef:
        int cost_del = 1
        int cost_ins = 1
        int cost_sub = 1
        int cost_cor = 0

        int cost_insdel = cost_ins + cost_del

        int i = 0
        int j, r, h, diagonal, up
        double [:] r_t
        double [:] h_t
        int allow_shift
        int tmp

        int hyp_start = 0


    cdef int hypothesis_len = len(hypothesis_)
    cdef int reference_len = len(reference_)

    reference_, hypothesis_ = obj2vec(reference_, hypothesis_)

    if reference_len == 0:
        return hypothesis_len * cost_ins
    if hypothesis_len == 0:
        return reference_len * cost_del
    cdef int [:] reference = np.array(reference_, dtype=np.int32)
    cdef int [:] hypothesis = np.array(hypothesis_, dtype=np.int32)
    cdef double [:, :] reference_timing = np.array(reference_timing_, dtype=np.float64)
    cdef double [:, :] hypothesis_timing = np.array(hypothesis_timing_, dtype=np.float64)

    cdef int [:] row = np.full(hypothesis_len + 1, hypothesis_len + reference_len + 1, dtype=np.int32)
    row[0] = 0

    cdef float max_ref_end_time = 0

    for j in range(reference_len):
#         print(np.array(row), 'hyp_start', hyp_start)

        r = reference[j]
        r_t = reference_timing[j]

        max_ref_end_time = max(r_t[1], max_ref_end_time);

        diagonal = row[hyp_start]
        row[hyp_start] += cost_del;

        allow_shift = True

        for i in range(hyp_start, hypothesis_len):
            h = hypothesis[i]
            h_t = hypothesis_timing[i]

            if allow_shift:
                if r_t[0] > h_t[1]:
                    hyp_start += 1
                else:
                    allow_shift = False

            up = min(
                row[i+1],
                diagonal + cost_ins,  # This happens, when row[i+1] is the first time used.
            )

            row[i+1] = min(
                row[i] + cost_ins,
                up + cost_del,
                diagonal + (
                    (cost_cor if r == h else cost_sub)
                    if r_t[0] < h_t[1] and r_t[1] > h_t[0] else
                    cost_insdel
                )
            )

            diagonal = up
            if max_ref_end_time < h_t[0]:
                break

    for i in range(i+1, hypothesis_len):
        row[i+1] = row[i] + cost_ins

#     print(np.array(row), 'hyp_start', hyp_start)
    return row[hypothesis_len]


def time_constrained_levenshtein_distance_v3(
        reference,  # list[int]
        hypothesis,  # list[int]
        reference_timing,  # list[tuple[int, int]]
        hypothesis_timing,  # list[tuple[int, int]]
        cost_del: uint = 1,
        cost_ins: uint = 1,
        cost_sub: uint = 1,
        cost_cor: uint = 0,
):
    _validate_costs(cost_del, cost_ins, cost_sub, cost_cor)
    # reference_timing, hypothesis_timing = times_to_int(reference_timing, hypothesis_timing)
    reference, hypothesis, reference_timing, hypothesis_timing = _validate_inputs(
        reference, hypothesis, reference_timing, hypothesis_timing
    )

    if len(reference) == 0:
        return len(hypothesis) * cost_ins
    if len(hypothesis) == 0:
        return len(reference) * cost_del

    if len(reference) and isinstance(reference[0], int) or len(hypothesis) and isinstance(hypothesis[0], int):
        pass
    else:
        reference, hypothesis = obj2vec(reference, hypothesis)

    args = (reference, hypothesis,
            reference_timing,
            hypothesis_timing,
            cost_del,
            cost_ins,
            cost_sub,
            cost_cor,)
    return time_constrained_levenshtein_distance_v2_[double](*args)
    # if np.issubdtype(reference_timing.dtype, np.signedinteger):
    #     return time_constrained_levenshtein_distance_v2_[int](*args)
    # elif np.issubdtype(reference_timing.dtype, np.unsignedinteger):
    #     return time_constrained_levenshtein_distance_v2_[uint](*args)
    # elif np.issubdtype(reference_timing.dtype, np.floating):
    #     return time_constrained_levenshtein_distance_v2_[double](*args)
    # else:
    #     raise TypeError(reference_timing.dtype)

cpdef equidistant_intervals(pair[double, double] interval, unsigned int count):
    """Divides the interval into `count` equally sized intervals"""
    cdef:
        double interval_length, start, stop
        unsigned int i
    if count == 0:
        return []
    elif count == 1:
        return [interval]
    interval_length = (interval.second - interval.first) / count
    cdef vector[pair[double, double]] out = vector[pair[double, double]](count)

    start = interval.first
    out[0].first = start
    start += interval_length
    out[0].second = start
    #     out.push_back(pair[double, double](start, stop))
    for i in range(1, count):
        out[i].first = start
        start += interval_length
        out[i].second = start

    #         out.push_back(pair[double, double](start, stop))
    return out
#     return [(interval.first + i * interval_length, interval.first + (i + 1) * interval_length) for i in range(count)]
