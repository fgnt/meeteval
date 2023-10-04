# distutils: language = c++
#cython: language_level=3

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool
ctypedef unsigned int uint


cdef extern from "levenshtein.h":
    const uint ALIGNMENT_EPS

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


cdef _timing_is_integer(timing):
    return all(isinstance(t[0], int) and isinstance(t[1], int) for t in timing)

def time_constrained_levenshtein_distance(
        reference,  # list[int]
        hypothesis,  # list[int]
        reference_timing,  # list[tuple[int, int]]
        hypothesis_timing,  # list[tuple[int, int]]
        cost_del: uint = 1,
        cost_ins: uint = 1,
        cost_sub: uint = 1,
        cost_cor: uint = 0,
        prune: bool = True
):
    _validate_costs(cost_del, cost_ins, cost_sub, cost_cor)

    # Shortcut when at least one of reference or hypothesis is empty
    if len(reference) == 0:
        return len(hypothesis) * cost_ins
    if len(hypothesis) == 0:
        return len(reference) * cost_del

    # Convert to vectors of integers for the C++ implementation
    reference, hypothesis = obj2vec(reference, hypothesis)

    args = (reference, hypothesis,
            reference_timing,
            hypothesis_timing,
            cost_del,
            cost_ins,
            cost_sub,
            cost_cor,)

    # Dispatch to the correct implementation
    if prune:
        if _timing_is_integer(reference_timing) and _timing_is_integer(hypothesis_timing):
            return time_constrained_levenshtein_distance_v2_[int](*args)
        else:
            return time_constrained_levenshtein_distance_v2_[double](*args)
    else:
        if _timing_is_integer(reference_timing) and _timing_is_integer(hypothesis_timing):
            return time_constrained_levenshtein_distance_unoptimized_[int](*args)
        else:
            return time_constrained_levenshtein_distance_unoptimized_[double](*args)



def time_constrained_levenshtein_distance_with_alignment(
        reference,  # list[int]
        hypothesis,  # list[int]
        reference_timing,  # list[tuple[int, int]]
        hypothesis_timing,  # list[tuple[int, int]]
        cost_del=1,
        cost_ins=1,
        cost_sub=1,
        cost_cor=0,
):
    _validate_costs(cost_del, cost_ins, cost_sub, cost_cor)

    if len(reference) == 0:
        return {
            'total': len(hypothesis),
            'alignment': [(None, i) for i in range(len(hypothesis))],
            'correct': 0,
            'insertions': len(hypothesis) * cost_ins,
            'deletions': 0,
            'substitutions': 0,
        }
    if len(hypothesis) == 0:
        return {
            'total': len(reference),
            'alignment': [(i, None) for i in range(len(reference))],
            'correct': 0,
            'insertions': 0,
            'deletions': len(reference) * cost_del,
            'substitutions': 0,
        }

    map2int = False
    if len(reference) and isinstance(reference[0], int) or len(hypothesis) and isinstance(hypothesis[0], int):
        assert len(reference) == 0 or isinstance(reference[0], int), reference
        assert len(hypothesis) == 0 or isinstance(hypothesis[0], int), hypothesis
        reference_ = reference
        hypothesis_ = hypothesis
    else:
        map2int = True
        int2sym = dict(enumerate(sorted(set(reference) | set(hypothesis))))
        sym2int = {v: k for k, v in int2sym.items()}
        reference_ = [sym2int[a_] for a_ in reference]
        hypothesis_ = [sym2int[b_] for b_ in hypothesis]

    cdef LevenshteinStatistics statistics

    if _timing_is_integer(reference_timing) and _timing_is_integer(hypothesis_timing):
        statistics = time_constrained_levenshtein_distance_with_alignment_[int](
            reference_, hypothesis_,
            reference_timing,
            hypothesis_timing,
            cost_del,
            cost_ins,
            cost_sub,
            cost_cor,
        )
    else:
        statistics = time_constrained_levenshtein_distance_with_alignment_[double](
            reference_, hypothesis_,
            reference_timing,
            hypothesis_timing,
            cost_del,
            cost_ins,
            cost_sub,
            cost_cor,
        )

    # Translate sentinel to None
    # This is hacky, but we want to support c++11, which doesn't have std::optional
    py_statistics: dict = statistics
    py_statistics['alignment'] = [
        (
            None if a == ALIGNMENT_EPS else a,
            None if b == ALIGNMENT_EPS else b,
         )
        for (a, b) in py_statistics['alignment']
    ]

    return py_statistics

import numpy as np

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

    return row[hypothesis_len]
