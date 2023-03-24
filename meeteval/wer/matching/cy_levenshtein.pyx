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



def obj2vec(a, b):
    int2sym = dict(enumerate(sorted(set(a) | set(b))))
    sym2int = {v: k for k, v in int2sym.items()}
    return [sym2int[a_] for a_ in a], [sym2int[b_] for b_ in b]

def levenshtein_distance(
        reference,
        hypothesis,
):
    reference, hypothesis = obj2vec(reference, hypothesis)

    return levenshtein_distance_(reference, hypothesis)

def levenshtein_distance_custom_cost(
        reference,
        hypothesis,
        cost_del,
        cost_ins,
        cost_sub,
        cost_cor,
):
    reference, hypothesis = obj2vec(reference, hypothesis)

    return levenshtein_distance_custom_cost_(
        reference, hypothesis, cost_del, cost_ins, cost_sub, cost_cor
    )

def time_constrained_levenshtein_distance(
        reference, # list[int]
        hypothesis, # list[int]
        reference_timing,   # list[tuple[int, int]]
        hypothesis_timing,  # list[tuple[int, int]]
        cost_del=1,
        cost_ins=1,
        cost_sub=1,
        cost_cor=0,
):
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
    if len(reference) == 0:
        return len(hypothesis)
    if len(hypothesis) == 0:
        return len(reference)
    reference, hypothesis = obj2vec(reference, hypothesis)

    reference_timing = np.array(reference_timing)
    hypothesis_timing = np.array(hypothesis_timing)

    assert reference_timing.shape == (len(reference), 2), (reference_timing.shape, len(reference))
    assert hypothesis_timing.shape == (len(hypothesis), 2), (hypothesis_timing.shape, len(hypothesis))
    assert reference_timing.dtype == hypothesis_timing.dtype, (reference_timing.dtype, hypothesis_timing.dtype)

    def check(timing):
        assert np.all(timing[:, 1] >= timing[:, 0]), timing
        assert np.all(np.diff(timing, axis=0) >= 0), timing

    check(reference_timing)
    check(hypothesis_timing)

    if np.issubdtype(reference_timing.dtype, np.signedinteger):
        return time_constrained_levenshtein_distance_[int](
            reference, hypothesis,
            reference_timing,
            hypothesis_timing,
            cost_del,
            cost_ins,
            cost_sub,
            cost_cor,
        )
    elif np.issubdtype(reference_timing.dtype, np.unsignedinteger):
        return time_constrained_levenshtein_distance_[uint](
            reference, hypothesis,
            reference_timing,
            hypothesis_timing,
            cost_del,
            cost_ins,
            cost_sub,
            cost_cor,
        )
    elif np.issubdtype(reference_timing.dtype, np.floating):
        return time_constrained_levenshtein_distance_[float](
            reference, hypothesis,
            reference_timing,
            hypothesis_timing,
            cost_del,
            cost_ins,
            cost_sub,
            cost_cor,
        )
    else:
        raise TypeError(reference_timing.dtype)


def time_constrained_levenshtein_distance_unoptimized(
        reference, # list[int]
        hypothesis, # list[int]
        reference_timing,   # list[tuple[int, int]]
        hypothesis_timing,  # list[tuple[int, int]]
        cost_del=1,
        cost_ins=1,
        cost_sub=1,
        cost_cor=0,
):
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
    if len(reference) == 0:
        return len(hypothesis)
    if len(hypothesis) == 0:
        return len(reference)
    reference, hypothesis = obj2vec(reference, hypothesis)

    reference_timing = np.array(reference_timing)
    hypothesis_timing = np.array(hypothesis_timing)

    assert reference_timing.shape == (len(reference), 2), (reference_timing.shape, len(reference))
    assert  hypothesis_timing.shape == (len(hypothesis), 2), (hypothesis_timing.shape, len(hypothesis))
    assert reference_timing.dtype == hypothesis_timing.dtype, (reference_timing.dtype, hypothesis_timing.dtype)

    def check(timing):
        # end >= start
        assert np.all(timing[:, 1] >= timing[:, 0]), timing
        # start and end values are increasing
        assert np.all(np.diff(timing, axis=0) >= 0), timing

    check(reference_timing)
    check(hypothesis_timing)

    if np.issubdtype(reference_timing.dtype, np.signedinteger):
        return time_constrained_levenshtein_distance_unoptimized_[int](
            reference, hypothesis,
            reference_timing,
            hypothesis_timing,
            cost_del,
            cost_ins,
            cost_sub,
            cost_cor,
        )
    elif np.issubdtype(reference_timing.dtype, np.unsignedinteger):
        return time_constrained_levenshtein_distance_unoptimized_[uint](
            reference, hypothesis,
            reference_timing,
            hypothesis_timing,
            cost_del,
            cost_ins,
            cost_sub,
            cost_cor,
        )
    elif np.issubdtype(reference_timing.dtype, np.floating):
        return time_constrained_levenshtein_distance_unoptimized_[float](
            reference, hypothesis,
            reference_timing,
            hypothesis_timing,
            cost_del,
            cost_ins,
            cost_sub,
            cost_cor,
        )
    else:
        raise TypeError(reference_timing.dtype)


def time_constrained_levenshtein_distance_with_alignment(
        reference, # list[int]
        hypothesis, # list[int]
        reference_timing,   # list[tuple[int, int]]
        hypothesis_timing,  # list[tuple[int, int]]
        cost_del=1,
        cost_ins=1,
        cost_sub=1,
        cost_cor=0,
        eps='*',
):
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
    if len(reference) == 0:
        return {
            'total': len(hypothesis),
            'alignment': [(eps, h) for h in hypothesis],
            'correct': 0,
            'insertions': len(hypothesis),
            'deletions': 0,
            'substitutions': 0,
        }
    if len(hypothesis) == 0:
        return {
            'total': len(reference),
            'alignment': [(r, eps) for r in reference],
            'correct': 0,
            'insertions': 0,
            'deletions': len(reference),
            'substitutions': 0,
        }
    # reference, hypothesis = obj2vec(reference, hypothesis)

    assert eps not in reference and eps not in hypothesis, (eps, reference, hypothesis)

    int2sym = dict(enumerate(sorted(set(reference) | set(hypothesis) | set([eps]))))
    sym2int = {v: k for k, v in int2sym.items()}
    reference = [sym2int[a_] for a_ in reference]
    hypothesis = [sym2int[b_] for b_ in hypothesis]
    eps = sym2int[eps]

    reference_timing = np.array(reference_timing)
    hypothesis_timing = np.array(hypothesis_timing)

    assert reference_timing.shape == (len(reference), 2), (reference_timing.shape, len(reference))
    assert hypothesis_timing.shape == (len(hypothesis), 2), (hypothesis_timing.shape, len(hypothesis))
    assert reference_timing.dtype == hypothesis_timing.dtype, (reference_timing.dtype, hypothesis_timing.dtype)

    def check(timing):
        assert np.all(timing[:, 1] >= timing[:, 0]), timing
        assert np.all(np.diff(timing, axis=0) >= 0), timing

    check(reference_timing)
    check(hypothesis_timing)

    if np.issubdtype(reference_timing.dtype, np.signedinteger):
        statistics = time_constrained_levenshtein_distance_with_alignment_[int](
            reference, hypothesis,
            reference_timing,
            hypothesis_timing,
            cost_del,
            cost_ins,
            cost_sub,
            cost_cor,
            eps
        )
    elif np.issubdtype(reference_timing.dtype, np.unsignedinteger):
        statistics = time_constrained_levenshtein_distance_with_alignment_[uint](
            reference, hypothesis,
            reference_timing,
            hypothesis_timing,
            cost_del,
            cost_ins,
            cost_sub,
            cost_cor,
            eps
        )
    elif np.issubdtype(reference_timing.dtype, np.floating):
        statistics = time_constrained_levenshtein_distance_with_alignment_[float](
            reference, hypothesis,
            reference_timing,
            hypothesis_timing,
            cost_del,
            cost_ins,
            cost_sub,
            cost_cor,
            eps
        )
    else:
        raise TypeError(reference_timing.dtype)

    statistics['alignment'] =  [
        (int2sym[e[0]], int2sym[e[1]])
        for e in statistics['alignment']
    ]

    return statistics
