# distutils: language = c++
#cython: language_level=3

from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

ctypedef unsigned int uint
import logging


cdef extern from "time_constrained_mimo_matching.h":
    pair[uint, vector[pair[uint, uint]]] time_constrained_mimo_levenshtein_distance_(
            vector[vector[vector[uint]]] reference,
            vector[vector[uint]] hypothesis,
            vector[vector[vector[pair[double, double]]]] reference_timings,
            vector[vector[pair[double, double]]] hypothesis_timings
    ) except +

def time_constrained_mimo_levenshtein_distance(
        reference,
        hypothesis,
        reference_timings,
        hypothesis_timings
):
    """
    Compute the time-constrained MIMO Levenshtein distance between two sequences of 
    sequences of symbols and returns the distance and the alignment.
    """
    # Validate inputs
    if len(reference) != len(reference_timings):
        raise ValueError("reference and reference_timings must have the same length")
    if len(hypothesis) != len(hypothesis_timings):
        raise ValueError("hypothesis and hypothesis_timings must have the same length")
    
    # Shortcuts for trivial cases
    if len(reference) == 0:
        return sum(len(h_) for h in hypothesis for h_ in h), []
    if len(hypothesis) == 0:
        # 0 is a dummy stream index
        return sum(len(r_) for r in reference for r_ in r), [
            (speaker_index, 0)
            for speaker_index, speaker in enumerate(reference)
            for _ in range(len(speaker))
        ]
    
    # Translate symbols/words to integers for the cpp code
    all_symbols = set()
    for r in reference:
        for r_ in r:
            all_symbols.update(set(list(r_)))
    for h in hypothesis:
        all_symbols.update(set(list(h)))
    int2sym = dict(enumerate(sorted(all_symbols)))
    sym2int = {v: k for k, v in int2sym.items()}

    reference = [[[sym2int[h__] for h__ in h_] for h_ in h] for h in reference]
    hypothesis = [[sym2int[h_] for h_ in h] for h in hypothesis]

    return time_constrained_mimo_levenshtein_distance_(
            reference,
            hypothesis,
            reference_timings,
            hypothesis_timings
    )
