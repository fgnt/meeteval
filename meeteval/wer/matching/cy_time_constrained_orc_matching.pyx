# distutils: language = c++
#cython: language_level=3

from libcpp.vector cimport vector
from libcpp.pair cimport pair

ctypedef unsigned int uint


cdef extern from "time_constrained_orc_matching.h":
    pair[uint, vector[uint]] time_constrained_orc_levenshtein_distance_(
            vector[vector[uint]] reference,
            vector[vector[uint]] hypothesis,
            vector[vector[pair[double, double]]] reference_timings,
            vector[vector[pair[double, double]]] hypothesis_timings
    ) except +

def time_constrained_orc_levenshtein_distance(
        reference,  # List (utterances) of list of symbols
        hypothesis, # List (streams) of list of symbols
        reference_timings,
        hypothesis_timings
):
    """
    Compute the time-constrained ORC Levenshtein distance between two sequences of 
    symbols and returns the distance and the alignment.
    """
    # Validate inputs
    if len(reference) != len(reference_timings):
        raise ValueError("reference and reference_timings must have the same length")
    if len(hypothesis) != len(hypothesis_timings):
        raise ValueError("hypothesis and hypothesis_timings must have the same length")

    # Shortcuts for trivial cases
    if len(reference) == 0:
        return sum(len(h) for h in hypothesis), []
    if len(hypothesis) == 0:
        # 0 is a dummy stream index
        return sum(len(r) for r in reference), [0] * len(reference)

    # Translate symbols/words to integers for the cpp code
    all_symbols = set()
    for r in reference:
        all_symbols.update(set(list(r)))
    for h in hypothesis:
        all_symbols.update(set(list(h)))
    int2sym = dict(enumerate(sorted(all_symbols)))
    sym2int = {v: k for k, v in int2sym.items()}

    reference = [[sym2int[h_] for h_ in h] for h in reference]
    hypothesis = [[sym2int[h_] for h_ in h] for h in hypothesis]

    return time_constrained_orc_levenshtein_distance_(
        reference,
        hypothesis,
        reference_timings,
        hypothesis_timings
    )
