# distutils: language = c++
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
cimport cython

ctypedef unsigned int uint

cdef uint update_lev_row(uint * row, vector[uint] ref, vector[uint] hyp):
    """
    Updates the levenshtein matrix row `row` with symbols from `ref`.
    The row must have size `hyp.size() + 1`.

    Variable names are named after the relative positions in the 2D 
    levenshtein matrix.
    """
    cdef:
        uint ref_symbol
        uint r, h
        uint left
        uint diagonal
        uint up

    # We assume that row is already initialized
    for r in range(ref.size()):

        ref_symbol = ref[r]
        diagonal = row[0]

        # The first entry of row counts the deletions when no symbol from ref is matched.
        # Simply count it up. Doing this here saves us an if statement in the loop
        row[0] = row[0] + 1
        left = row[0]

        for h in range(1, hyp.size() + 1):
            hyp_symbol = hyp[h - 1]

            up = row[h]

            # Re-use left. The current value in this cell becomes the next cell's "left".
            if ref_symbol == hyp_symbol:
                left = diagonal
            else:
                left = 1 + min(diagonal, min(left, up))

            # When moving right one cell, this cell's up becomes the next cell's diagonal
            diagonal = up
            row[h] = left

def cy_levenshtein_distance(vector[uint] ref, vector[uint] hyp):
    cdef:
        uint * lev_row
        uint v

    lev_row = <uint*>malloc(sizeof(uint)*(hyp.size() + 1))
    for i in range(hyp.size() + 1):
        lev_row[i] = i
    update_lev_row(lev_row, ref, hyp)
    v = lev_row[hyp.size()]
    free(lev_row)
    return v



cdef struct MetaIndex:
    vector[uint] index_lengths
    vector[uint] index_factors
    uint total_size


cdef inline MetaIndex make_meta_index(vector[uint] index_lengths):
    cdef:
        uint i, j
        vector[uint] index_factors

    index_factors = vector[uint]()
    j = 1
    for i in index_lengths:
        index_factors.push_back(j)
        j *= i

    # Compute total size
    j = 1
    for i in index_lengths:
        j *= i
    return MetaIndex(index_lengths, index_factors, j)


@cython.cdivision(True)
cdef inline uint index_for_element(MetaIndex metaindex, uint index, uint element):
    return (index // metaindex.index_factors[element]) % metaindex.index_lengths[element]


def cy_mimo_matching(vector[vector[vector[uint]]] refs, vector[vector[uint]] hyps):
    cdef:
        uint v, i, j
        uint hyps_tensor_size
        vector[uint] h
        vector[uint] hyps_index_lengths
        vector[uint] ref_utterance_index_lengths
        vector[uint] ref_indices
        vector[uint] active_reference
        uint active_ref_index
        MetaIndex hyps_metaindex
        MetaIndex refs_metaindex
        uint ref_utterance_index
        uint * tmp_row
        uint * prev_state

    hyps_index_lengths = vector[uint]()
    for h in hyps:
        hyps_index_lengths.push_back(h.size() + 1)  # +1 because of "only insertions" field
    hyps_metaindex = make_meta_index(hyps_index_lengths)

    # Store all full-utterance states TODO: discard as much data as possible
    ref_indices = vector[uint](refs.size(), 0)
    ref_utterance_index_lengths = vector[uint]()
    for r in refs:
        ref_utterance_index_lengths.push_back(r.size() + 1)
    refs_metaindex = make_meta_index(ref_utterance_index_lengths)
    ref_utterance_storage = <unsigned int**> malloc(sizeof(unsigned int *) * refs_metaindex.total_size)

    # Initialize first element
    hyps_state = <unsigned int *> malloc(sizeof(unsigned int) * hyps_metaindex.total_size)
    ref_utterance_storage[0] = hyps_state
    for i in range(hyps_metaindex.total_size):
        j = 0
        for v in range(hyps_metaindex.index_lengths.size()):
            j += index_for_element(hyps_metaindex, i, v)
        hyps_state[i] = j

    # Tmp for row of the levenshtein matrix for a hypothesis. Must be size of at least max size of all hypotheses
    i = hyps_metaindex.index_lengths[0]
    for j in hyps_metaindex.index_lengths:
        if i < j:
            i = j
    tmp_row = <unsigned int *> malloc(sizeof(unsigned int) * (i + 100))

    # Walk through assignments and update lev states
    for ref_utterance_index in range(1, refs_metaindex.total_size):
        for i in range(ref_indices.size()):
            ref_indices[i] = index_for_element(refs_metaindex, ref_utterance_index, i)

        # Get the destination storage
        state = <unsigned int *> malloc(sizeof(unsigned int) * hyps_metaindex.total_size)

        ref_utterance_storage[ref_utterance_index] = state
        first_update = True

        # Loop through all adjacent reference combinations and advance to this one
        # TODO: make this the inner loop
        for active_ref_index in range(ref_indices.size()):
            # i is the active reference
            if ref_indices[active_ref_index] == 0:
                continue

            prev_state = ref_utterance_storage[ref_utterance_index - refs_metaindex.index_factors[active_ref_index]]
            active_reference = refs[active_ref_index][ref_indices[active_ref_index] - 1]

            for active_hypothesis in range(hyps.size()):
                # Advance state vector for every hyp entry
                for i in range(hyps_metaindex.total_size):
                    if index_for_element(hyps_metaindex, i, active_hypothesis) != 0:
                        continue

                    for j in range(hyps_metaindex.index_lengths[active_hypothesis]):
                        tmp_row[j] = prev_state[i + j * hyps_metaindex.index_factors[active_hypothesis]]

                    update_lev_row(tmp_row, active_reference, hyps[active_hypothesis])

                    for j in range(hyps_metaindex.index_lengths[active_hypothesis]):
                        if first_update or state[i + j * hyps_metaindex.index_factors[active_hypothesis]] > tmp_row[j]:
                            state[i + j * hyps_metaindex.index_factors[active_hypothesis]] = tmp_row[j]

                first_update = False

    v = ref_utterance_storage[refs_metaindex.total_size - 1][hyps_metaindex.total_size - 1]

    free(tmp_row)
    for i in range(refs_metaindex.total_size):
        free(ref_utterance_storage[i])
    free(ref_utterance_storage)

    return v, None