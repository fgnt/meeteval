# distutils: language = c++
#cython: language_level=3
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from libcpp.pair cimport pair
cimport cython

ctypedef unsigned int uint


cdef extern from "mimo_matching.h":
    uint levenshtein_distance_(vector[uint] reference, vector[uint] hypothesis)
    pair[uint, vector[pair[uint, uint]]] mimo_matching_(vector[vector[vector[uint]]] references, vector[vector[uint]] hypotheses) except +


def obj2vec(a, b):
    int2sym = dict(enumerate(sorted(set(a) | set(b))))
    sym2int = {v: k for k, v in int2sym.items()}
    return [sym2int[a_] for a_ in a], [sym2int[b_] for b_ in b]


def levenshtein_distance_cpp(
        reference,
        hypothesis,
):
    reference, hypothesis = obj2vec(reference, hypothesis)
    return levenshtein_distance_(reference, hypothesis)


def cpp_mimo_matching(
        references,
        hypotheses,
):
    all_symbols = set()
    for r in references:
        for r_ in r:
            all_symbols.update(set(list(r_)))
    for h in hypotheses:
        all_symbols.update(set(list(h)))
    int2sym = dict(enumerate(sorted(all_symbols)))
    sym2int = {v: k for k, v in int2sym.items()}

    references = [[[sym2int[r__] for r__ in r_] for r_ in r] for r in references]
    hypotheses = [[sym2int[h_] for h_ in h] for h in hypotheses]

    return mimo_matching_(references, hypotheses)


cdef uint update_lev_row(uint * row, uint * from_index_buffer, vector[uint] ref, vector[uint] hyp):
    """
    Updates the levenshtein matrix row `row` with symbols from `ref`.
    The row must have size `hyp.size() + 1`.
    
    Also fills `from_index_buffer` with the indices into the beginning row 
    that the perfect paths go through. That means that the matching path
    that goes through `i` in the output row goes through `from_index_buffer[i]`
    in the input row. This is ues to track the assignment.
    """
    cdef:
        uint ref_symbol
        uint r, h, v
        # The variables below are named after the relative positions in the 2D
        # levenshtein matrix
        uint left, up, diagonal
        uint left_from, up_from, diagonal_from, v_from

    # Initialize from_index_buffer with a range
    for h in range(hyp.size() + 1):
        from_index_buffer[h] = h

    # We assume that row is already initialized
    for r in range(ref.size()):
        ref_symbol = ref[r]
        diagonal = row[0]
        diagonal_from = from_index_buffer[0]

        # The first entry of row counts the deletions when no symbol from ref is matched.
        # Simply count it up. Doing this here saves us an if statement in the loop
        row[0] = row[0] + 1
        left = row[0]
        left_from = from_index_buffer[0]

        for h in range(1, hyp.size() + 1):
            hyp_symbol = hyp[h - 1]

            up = row[h]
            up_from = from_index_buffer[h]

            # Take the diagonal update first. This is likely the correct path
            v = diagonal
            v_from = diagonal_from

            # If the symbols don't match (no correct diagonal update), check
            # whether an update from above or left is better than diagonal
            if ref_symbol != hyp_symbol:
                if up < v:
                    v = up
                    v_from = up_from
                if left < v:
                    v = left
                    v_from = left_from

                # Add costs: 1 for all operations here
                v += 1

            # Re-use left. The current value in this cell becomes the next cell's "left".
            left = v
            left_from = v_from

            # When moving right one cell, this cell's up becomes the next cell's diagonal
            diagonal = up
            diagonal_from = up_from
            row[h] = left
            from_index_buffer[h] = left_from

def cy_levenshtein_distance(vector[uint] ref, vector[uint] hyp):
    """
    Computes the levenshtein distance between `ref` and `hyp`.

    This function mainly exists for testing the levenshtein distance algorithm
    against known libraries.
    """
    cdef:
        uint * lev_row
        uint v

    lev_row = <uint*>malloc(sizeof(uint)*(hyp.size() + 1))
    lev_index_row = <uint*>malloc(sizeof(uint)*(hyp.size() + 1))
    for i in range(hyp.size() + 1):
        lev_row[i] = i
    update_lev_row(lev_row, lev_index_row, ref, hyp)
    v = lev_row[hyp.size()]
    free(lev_row)
    return v


# This "MetaIndex" collects information about the dimension strides and
# sizes. Not sure if this is a good name for this construct
cdef struct MetaIndex:
    vector[uint] index_lengths
    vector[uint] index_factors
    uint total_size

cdef inline MetaIndex make_meta_index(vector[uint] index_lengths):
    cdef:
        uint i, j
        vector[uint] index_factors

    # Get the factors needed to find the index into the tensor from the indices
    # into the dimensions
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
    """Gets the index along a dimension from the tensor index"""
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
        uint ** ref_utterance_storage_ref_link
        uint ** ref_utterance_storage_active_hyp
        uint ** ref_utterance_storage
        vector[pair[uint, uint]] assignment

    ###########################################################################
    # Initialization
    ###########################################################################

    # Initialize the hypothesis metaindex
    hyps_index_lengths = vector[uint]()
    for h in hyps:
        hyps_index_lengths.push_back(h.size() + 1)  # +1 because of "only insertions" field
    hyps_metaindex = make_meta_index(hyps_index_lengths)

    # Initialize the reference metaindex
    ref_utterance_index_lengths = vector[uint]()
    for r in refs:
        ref_utterance_index_lengths.push_back(r.size() + 1)
    refs_metaindex = make_meta_index(ref_utterance_index_lengths)

    # Initialize storage along reference dimensions.
    #
    # We have to store the cost at each reference combination and a link to
    # the place we came from to obtain this value so that we can reconstruct
    # the assignment.
    #
    # Store all full-utterance states TODO: discard as much data as possible
    ref_indices = vector[uint](refs.size(), 0)
    ref_utterance_storage = <unsigned int**> malloc(sizeof(unsigned int *) * refs_metaindex.total_size)
    ref_utterance_storage_ref_link = <unsigned int**> malloc(sizeof(unsigned int *) * refs_metaindex.total_size)
    ref_utterance_storage_active_hyp = <unsigned int**> malloc(sizeof(unsigned int *) * refs_metaindex.total_size)
    ref_utterance_storage_hyp_link = <unsigned int**> malloc(sizeof(unsigned int *) * refs_metaindex.total_size)

    # Initialize first element
    hyps_state = <unsigned int *> malloc(sizeof(unsigned int) * hyps_metaindex.total_size)
    state_ref_link = <unsigned int *> malloc(sizeof(unsigned int) * hyps_metaindex.total_size)
    state_hyp_link = <unsigned int *> malloc(sizeof(unsigned int) * hyps_metaindex.total_size)
    state_active_hyp = <unsigned int *> malloc(sizeof(unsigned int) * hyps_metaindex.total_size)
    ref_utterance_storage[0] = hyps_state
    ref_utterance_storage_active_hyp[0] = state_active_hyp
    ref_utterance_storage_hyp_link[0] = state_hyp_link
    ref_utterance_storage_ref_link[0] = state_ref_link
    for i in range(hyps_metaindex.total_size):
        j = 0
        for v in range(hyps_metaindex.index_lengths.size()):
            j += index_for_element(hyps_metaindex, i, v)
        hyps_state[i] = j

        state_ref_link[i] = 0
        state_hyp_link[i] = 0
        state_active_hyp[i] = 0

    # Temp variable for row of the levenshtein matrix for a hypothesis.
    # These are the buffers passed to `update_lev_row`
    # Must be size of at least max size of all hypotheses
    i = hyps_metaindex.index_lengths[0]
    for j in hyps_metaindex.index_lengths:
        if i < j:
            i = j
    tmp_row = <unsigned int *> malloc(sizeof(unsigned int) * i)
    tmp_index_row = <unsigned int *> malloc(sizeof(unsigned int) * i)

    ###########################################################################
    # Algorithm: Main loop
    ###########################################################################

    # Walk through assignments and update lev states.
    # We can guarantee that all previous cells have been filled by running
    # through the tensor indices incrementally, given the way we defined our
    # memory layout
    for ref_utterance_index in range(1, refs_metaindex.total_size):
        for i in range(ref_indices.size()):
            ref_indices[i] += 1
            if ref_indices[i] < refs_metaindex.index_lengths[i]:
                break
            ref_indices[i] = 0

        # Get the destination storage
        state = <unsigned int *> malloc(sizeof(unsigned int) * hyps_metaindex.total_size)
        state_ref_link = <unsigned int *> malloc(sizeof(unsigned int) * hyps_metaindex.total_size)
        state_hyp_link = <unsigned int *> malloc(sizeof(unsigned int) * hyps_metaindex.total_size)
        state_active_hyp = <unsigned int *> malloc(sizeof(unsigned int) * hyps_metaindex.total_size)

        ref_utterance_storage[ref_utterance_index] = state
        ref_utterance_storage_ref_link[ref_utterance_index] = state_ref_link
        ref_utterance_storage_hyp_link[ref_utterance_index] = state_hyp_link
        ref_utterance_storage_active_hyp[ref_utterance_index] = state_active_hyp
        first_update = True

        # Loop through all adjacent reference combinations and advance to this one
        for active_ref_index in range(ref_indices.size()):
            # i is the active reference
            if ref_indices[active_ref_index] == 0:
                continue

            # Get the previous state, only based on the current assignment of
            # references
            prev_state = ref_utterance_storage[ref_utterance_index - refs_metaindex.index_factors[active_ref_index]]
            active_reference = refs[active_ref_index][ref_indices[active_ref_index] - 1]

            # Forward the Levenshtein row for this reference assignment for
            # each active hypothesis and keep the min values
            for active_hypothesis in range(hyps.size()):
                # Advance state vector for every hyp entry
                for i in range(hyps_metaindex.total_size):
                    if index_for_element(hyps_metaindex, i, active_hypothesis) != 0:
                        continue

                    # Copy the Levenshtein row into the buffer. This gives us
                    # cache locality and a fast algorithm for Levenshtein
                    # updates
                    for j in range(hyps_metaindex.index_lengths[active_hypothesis]):
                        tmp_row[j] = prev_state[i + j * hyps_metaindex.index_factors[active_hypothesis]]

                    # Apply Levenshtein algorithm
                    update_lev_row(tmp_row, tmp_index_row, active_reference, hyps[active_hypothesis])

                    # Keep only the min value in the state
                    for j in range(hyps_metaindex.index_lengths[active_hypothesis]):
                        if first_update or state[i + j * hyps_metaindex.index_factors[active_hypothesis]] > tmp_row[j]:
                            state[i + j * hyps_metaindex.index_factors[active_hypothesis]] = tmp_row[j]
                            state_hyp_link[i + j * hyps_metaindex.index_factors[active_hypothesis]] = i + tmp_index_row[
                                j] * hyps_metaindex.index_factors[active_hypothesis]
                            state_ref_link[i + j * hyps_metaindex.index_factors[active_hypothesis]] = active_ref_index
                            state_active_hyp[
                                i + j * hyps_metaindex.index_factors[active_hypothesis]] = active_hypothesis

                first_update = False

    # The distance value is the last entry
    v = ref_utterance_storage[refs_metaindex.total_size - 1][hyps_metaindex.total_size - 1]

    ###########################################################################
    # Backtracking for assignment
    ###########################################################################

    assignment = vector[pair[uint, uint]]()
    ref_utterance_index = refs_metaindex.total_size - 1
    hyp_index = hyps_metaindex.total_size - 1
    j = 0
    while ref_utterance_index != 0:
        active_ref_index = ref_utterance_storage_ref_link[ref_utterance_index][hyp_index]
        active_hypothesis = ref_utterance_storage_active_hyp[ref_utterance_index][hyp_index]
        hyp_index = ref_utterance_storage_hyp_link[ref_utterance_index][hyp_index]
        assignment.push_back(pair[uint, uint](active_ref_index, active_hypothesis))
        ref_utterance_index -= refs_metaindex.index_factors[active_ref_index]
        j += 1

    ###########################################################################
    # Cleanup
    ###########################################################################

    free(tmp_row)
    free(tmp_index_row)
    for i in range(refs_metaindex.total_size):
        free(ref_utterance_storage_active_hyp[i])
        free(ref_utterance_storage_ref_link[i])
        free(ref_utterance_storage[i])
    free(ref_utterance_storage_active_hyp)
    free(ref_utterance_storage_ref_link)
    free(ref_utterance_storage)

    return v, assignment[::-1]