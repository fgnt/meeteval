# distutils: language = c++
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free

ctypedef unsigned int uint

cdef inline uint advance_index(
        vector[uint] *index,
        vector[uint] *index_factors,
        vector[uint] *index_lengths,
        uint *update_index
):
    """
    Advances an index to the next cell by increasing it along the first
    dimension and wrapping around to the following dimensions.
    
    Returns True when the index could be advanced and False if we reached the
    end.
    """
    cdef:
        uint i
        uint j
    for i in range(index.size() - 1, 1, -1):
        index[0][i] += 1
        if index[0][i] < index_lengths[0][i]:
            # TODO: don't use multiplications
            update_index[0] = uint_for_index(index, index_factors)
            return True
        else:
            index[0][i] = 0
    return False

cdef inline void reset_index(
        vector[uint] *index,
        uint *update_index,
        vector[uint] *index_factors,
        uint active_index,
        uint ref_index
):
    """
    Resets an index pointer to the beginning position given an active and ref 
    index.
    """
    for i in range(2, index.size()):
        index[0][i] = 0
    index[0][0] = active_index
    index[0][1] = ref_index
    update_index[0] = active_index * index_factors[0][0] + ref_index * index_factors[0][1]

cdef inline uint uint_for_index(
        vector[uint] *index,
        vector[uint] *index_factors
):
    """
    Returns an uint index for a index vector.
    """
    cdef:
        uint i
        uint j
    i = 0
    for j in range(index_factors[0].size()):
        i += index_factors[0][j] * index[0][j]
    return i

def cy_orc_matching(ref: vector[uint], hyps: vector[vector[uint]]):
    cdef:
        uint index_length
        vector[uint] index
        uint i
        uint j
        uint a
        uint r
        uint h1
        uint h2
        uint ma
        uint mr
        uint mh1
        uint mh2
        uint v
        uint v2
        uint len_a
        uint len_r
        uint len_h1
        uint len_h2
        uint *lev_matrix
        uint matrix_size
        uint ref_str_index
        uint ref_offset
        uint ref_index
        uint hyp_symbol
        uint ref_symbol
        uint filler = 32000
        uint update_index
        uint update_index1
        uint update_index2
        uint update_index3
        uint update_index4
        uint diagonal_update

        vector[uint] index_lengths
        vector[uint] index_factors
        vector[uint] assignment

    if hyps.size() == 0:
        # Shortcut
        v = 0
        for i in ref:
            if i != 0:
                v += 1
        return v, ()

    # Initialize indices
    len_a = hyps.size()
    len_r = ref.size() + 1
    len_str_r = ref.size()

    index_lengths = vector[uint]()
    index_lengths.push_back(len_a)
    index_lengths.push_back(len_r)
    for h in hyps:
        index_lengths.push_back(h.size() + 1)
    matrix_size = 1
    for i in index_lengths:
        matrix_size *= i

    lev_matrix = <unsigned int *> malloc(sizeof(uint) * matrix_size)
    for i in range(matrix_size):
        lev_matrix[i] = filler

    # TODO: memory layout
    index_factors = vector[uint]()
    i = 1
    for j in index_lengths:
        index_factors.push_back(i)
        i = i * j

    index = vector[uint]()
    for i in range(hyps.size() + 2):
        index.push_back(0)


    # Initialize the matrix
    update_index = 0
    # Fill the case ref=0
    index[1] = 0  # ref = 0
    for a in range(len_a):
        reset_index(&index, &update_index, &index_factors, a, 0)
        # update_inex = a*index_factors[0]
        while True:
            i = 0
            for j in range(2, index.size()):
                i += index[j]
            lev_matrix[update_index] = i

            # Emulate do-while loop
            if not advance_index(&index, &index_factors, &index_lengths, &update_index):
                break

    # Go through the reference character by character
    for ref_str_index in range(len_str_r):
        ref_symbol = ref[ref_str_index]
        ref_index = ref_str_index + 1

        if ref_symbol == 0:
            # The current symbol is a change token. Do the change token update.
            # Go through all cells and pick the one with the minimal value
            # along all hyps
            reset_index(&index, &update_index, &index_factors, 0, ref_index)

            while True:
                i = filler
                for a in range(len_a):
                    j = lev_matrix[update_index + a * index_factors[0] - index_factors[1]]
                    if j < i:
                        i = j
                for a in range(len_a):
                    lev_matrix[update_index + a * index_factors[0]] = i
                if not advance_index(&index, &index_factors, &index_lengths, &update_index):
                    break
            continue

        # Go over all active indices
        for a in range(0, len_a):
            reset_index(&index, &update_index, &index_factors, a, ref_index)

            while True:
                if lev_matrix[update_index] == filler:
                    v = filler
                    hyp_symbol = filler

                    # "Diagonal" update
                    if index[a + 2] > 0:
                        hyp_symbol = hyps[a][index[a + 2] - 1]  # index is offset by 1
                        v = lev_matrix[update_index - index_factors[1] - index_factors[a + 2]]

                    # Other updates: Only necessary when the current match is
                    # not a correct match because the correct match is always
                    # the minimum
                    if hyp_symbol != ref_symbol:
                        # We here have only two cases left: coming from the
                        # previous reference index or from the previous
                        # active hypothesis index
                        if index[1] > 0:
                            v2 = lev_matrix[update_index - index_factors[1]]
                            if v2 < v:
                                v = v2

                        if index[a + 2] > 0:
                            v2 = lev_matrix[update_index - index_factors[a + 2]]
                            if v2 < v:
                                v = v2

                        # Add cost: We here use a constant cost of 1 for
                        # everything
                        v += 1

                    lev_matrix[update_index] = v

                if not advance_index(&index, &index_factors, &index_lengths, &update_index):
                    break

    # Get assignment by backtracking through the matrix
    # TODO: Can this be shortened? Yes, this one goes over too many neighbors
    for i in range(index_lengths.size()):
        index[i] = index_lengths[i] - 1

    assignment = vector[uint]()
    update_index1 = 0
    while True:
        # Check if we reached the beginning. All indices except a must be 0.
        # This means we can't check for update_index == 0 because a can be
        # != 0
        done = True
        for i in range(1, index.size()):
            i = index[i]
            if i != 0:
                done = False
                break
        if done:
            break

        update_index = uint_for_index(&index, &index_factors)

        # Find the minimum over all neighbors
        v = filler
        j = filler
        diagonal_update = False

        # Updates in this direction only possible when there is a symbol
        if index[1] > 0:
            ref_symbol = ref[index[1] - 1]

            # ref_symbol == 0 is a change token. This is the assignment that we
            # are interested in
            if ref_symbol == 0:
                index[0] = 0
                index[1] -= 1
                update_index = uint_for_index(&index, &index_factors)
                v = lev_matrix[update_index]
                j = 0
                for a in range(1, len_a):
                    v2 = lev_matrix[update_index + a * index_factors[0]]
                    if v2 < v:
                        v = v2
                        j = a
                index[0] = j
                assignment.push_back(j)
                continue
            else:
                if index[index[0] + 2] > 0:
                    v = lev_matrix[update_index - index_factors[index[0] + 2] - index_factors[1]]
                    diagonal_update = True
                    j = index[0] + 2
        for i in range(1, index_lengths.size()):
            if index[i] > 0:

                v2 = lev_matrix[update_index - index_factors[i]]
                if v > v2:
                    v = v2
                    j = i
                    diagonal_update = False

        if diagonal_update:
            index[1] -= 1
            index[index[0] + 2] -= 1
        else:
            index[j] -= 1

        update_index1 += 1

    v = lev_matrix[matrix_size - 1]
    free(lev_matrix)
    return v, assignment[::-1]
