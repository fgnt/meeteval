# distutils: language = c++
#cython: language_level=3
import numpy as np
cimport cython
cimport numpy as np
from numpy cimport npy_uint64, npy_float64
np.import_array()
ctypedef unsigned int uint


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cy_forward_col(
    npy_uint64[:] column not None,
    npy_uint64[:] a not None,
    npy_uint64[:] b not None,
    uint cost_substitution = 1,
) -> np.ndarray:
    """
    Args:
        column: The column to be updated
        a: Sequence in column direction (make sure that `len(column) == len(a) + 1`! otherwise SEGFAULT!!)
        b: Sequence in row direction. This function updates `column` `len(b)` times
        cost_substitution: Cost for a substitution
    """
    cdef uint i, j, a_, b_, current, prev
    cdef npy_uint64[:, :] tmp = np.empty((2, column.shape[0]), dtype=np.uint)
    tmp[0, ...] = column
    current = 0
    for j in range(b.shape[0]):
        current = (j + 1) % 2
        prev = j % 2
        b_ = b[j]
        tmp[current, 0] = tmp[prev, 0] + 1
        for i in range(1, a.shape[0] + 1):
            a_ = a[i - 1]
            if a_ == b_:
                tmp[current, i] = tmp[prev, i - 1]
            else:
                tmp[current, i] = min([tmp[prev, i - 1] + cost_substitution, tmp[current, i - 1] + 1, tmp[prev, i] + 1])
    return np.asarray(tmp[current]).copy()


@cython.boundscheck(False)  # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cy_forward_col_time_constrained(
    npy_uint64[:] column not None,
    npy_uint64[:] a not None,
    npy_uint64[:] b not None,
    npy_float64[:] a_begin not None,
    npy_float64[:] a_end not None,
    npy_float64[:] b_begin not None,
    npy_float64[:] b_end not None,
    uint cost_substitution = 1,
) -> np.ndarray:
    """
    Args:
        column: The column to be updated
        a: Sequence in column direction (make sure that `len(column) == len(a) + 1`! otherwise SEGFAULT!!)
        b: Sequence in row direction. This function updates `column` `len(b)` times
        cost_substitution: Cost for a substitution
    """
    cdef uint i, j, a_, b_, current, prev
    cdef double a_begin_, a_end_, b_begin_, b_end_
    cdef npy_uint64[:, :] tmp = np.empty((2, column.shape[0]), dtype=np.uint)
    tmp[0, ...] = column
    current = 0
    for j in range(b.shape[0]):
        current = (j + 1) % 2
        prev = j % 2
        b_ = b[j]
        b_begin_ = b_begin[j]
        b_end_ = b_end[j]

        tmp[current, 0] = tmp[prev, 0] + 1
        for i in range(1, a.shape[0] + 1):
            a_ = a[i - 1]
            if a_begin[i - 1] >= b_end_ or b_begin_ >= a_end[i - 1]:
                # No overlap
                tmp[current, i] = min([tmp[current, i - 1] + 1, tmp[prev, i] + 1])
            elif a_ == b_:
                # Overlap correct
                tmp[current, i] = tmp[prev, i - 1]
            else:
                # Overlap incorrect
                tmp[current, i] = min([tmp[prev, i - 1] + cost_substitution, tmp[current, i - 1] + 1, tmp[prev, i] + 1])
    return np.asarray(tmp[current]).copy()
