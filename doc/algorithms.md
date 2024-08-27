# Core Algorithms used in MeetEval

This file contains simplified implementations of the core algorithms used in the MeetEval toolkit.
These implementations are meant for documentation.
The actual implementations in the toolkit are more complex and track the alignment along with the distance computation, which is omitted here for clarity.

To simplify the implementations, in this file:
 - words are represented as characters (The string `"abc"` represents three words)
 - some implementations will complain about empty inputs
For these cases the implementation is trivial and left out here for brevity.
The implementations in MeetEval treat these cases correctly.

Note that the code in this file requires at least Python version 3.11.

## Levenshtein Distance

> [!NOTE]
> 
> [View C++ Implementation](../meeteval/wer/matching/levenshtein.h#L8)

The Levenshtein distance (https://en.wikipedia.org/wiki/Levenshtein_distance, https://nymity.ch/sybilhunting/pdf/Levenshtein1966a.pdf) is the core of the word error rate.
We use a version of the Wagner-Fisher algorithm that only uses linear memory (see, for example, https://www.baeldung.com/cs/levenshtein-distance-computation).
There are other algorithms that are more efficient (.e.g. a bit-parallel algorithm in https://dl.acm.org/doi/10.1145/316542.316550), but extensions to computationally more extensive algorithms presented below are unknown.

We'll split the Levenshtein distance into two functions so that the core algorithm can be re-used later.

```python
def update_lev_row(row: list, a: str, b: str):
    """
    Takes a `row` from the Levenshtein matrix and computes the following 
    `len(b)` rows. Returns only the last row.
    
    Args:
        row: A single row from the Levenshtein matrix. It's length must be
            `len(a) + 1`.
        a: The hypothesis string
        b: The reference string. One update will be computed for every word (here: character)
            in `b`.
    """
    # Iterate over words in b
    for i, b_word in enumerate(b, start=1):
        diagonal = row[0]
        row[0] += 1  # Deletion
    
        # Iterate over words in a
        for j, a_word in enumerate(a, start=1):
            dist = min([
                row[j - 1] + 1, # Insertion
                row[j] + 1, # Deletion
                diagonal + (0 if b_word == a_word else 1) # Correct / substitution
            ])
            diagonal = row[j]
            row[j] = dist
    return row

def levenshtein_distance(a: str, b: str):
    """The standard Levenshtein distance"""
    # Initialize the row with only insertions
    row = list(range(len(a) + 1))
    
    # Update row for the full string `b`
    row = update_lev_row(row, a, b)

    # The distance is the last element in the row
    return row[-1]

```

<!-- Tests 
```python
assert levenshtein_distance('kitten', 'sitting') == 3
assert levenshtein_distance('abc', 'abc') == 0 
```
-->

<details>
<summary>Version with numpy</summary>

```python
import numpy as np

def update_lev_row_np(row: list | np.ndarray, a: str, b: str):
    """
    Takes a `row` from the Levenshtein matrix and computes the following 
    `len(b)` rows. Returns only the last row.
    
    Args:
        row: A single row from the Levenshtein matrix. It's length must be
            `len(a) + 1`.
        a: The hypothesis string
        b: The reference string. One update will be computed for every word
            in `b`.
    """
    assert len(a) + 1 == len(row), (len(row), len(a), len(b))
    row2 = np.copy(row)
    # Iterate over words in b
    for i, b_word in enumerate(b, start=1):
        # row2[0] = row[0] + 1  # Deletion
        np.add(row[0:1], 1, out=row2[0:1])
    
        # Iterate over words in a
        for j, a_word in enumerate(a, start=1):
            if b_word == a_word:
                # The diagonal is always best if we have a correct match
                row2[j] = row[j-1]
            else:
                np.minimum(
                    row2[j - 1:j], # Insertion
                    row[j:j+1], # Deletion
                    out = row2[j:j+1],
                )
                np.minimum(
                    row2[j:j+1],
                    row[j - 1:j],   # Substitution
                    out = row2[j:j+1]
                )
                row2[j] += 1
            
        row, row2 = row2, row
    return row
```
</details>

## Time-constrained Levenshtein Distance

> [!NOTE]
> 
> [View C++ Implementation](../meeteval/wer/matching/levenshtein.h#398) <br>
> [View C++ Implementation with alignment tracking](../meeteval/wer/matching/levenshtein.h#254)

The time-constrained Levenshtein Distance adds the constraint that two words that are temporally far apart cannot be matched as correct or substituted.
This can be achieved with a simple overlaps check in the minimum operation.

The search space can be pruned often dramatically by only computing a narrow band through the Levenshtein matrix.
This pruning is not implemented here to keep this example simple, but the idea is simple: In the loop (a), iterate only over those words in `a` that overlap with `b_word`.

The final implementation is not that simple, though.
The overhead of the naive implementation is too large to see a good speedup.
The begin _and_ end times used for this pruning must be monotonically increasing which can be achieved by introducing accumulated minimum and maximum times for the pruning.
Additionally, the edges of the pruned area have to be treated correctly.
You can check the [actual implementation](../meeteval/wer/matching/levenshtein.h#L398) for details.

```python
def overlaps(a_begin, a_end, b_begin, b_end, collar):
    """Checks whether two words overlap temporally based on their begin and end times"""
    return a_begin < b_end + collar and b_begin < a_end + collar

def time_constrained_levenshtein_distance(a, b, collar: float):
    """
    Args:
        a: [(word, start, end), ...]
        b: [(word, start, end), ...]
        collar: 
    """
    row = list(range(len(a) + 1))  # Initialize with only insertions
    
    # Iterate over words in b.
    for i, (b_word, b_begin, b_end) in enumerate(b, start=1):
        diagonal = row[0]
        row[0] += 1  # Deletion

        # (a) Iterate over words in a.
        #
        # This can be sped up by only iterating over the words that overlap 
        # with b[i-1]. When doing this, one has to take care of a few edge-cases,
        # especially handling words that don't overlap with any other words
        for j, (a_word, a_begin, a_end) in enumerate(a, start=1):
            tmp = min(
                row[j - 1] + 1, # Insertion
                row[j] + 1, # Deletion
            )

            # Only allow correct/substitution when words overlap (including collar)
            if overlaps(a_begin, a_end, b_begin, b_end, collar):
                tmp = min(tmp, diagonal + (0 if a_word == b_word else 1))

            diagonal = row[j]
            row[j] = tmp
    return row[-1]    
```

<!-- Tests 
```python
assert time_constrained_levenshtein_distance([('a', 0, 1)], [('a', 1, 2)], 5) == 0
assert time_constrained_levenshtein_distance([('a', 0, 1)], [('a', 1, 2)], 0) == 2
```
-->


## Concatenated minimum-Permutation Levenshtein Distance (cpLev)

The core algorithm of the Concatenated minimum-Permutation Word Error Rate (cpWER) is the concatenated minimum-permutation Levenshtein Distance (cpLev).
It computes a distance between two sets of word sequences by finding a bijective mapping (permutation) between the two sets that minimizes the sum of the pairwise distances. 

But, we can find the optimum faster than computing the total distance for every permutation, which would mean computing $K!$ distances for $K$ streams.
The Hungarian algorithm solves the linear sum assignment or weighted bipartite graph matching problem in polynomial time. 
First, the pairwise distances are computed between every pair of strings.
Then, the optimal assignment is found by the Hungarian algorithm.

```python
def cp_lev(a: list[str], b: list[str]):
    # Add dummy speakers for over-/under-estimation
    a = a + [''] * (len(b) - len(a))
    b = b + [''] * (len(a) - len(b))

    # Compute the Levenshtein distance for all combinations.
    # This is faster than brute-forcefully computing the distances
    # for all permutations
    distances = [
        [
            levenshtein_distance(a_, b_)
            for b_ in b
        ] for a_ in a
    ]

    # Find the permutation (combination of entries in `distances`)
    # that minimizes the overall distance
    algorithm = 'hungarian'
    if algorithm == 'brute-force':
        # straightforward brute-force version. Its complexity is factorial in the number of speakers
        from itertools import permutations
        return min(
          sum([distances[i][p] for i, p in enumerate(permutation)])
          for permutation in permutations(range(len(a)))  
        )
    elif algorithm == 'hungarian':
        # The Hungarian algorithm / linear sum assignment is usually faster 
        # Its complexity is polynomial in the number of speakers
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(distances)
        return sum(distances[r_][c_] for r_, c_ in zip(r, c))
```

<!-- Tests 
```python
assert cp_lev(['a', 'b'], ['b', 'a']) == 0
assert cp_lev(['a', 'b'], ['b', 'a', 'c']) == 1
assert cp_lev(['a', 'b', 'c'], ['b', 'a']) == 1 
```
-->

<details>
<summary>View the time-constrained Concatenated minimum-Permutation Levenshtein distance algorithm</summary>

```python
def tcp_lev(a: list[list[tuple[str, float, float]]], b: list[list[tuple[str, float, float]]], collar):
    # Add dummy speakers for over-/under-estimation
    a = a + [[]] * (len(b) - len(a))
    b = b + [[]] * (len(a) - len(b))

    # Compute the Levenshtein distance for all combinations.
    # This is faster than brute-forcefully computing the distances
    # for all permutations
    distances = [
        [
            time_constrained_levenshtein_distance(a_, b_, collar=collar)
            for b_ in b
        ] for a_ in a
    ]

    # Find the permutation (combination of entries in `distances`)
    # that minimizes the overall distance
    algorithm = 'hungarian'
    if algorithm == 'brute-force':
        # straightforward brute-force version. Its complexity is factorial in the number of speakers
        from itertools import permutations
        return min(
          sum([distances[i][p] for i, p in enumerate(permutation)])
          for permutation in permutations(range(len(a)))  
        )
    elif algorithm == 'hungarian':
        # The Hungarian algorithm / linear sum assignment is usually faster 
        # Its complexity is polynomial in the number of speakers
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(distances)
        return sum(distances[r_][c_] for r_, c_ in zip(r, c))
```

<!-- Tests 
```python
assert tcp_lev([[('a', 0, 1)], [('b', 1, 2)]], [[('b', 5, 6)], [('a', 4, 5)]], collar=5) == 0
assert tcp_lev([[('a', 0, 1)], [('b', 1, 2)]], [[('b', 5, 6)], [('a', 4, 5)]], collar=0) == 4
```
-->

</details>

## Optimal Reference Combination Levenshtein Distance

The core algorithm of the ORC-WER.
The ORC-WER chooses the "combination" (or assignment) of references (to output streams) that minimizes the WER.
The simplest way is to compute every assignment and choose the one with the minimum WER:

```python
import numpy as np

def orc_lev_brute_force(reference: list[str], hypothesis: list[str]):
    """Brute-force variant of the ORC Levenshtein distance"""
    
    num_system_output_streams = len(hypothesis)
    # The index of distances is interpreted as the combination/assignment of reference utterances to hypothesis streams
    # So distances contains all possible combinations
    distances = np.zeros((num_system_output_streams,) * len(reference))
    
    for combination in np.ndindex(distances.shape):
        # Build reference streams for this combination
        reference_streams = ['' for _ in range(len(hypothesis))]
        for c, r in zip(combination, reference, strict=True):
            reference_streams[c] = reference_streams[c] + r

        # Compute distance between the constructed reference streams and the hypothesis
        distances[combination] = sum(levenshtein_distance(h, r) for h, r in zip(hypothesis, reference_streams))

    # Return minimum distance
    return np.amin(distances)
```

<!-- Tests 
```python
assert orc_lev_brute_force(['a', 'bc'], ['ab', 'c']) == 2
assert orc_lev_brute_force(['a', 'bc'], ['abc']) == 0
assert orc_lev_brute_force(['a', 'bc'], ['bc', 'a']) == 0
```
-->

This can be done more efficiently, though, by using a dynamic programming approach.
It implicitly keeps track of the assignments and only stores partial solutions that can at this point still lead to a better solution.
It uses the standard Levenshtein row update already described above.

```python
def orc_lev_dynamic_programming(reference: list[str], hypothesis: list[str]):
    """Iterative dynamic-programming implementation of the ORC Levenshtein distance"""
    from itertools import product
    import numpy as np

    # Reserve memory for the multi-dimensional Levenshtein matrix
    L = np.zeros([len(hypothesis)] + [len(h) + 1 for h in hypothesis], dtype=int)

    # Initialize with only deletions
    a = np.arange(np.sum(L.shape))
    L[...] = np.lib.stride_tricks.as_strided(a, L.shape[1:], strides=(a.itemsize, )*(L.ndim - 1))
        
    # Go through reference utterances in (temporal) order
    for reference_utterance in reference:
        
        # Compute the updates separately for every hypothesis stream.
        # This is the utterance consistency constraint that makes sure that
        # there is no stream switch within an utterance
        for active in range(len(hypothesis)):
            # update_lev_row_np performs a batch update where the trailing dimensions
            # are the free batch dimensions
            L[active] = np.moveaxis(
                update_lev_row_np(
                    np.moveaxis(L[active], active, 0),
                    hypothesis[active],
                    reference_utterance
                ), 0, active
            )
            
        # End of utterance (position of a "change token" in [1])
        # Take the minimum over all possible assignments for this utterance, element-wise
        L[...] = np.min(L, axis=0)
    return L.flatten()[-1]
```

<!-- Tests 
```python
assert orc_lev_dynamic_programming(['a', 'bc'], ['ab', 'c']) == 2
assert orc_lev_dynamic_programming(['a', 'bc'], ['abc']) == 0
assert orc_lev_dynamic_programming(['a', 'bc'], ['bc', 'a']) == 0
```
-->

### Greedy Algorithm for the ORC Levenshtein distance

> [!NOTE]
> 
> [View Python / Cython / Numpy Implementation](../meeteval/wer/matching/greedy_combination_matching.py)

Even the dynamic programming algorithm can be infeasible when the number of output streams increases.
The ORC Levenshtein distance can be approximated with high accuracy by a greedy algorithm.
The greedy algorithm gradually improves an assignment by checking for every utterance whether switching its label would decrease the distance.
If so, the label is changed.
This iteration is continued until the distance cannot be improved by switching the label of a single utterance.

```python
def greedy_orc_lev(reference: list[str], hypothesis: list[str]):    
    # Initialize assignment with 0s (every reference utterance is assigned to the first hyp stream)
    #
    # If the hypothesis contains speaker labels, the cpWER can be used for initialization, as 
    # described in the paper, which gives a much better starting point.
    #
    assignment = [0] * len(reference)
    current_distance = levenshtein_distance(''.join(reference), hypothesis[0]) + sum([len(h) for h in hypothesis[1:]])
    
    # Iterate until the assignment is not changed
    while True:
        best_assignment = list(assignment)
        
        # Check for every reference utterance ...
        for utterance_index in range(len(assignment)):
            current = assignment[utterance_index]
            best = current
            
            # ... if putting it on another stream improves the distance
            for stream_index in range(len(hypothesis)):    
                assignment[utterance_index] = stream_index
                
                # Build reference streams for this modified assignment / combination
                streams = ['' for _ in range(len(hypothesis))]
                for c, r in zip(assignment, reference):
                    streams[c] = streams[c] + r
                    
                # Compute cost (compare orc_lev)
                modified_distance = sum([levenshtein_distance(h, r) for h, r in zip(hypothesis, streams)])
                
                # Update choice if the distance improved
                if modified_distance < current_distance:
                    best = stream_index
                    current_distance = modified_distance

            # Update the assignment and make sure to continue iteration when the assignment changed
            assignment[utterance_index] = best

        # Stop iterating if the assignment did not change
        if best_assignment == assignment:
            break
    return current_distance
```

<!-- Tests 
```python
assert greedy_orc_lev(['a', 'bc'], ['ab', 'c']) == 2
assert greedy_orc_lev(['a', 'bc'], ['abc']) == 0
assert greedy_orc_lev(['a', 'bc'], ['bc', 'a']) == 0
```
-->

### Faster version of the greedy ORC Levenshtein distance algorithm

The algorithm presented above computes the full Levenshtein distance (across the full streams) every time a modified label is checked.
This complexity can be reduced by re-using state from the previous computations.
We can make use of the fact that the Levenshtein distance is symmetric (`lev(a,b) == lev(a[::-1], b[::-1])`) to see the following:

```python
def levenshtein_matrix(a, b):
    """Fills the full levenshtein matrix using the `update_lev_row` helper function"""
    m = np.zeros((len(b) + 1, len(a) + 1), dtype=int)
    m[0, :] = np.arange(m.shape[1])
    for i, b_ in enumerate(b, start=1):
        m[i, :] = update_lev_row_np(m[i-1, :].copy(), a, b_)
    return m

a = 'abcf'
b = 'cde'

distance = levenshtein_distance(a, b)
forward_matrix = levenshtein_matrix(a, b)
backward_matrix = levenshtein_matrix(a[::-1], b[::-1])[::-1, ::-1]

print(forward_matrix + backward_matrix)
# [[4 4 4 6 7]
#  [5 4 4 4 5]
#  [6 5 4 4 4]
#  [7 6 5 4 4]]

# The minimum across any row or column in the sum of the two matrices is equal to the distance!
assert np.all(np.min(forward_matrix + backward_matrix, axis=0) == distance)
assert np.all(np.min(forward_matrix + backward_matrix, axis=1) == distance)
```

At any point in the strings, we can compute the cost of removing words by adding the corresponding columns of the forward and backward matrices 
and compute the cost of inserting words by updating a column in the forward matrix and adding the corresponding column from the backward matrix:

```python
# Remove the second word from the string b
assert np.min(forward_matrix[1] + backward_matrix[2]) == levenshtein_distance('abcf', 'ce')

# Add a word "x" to the string b
assert np.min(update_lev_row(forward_matrix[2], 'abcf', 'x') + backward_matrix[2]) == levenshtein_distance('abcf', 'cdxe')
```

Additionally, it is only required for every assignment to compute the cost difference between having the utterance on the stream and not having it on the stream.
With this trick, we can further reduce the number of times a Levenshtein distance has to be computed because the assignment with the smallest cost difference is the one that has the smallest cost.

```python
def construct_streams(reference, assignment, num_streams):
    streams = ['' for _ in range(num_streams)]
    for c, r in zip(assignment, reference):
        streams[c] = streams[c] + r
    return streams

def greedy_orc_lev_forward_backward(reference: list[str], hypothesis: list[str]):    
    # Initialize assignment with 0s (every reference utterance is assigned to the first hyp stream)
    #
    # If the hypothesis contains speaker labels, the cpWER can be used for initialization, as 
    # described in the paper, which gives a much better starting point.
    #
    assignment = [0] * len(reference)
    
    # Iterate until the assignment is not changed
    while True:
        best_assignment = list(assignment)
        
        # Compute backward matrices for every stream
        backward_matrices = [
            levenshtein_matrix(hyp_stream[::-1], ref_stream[::-1])[::-1, ::-1] 
            for ref_stream, hyp_stream in zip(construct_streams(reference, assignment, len(hypothesis)), hypothesis)
        ]
        # Track the current position in the backward matrix for every stream
        backward_matrix_indices = [0] * len(hypothesis)
        # Track the costs for each stream
        stream_costs = [bm[0, 0] for bm in backward_matrices]
        
        # We only need to keep track of a single column of the forward matrix. This is done in forward_columns
        forward_columns = [
            np.arange(len(r) + 1, dtype=np.uint) for r in hypothesis
        ]
        
        # Temporary variables
        cost_differences = np.zeros((len(hypothesis),), dtype=int)
        costs_with_utterance = np.zeros((len(hypothesis),), dtype=int)
        costs_without_utterance = np.zeros((len(hypothesis),), dtype=int)
        updated_forward_columns = [None] * len(hypothesis)
        
        # For every utterance ...
        for utterance_index in range(len(assignment)):
            current = assignment[utterance_index]
            best = current
            
            # ... find the stream with the lowest cost
            for stream_index in range(len(hypothesis)):  
                
                # Compute the forward column for mapping the utterance on stream `stream_index`
                # The row and column updates are symmetrical, so `update_lev_row` can also be used to update a column in 
                # the Levenshtein matrix
                updated_forward_column = update_lev_row(forward_columns[stream_index], hypothesis[stream_index], reference[utterance_index])
                
                # Compute the cost difference between having the the utterance on stream `stream_index` vs not having it on this stream
                if stream_index == current:
                    # Keep current assignment
                    
                    # The cost with the utterance on the stream is already computed in stream_costs
                    cost_with_utterance = stream_costs[stream_index]
                    
                    # Compute the cost without the utterance on this stream with the forward-backward matrix trick
                    backward_column = backward_matrices[stream_index][backward_matrix_indices[stream_index] + len(reference[utterance_index])]
                    cost_without_utterance = int(np.min(forward_columns[stream_index] + backward_column))
                else:
                    # Switch label
                    
                    # Compute the cost with the utterance on this stream with the forward-backward matrix trick
                    cost_with_utterance = int(np.min(updated_forward_column + backward_matrices[stream_index][backward_matrix_indices[stream_index]]))
                    
                    # The cost without the utterance on this stream is already computed in stream_costs
                    cost_without_utterance = stream_costs[stream_index]
                    
                # Record the computed values for every stream index
                cost_difference = cost_with_utterance - cost_without_utterance
                cost_differences[stream_index] = cost_difference
                costs_with_utterance[stream_index] = cost_with_utterance
                costs_without_utterance[stream_index] = cost_without_utterance
                updated_forward_columns[stream_index] = updated_forward_column
                
            # Find the stream assignment with the best cost (aka best cost difference)
            best = int(np.argmin(cost_differences))
            
            # Update the corresponding entries
            forward_columns[best] = updated_forward_columns[best]
            backward_matrix_indices[current] += len(reference[utterance_index])
            assignment[utterance_index] = best
            if best != current:
                stream_costs[current] = costs_without_utterance[current]
                stream_costs[best] = costs_with_utterance[best]

        # Stop iterating if the assignment did not change
        if best_assignment == assignment:
            break
    return np.sum(stream_costs)
```

<!-- Tests 
```python
assert greedy_orc_lev_forward_backward(['a', 'bc'], ['ab', 'c']) == 2
assert greedy_orc_lev_forward_backward(['a', 'bc'], ['abc']) == 0
assert greedy_orc_lev_forward_backward(['a', 'bc'], ['bc', 'a']) == 0
```
-->

## MIMO Levenshtein Distance

The MIMO Levenshtein distance is the core algorithm of the MIMO-WER and extends the ORC Levenshtein distance to multiple reference streams.
The brute-force implementation adds a step that permutes the utterances on the output streams for each assignment.
Only permutations are allowed where the order of utterances from the same speaker is the same as in the reference.

```python
import copy

def mimo_assignments(reference: list[list[str]], num_streams: int):
    """
    Yields all valid assignments according to the constraints that the 
    order of utterances from the same speaker should not change.
    
    Builds the assignments recursively by starting with an empty list of references
    and adding utterances one by one.
    """
    reference = [r for r in reference if r]  # remove empty
    if not reference:  # all empty, yield empty streams
        yield [[] for _ in range(num_streams)]
    else:
        for r in reference:
            utt = r.pop(0)
            for assignment in mimo_assignments(reference, num_streams):
                for stream in range(num_streams):
                    assignment[stream].insert(0, utt)
                    yield copy.deepcopy(assignment)
                    assignment[stream].pop(0)
            r.insert(0, utt)


def mimo_lev_bruteforce(reference: list[list[str]], hypothesis: list[str]):
    """Brute-force variant of the MIMO Levenshtein distance"""
    distances = []
    
    for streams in mimo_assignments(reference, len(hypothesis)):
        distances.append(sum(levenshtein_distance(h, r) for h, r in zip(hypothesis, streams)))
            
    # Report minimum distance
    return min(distances)
```

<!-- Tests 
```python
assert mimo_lev_bruteforce([['a', 'b'], ['c']], ['ac', 'b']) == 0
assert mimo_lev_bruteforce([['a', 'b'], ['c']], ['ca', 'b']) == 0
assert mimo_lev_bruteforce([['a', 'b'], ['c']], ['a', 'bc']) == 0
assert mimo_lev_bruteforce([['a', 'b'], ['c']], ['ac', 'bd']) == 1
assert mimo_lev_bruteforce([['a', 'b', 'c']], ['ca', 'b']) == 2
```
-->

The ORC dynamic programming algorithm can also be extended to MIMO by adding an iteration over the combinations of reference utterances:

```python
def mimo_lev_dynamic_programming(reference: list[list[str]], hypothesis: list[str]):
    """Iterative dynamic-programming implementation of the MIMO Levenshtein distance"""
    import itertools
    import numpy as np

    # Reserve memory for the multi-dimensional Levenshtein matrix (this is eq. (3) in ICASSSP 2023 paper without the change token)
    L = np.zeros([len(reference), len(hypothesis)] + [len(r) + 1 for r in reference] + [len(h) + 1 for h in hypothesis], dtype=int)

    # Initialize with only insertions
    a = np.arange(np.sum(L.shape))
    L[...] = np.lib.stride_tricks.as_strided(a, L.shape[2 + len(reference):], strides=(a.itemsize, )*(L.ndim - 2 - len(reference)))
        
    # Go through reference utterances in (temporal) order
    for reference_cell in itertools.product(*[range(len(r) + 1) for r in reference]):
        
        # The initial cell is already initialized
        if all(r == 0 for r in reference_cell):
            continue
        
        # We can reach this cell from every speaker
        for r in range(len(reference)):
            if reference_cell[r] == 0:
                # If this is a border cell, ignore in minimum
                L[r, :, *reference_cell, ...] = 2147483647 # 2^31 - 1
                continue
                
            # Get the current reference utterance that corresponds to the speaker `r` in 
            # `reference_cell`
            reference_utterance = reference[r][reference_cell[r] - 1]
            
            # Compute the index of the "parent" cell
            parent_cell = list(reference_cell)
            parent_cell[r] -= 1
            
            # Compute the updates separately for every hypothesis stream.
            # This is the utterance consistency constraint that makes sure that
            # there is no stream switch within an utterance
            for active in range(len(hypothesis)):
                L[r, active, *reference_cell] = np.moveaxis(
                    update_lev_row_np(
                        np.ascontiguousarray(np.moveaxis(L[r, active, *parent_cell], active, 0)),
                        hypothesis[active],
                        reference_utterance
                    ),
                    0,
                    active
                )
            
        # End of utterance (position of a "change token" in [1])
        # Take the minimum over all possible assignments for this cell, across all speakers
        L[:, :, *reference_cell, ...] = np.min(L[:, :, *reference_cell, ...], axis=(0, 1))
    return L.flatten()[-1]
```

<!-- Tests 
```python
assert mimo_lev_dynamic_programming([['a', 'b'], ['c']], ['ac', 'b']) == 0
assert mimo_lev_dynamic_programming([['a', 'b'], ['c']], ['ca', 'b']) == 0
assert mimo_lev_dynamic_programming([['a', 'b'], ['c']], ['a', 'bc']) == 0
assert mimo_lev_dynamic_programming([['a', 'b'], ['c']], ['ac', 'bd']) == 1
assert mimo_lev_dynamic_programming([['a', 'b', 'c']], ['ca', 'b']) == 2
```
-->

## Diarization-Invariant Concatenated minimum-Permutation Levenshtein Distance (DI-cpLev)

The Diarization-Invariant Concatenated minimum-Permutation Levenshtein Distance (DI-cpLev) is the core of the DI-cpWER.
It can be computed by swapping the reference and hypothesis in the ORC Levenshtein distance:

```python
def di_cp_lev(reference: list[str], hpyothesis: list[str]):
    return orc_lev_dynamic_programming(hypothesis, reference)
```