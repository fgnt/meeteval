"""
This file contains different iterations of the Optimal Reference Combination
(ORC) matching.

All functions in here share a common interface.
The (single) reference must be a list of utterances where each utterances must
be an iterable of hashables.
The hypothesis must be a list (channels) of transcriptions (iterable of
hashables)
"""
import typing

Utterance = typing.Iterable[typing.Hashable]
Assignment = 'tuple[int, ...]'


def _get_channel_transcription_from_assignment(
        utterances: 'list[Utterance]',
        assignment: Assignment,
        num_channels: int
) -> 'list[list[typing.Hashable]]':
    import itertools
    c = [[] for _ in range(num_channels)]

    for r, a in zip(utterances, assignment):
        c[a].append(r)

    c = [list(itertools.chain.from_iterable(c_)) for c_ in c]

    return c


def _levensthein_distance_for_assignment(ref, hyps, assignment):
    from meeteval.wer.matching.cy_levenshtein import levenshtein_distance
    c = _get_channel_transcription_from_assignment(
        ref, assignment, num_channels=len(hyps)
    )
    d = sum([levenshtein_distance(h, r) for h, r in zip(hyps, c)])
    return d


def orc_matching_v1(ref, hyps):
    """
    Version 1 is a brute-force implementation of the orc levenshtein distance
    in Python using a fast implementation of the Levenshtein distance.
    """
    import itertools
    num_channels = len(hyps)

    best_distance = None
    best_assignment = None
    for assignment in itertools.product(range(num_channels), repeat=len(ref)):
        d = _levensthein_distance_for_assignment(ref, hyps, assignment)
        if best_distance is None or d < best_distance:
            best_distance = d
            best_assignment = assignment
    return best_distance, best_assignment


def orc_matching_v2(ref, hyps):
    """
    A Python implementation of the Dynamic Programming variant of the
    ORC matching algorithm.

    This recursive implementation is slow (slower than brute-force for small
    examples, becuase the brute-force variant computes the Levenshtein
    distance in optimized C code), but displays the algorithm nicely.
    """
    import itertools
    change_token = object()

    # Handle trivial edge cases
    if len(hyps) == 0:
        return sum(len(r) for r in ref), (None,) * len(ref)
    if len(ref) == 0:
        return sum(len(h) for h in hyps), ()

    # Insert change tokens between reference utterances
    ref = list(itertools.chain.from_iterable([list(r) + [change_token] for r in ref]))
    ref.pop()   # Remove last change token
    hyps = [list(h) for h in hyps]

    # This cache caches pairs of (cost, partial_assignment) for each element
    # in the Levenshtein tensor
    cache = {}

    def argmin(iterable):
        """A pure-Python argmin"""
        return min(enumerate(iterable), key=lambda x: x[1])[0]

    def lev(ref_len, hyps_len, idx=None):
        key = (ref_len, *hyps_len, idx)

        # Shortcut if we already computed this value
        if key in cache:
            return cache[key]

        if idx is not None and ref_len and ref[ref_len - 1] == change_token:
            # We reached a change token. Allow a channel switch. We set idx=None
            # here for caching of the result.
            # The reference index is decremented already here. That's arbitrary
            cache[key] = lev(ref_len - 1, hyps_len, idx=None)
            return cache[key]

        if idx is None:
            # Handle a channel switch
            # The reference index is already decremented. Find the minimum
            # over all hyps
            tmp = [
                lev(ref_len, hyps_len, idx=i)
                for i in range(len(hyps))
            ]
            index = argmin(tmp)

            # Append the index of the best match to the partial assignment
            cache[key] = tmp[index][0], tmp[index][1] + (index, )
            return cache[key]
        else:
            # A normal Levenshtein distance update
            if ref_len == 0:
                # Corner case: empty reference
                cost = sum(hyps_len), ()
            elif hyps_len[idx] == 0:
                # Corner case: empty hypothesis
                cost, assignment = lev(ref_len - 1, hyps_len, idx)
                cost = 1 + cost, assignment
            elif ref[ref_len - 1] == hyps[idx][hyps_len[idx] - 1]:
                # Correct match: diagonal update
                hyps_len[idx] -= 1
                cost = lev(ref_len - 1, hyps_len, idx)
                hyps_len[idx] += 1
            else:
                # No correct match. This is a substitution/insertion/deletion.
                # Find the min
                a = lev(ref_len - 1, hyps_len, idx)
                hyps_len[idx] -= 1
                b = lev(ref_len, hyps_len, idx)
                c = lev(ref_len - 1, hyps_len, idx)
                hyps_len[idx] += 1

                cost, assignment = min(a, b, c)
                cost = 1 + cost, assignment

            cache[key] = cost
            return cache[key]

    out = lev(len(ref), [len(h) for h in hyps])

    return out


def orc_matching_v3(
        ref: 'list[Utterance]',
        hyps: 'list[list[typing.Hashable]]'
):
    """
    A Cython implementation of the ORC matching algorithm
    """
    import itertools
    from .cy_orc_matching import cy_orc_matching

    # The Cython implementation uses integers as tokens, so translate ref and
    # hyps to integers
    seen = set()
    for utterance in ref:
        seen.update(set(utterance))
    for channel in hyps:
        seen.update(set(channel))
    mapping = {t: i + 1 for i, t in enumerate(seen)}  # 0 is the change token
    ref = [[mapping[t] for t in utterance] + [0] for utterance in ref]
    ref = list(itertools.chain.from_iterable(ref))
    hyps = [[mapping[t] for t in channel] for channel in hyps]

    distance, assignemnt = cy_orc_matching(ref, hyps)

    return distance, assignemnt
