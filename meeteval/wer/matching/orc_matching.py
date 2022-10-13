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
Assignment = typing.Tuple[int, ...]


def _get_channel_transcription_from_assignment(
        utterances: typing.List[Utterance],
        assignment: Assignment,
        num_channels: int
) -> typing.List[typing.List[typing.Hashable]]:
    import itertools
    c = [[] for _ in range(num_channels)]

    for r, a in zip(utterances, assignment):
        c[a].append(r)

    c = [list(itertools.chain.from_iterable(c_)) for c_ in c]

    return c


def _levensthein_distance_for_assignment(ref, hyps, assignment):
    import editdistance
    c = _get_channel_transcription_from_assignment(
        ref, assignment, num_channels=len(hyps)
    )
    d = sum([editdistance.distance(h, r) for h, r in zip(hyps, c)])
    return d


def orc_matching_v1(ref, hyps):
    """
    Version 1 is a brute-force implementation of the orc levenshtein distance
    in Python using the fast editdistance package.
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

    TODO: fix assignment!
    # This is wrong:
    >>> orc_matching_v2(['a', 'a'], ['a', 'b'])
    (1, [0, 0])
    """
    import itertools
    change_token = object()

    ref = list(itertools.chain.from_iterable([list(r) + [change_token] for r in ref]))
    hyps = [list(h) for h in hyps]
    cache = {}

    def argmin(iterable):
        return min(enumerate(iterable), key=lambda x: x[1])[0]

    assignments = {}

    def lev(ref_len, hyps_len, idx=None):
        key = (ref_len, *hyps_len, idx)

        if key in cache:
            return cache[key]

        if ref_len and ref[ref_len-1] == change_token:
            # Allow a channel switch
            return lev(ref_len-1, hyps_len, idx=None)

        if idx is None:
            tmp = [
                lev(ref_len, hyps_len, idx=i)
                for i in range(len(hyps))
            ]
            index = argmin(tmp)

            if ref_len not in assignments or assignments[ref_len][0] > tmp[index]:
                assignments[ref_len] = (tmp[index], index, tmp)
            return tmp[index]
        else:
            if ref_len == 0:
                cost = sum(hyps_len)
            elif hyps_len[idx] == 0:
                cost = 1 + lev(ref_len-1, hyps_len, idx)
            elif ref[ref_len-1] == hyps[idx][hyps_len[idx]-1]:
                hyps_len[idx] -= 1
                cost = lev(ref_len-1, hyps_len, idx)
                hyps_len[idx] += 1
            else:
                a = lev(ref_len - 1, hyps_len, idx)
                hyps_len[idx] -= 1
                b = lev(ref_len, hyps_len, idx)
                c = lev(ref_len - 1, hyps_len, idx)
                hyps_len[idx] += 1

                cost = 1 + min(a, b, c)

            cache[key] = cost
            return cache[key]

    out = lev(len(ref), [len(h) for h in hyps])

    return out, [a[1] for a in assignments.values()]


def orc_matching_v3(
        ref: typing.List[Utterance],
        hyps: typing.List[typing.List[typing.Hashable]]
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
    mapping = {t: i + 1 for i, t in enumerate(seen)}    # 0 is the change token
    ref = [[mapping[t] for t in utterance] + [0] for utterance in ref]
    ref = list(itertools.chain.from_iterable(ref))
    hyps = [[mapping[t] for t in channel] for channel in hyps]

    distance, assignemnt = cy_orc_matching(ref, hyps)

    # TODO: have a "self-test" here against editdistance?
    # d = _levensthein_distance_for_assignemnt(ref, hyps, assignment)
    # assert d == distance, (d, distance, assignment)
    return distance, assignemnt
