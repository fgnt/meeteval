import typing

Utterance = typing.Sequence[typing.Hashable]
Hypothesis = 'list[Utterance]'
Reference = 'list[list[Utterance]]'
Assignment = 'list[tuple[int, int]]'


def levenshtein_distance(ref, hyp):
    """Computes the levenshtein distance using the internals of MIMO WER.
    This function only exists for testing."""
    from .cy_mimo_matching import cy_levenshtein_distance
    return cy_levenshtein_distance(ref, hyp)


def mimo_matching_v1():
    """
    A brute-force implementation of the MIMO matching
    TODO!
    """
    pass


def mimo_matching_v2(
        refs: Reference,
        hyps: Hypothesis,
):
    """
    A Python implementation of the MIMO matching
    """
    import itertools
    change_token = object()
    refs = [
        list(itertools.chain.from_iterable([list(r) + [change_token] for r in ref]))
        for ref in refs
    ]
    hyps = [list(h) for h in hyps]

    cache = {}

    def argmin(iterable):
        return min(enumerate(iterable), key=lambda x: x[1])[0]

    def decrease_index(index, i):
        return (*index[:i], index[i] - 1, *index[i + 1:])

    assignments = {}

    def lev(refs_len, hyps_len, ref_idx=None, hyp_idx=None):
        """
        Args:
            refs_len: List of reference lengths
            hyps_len: List of hypothesis lengths
            ref_idx: Currently active reference
            hyp_idx: Currently active hypothesis
        """
        key = (*refs_len, *hyps_len, ref_idx, hyp_idx)

        if key in cache:
            return cache[key]

        # Change token: allow channel switch
        # We also have to allow channel switches at the beginning of each
        # reference.
        if (
                ref_idx is not None
                and (refs_len[ref_idx] and refs[ref_idx][refs_len[ref_idx] - 1] == change_token
                     or refs_len[ref_idx] == 0 and any(r > 0 for r in refs_len)
        )):
            return lev(refs_len, hyps_len, ref_idx=None, hyp_idx=None)

        # Change token, 2nd case: select best channel
        if ref_idx is None:
            assert hyp_idx is None

            indices = [
                (r, h)
                for h in range(len(hyps))
                for r in range(len(refs))
                if refs_len[r] > 0
            ]
            tmp = [
                lev(decrease_index(refs_len, r), hyps_len, ref_idx=r, hyp_idx=h)
                for r, h in indices
            ]

            index = argmin(tmp)
            cache[key] = tmp[index][0], tmp[index][1] + (indices[index],)
            return cache[key]
        else:
            # No channel switch. Perform normal lev update along slice of tensor

            assert refs_len[ref_idx] >= 0 and hyps_len[hyp_idx] >= 0

            # Edge case: initialize
            if refs_len[ref_idx] == 0:
                if all(r == 0 for r in refs_len):
                    cost = sum(hyps_len), ()
                else:
                    cost, assignment = lev(refs_len, decrease_index(hyps_len, hyp_idx), ref_idx, hyp_idx)
                    cost = cost + 1, assignment

            # Empy hypothesis: only deletion possible
            elif hyps_len[hyp_idx] == 0:
                cost, assignment = lev(decrease_index(refs_len, ref_idx), hyps_len, ref_idx, hyp_idx)
                cost = cost + 1, assignment

            # Correct match: Diagonal update
            elif refs[ref_idx][refs_len[ref_idx] - 1] == hyps[hyp_idx][hyps_len[hyp_idx] - 1]:
                cost = lev(decrease_index(refs_len, ref_idx), decrease_index(hyps_len, hyp_idx), ref_idx, hyp_idx)

            # No correct match: Pick best from insertion, deletion and substitution
            else:
                a = lev(decrease_index(refs_len, ref_idx), hyps_len, ref_idx, hyp_idx)
                b = lev(refs_len, decrease_index(hyps_len, hyp_idx), ref_idx, hyp_idx)
                c = lev(decrease_index(refs_len, ref_idx), decrease_index(hyps_len, hyp_idx), ref_idx, hyp_idx)

                cost, assignment = min(a, b, c)
                cost = cost + 1, assignment
            cache[key] = cost
            return cache[key]

    out = lev(tuple([len(r) for r in refs]), tuple([len(h) for h in hyps]))

    return out


def mimo_matching_v3(
        refs: Reference,
        hyps: Hypothesis,
):
    """
    A Cython implementation of mimo matching
    """
    from .cy_mimo_matching import cy_mimo_matching

    # The Cython implementation uses integers as tokens, so translate ref and
    # hyps to integers
    seen = set()
    for ref in refs:
        for utterance in ref:
            seen.update(set(utterance))
    for channel in hyps:
        seen.update(set(channel))
    mapping = {t: i for i, t in enumerate(seen)}
    refs = [[[mapping[t] for t in utterance] for utterance in ref] for ref in refs]
    hyps = [[mapping[t] for t in channel] for channel in hyps]

    distance, assignment = cy_mimo_matching(refs, hyps)

    return distance, assignment


def mimo_matching_v4(
        refs: Reference,
        hyps: Hypothesis,
):
    """
    A C++ implementation of mimo matching
    """
    from .cy_mimo_matching import cpp_mimo_matching
    try:
        return cpp_mimo_matching(refs, hyps)
    except (MemoryError, OverflowError) as e:
        import math
        memory_size = math.prod([len(hyp) for hyp in hyps]) * math.prod([len(ref) for ref in refs]) * 16
        raise MemoryError(
            f'Not enough memory to compute the MIMO WER. \n'
            f'You are trying to compute the MIMO or ORC WER for {len(refs)} references streams (for ORC always 1) and '
            f'{len(hyps)} hypothesis streams. \n'
            f'The reference streams contain {[len(ref) for ref in refs]} utterances and the hypothesis streams have '
            f'lengths {[len(hyp) for hyp in hyps]} (in words).\n'
            f'The requested computation needs more than {memory_size/10**9:_} G bytes.\n'
            f'MIMO-WER and ORC-WER are designed for 1 and 2 hypothesis streams of 10 minutes of audio.\n'
            f'If you use significantly more, you can encounter OOM errors.'
        ) from e


# Export the recommended version without a v* postfix
mimo_matching = mimo_matching_v4
