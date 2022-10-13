import typing

Utterance = typing.Iterable[typing.Hashable]
Reference = typing.List[Utterance]
Hypothesis = typing.List[typing.Iterable[typing.Hashable]]


def levenshtein_distance(ref, hyp):
    from .cy_mimo_matching import cy_levenshtein_distance
    return cy_levenshtein_distance(ref, hyp)


def mimo_matching_v1():
    """
    A brute-force implmentation of the MIMO matching
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
        # print(ref_idx, refs_len)
        if ref_idx is not None and (
                refs_len[ref_idx] and refs[ref_idx][refs_len[ref_idx] - 1] == change_token or refs_len[ref_idx] == 0 and any(
            r > 0 for r in refs_len)):
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
            # print(indices)
            # print(tmp)
            # print(index)

            # num_utts = tuple(
            #     [Counter(refs[i][:refs_len[i]]).get('#', 0) if refs_len[i] > 0 else 0 for i in range(len(refs))])
            #
            # if num_utts not in assignments or assignments[num_utts][0] > tmp[index]:
            #     assignments[num_utts] = (tmp[index], indices[index], indices, tmp,
            #                              [[refs[i][:refs_len[i]] if refs_len[i] > 0 else '' for i in range(len(refs))],
            #                               [hyps[i][:hyps_len[i]] if hyps_len[i] > 0 else '' for i in range(len(hyps))]])
            # print(assignments[ref_len])
            # print('change token', [refs[i][:refs_len[i]] if refs_len[i] > 0 else '' for i in range(len(refs))], [hyps[i][:hyps_len[i]] if hyps_len[i] > 0 else '' for i in range(len(hyps))], tmp[index], key)
            return tmp[index]
        else:
            # No channel switch. Perform normal lev update along slice of tensor

            assert refs_len[ref_idx] >= 0 and hyps_len[hyp_idx] >= 0

            # Edge case: initialize
            if refs_len[ref_idx] == 0:
                if all(r == 0 for r in refs_len):
                    cost = sum(hyps_len)
                else:
                    cost = 1 + lev(refs_len, decrease_index(hyps_len, hyp_idx), ref_idx, hyp_idx)

            # Empy hypothesis: only deletion possible
            elif hyps_len[hyp_idx] == 0:
                cost = 1 + lev(decrease_index(refs_len, ref_idx), hyps_len, ref_idx, hyp_idx)

            # Correct match: Diagonal update
            elif refs[ref_idx][refs_len[ref_idx] - 1] == hyps[hyp_idx][hyps_len[hyp_idx] - 1]:
                cost = lev(decrease_index(refs_len, ref_idx), decrease_index(hyps_len, hyp_idx), ref_idx, hyp_idx)

            # No correct match: Pick best from insertion, deletion and substitution
            else:
                a = lev(decrease_index(refs_len, ref_idx), hyps_len, ref_idx, hyp_idx)
                b = lev(refs_len, decrease_index(hyps_len, hyp_idx), ref_idx, hyp_idx)
                c = lev(decrease_index(refs_len, ref_idx), decrease_index(hyps_len, hyp_idx), ref_idx, hyp_idx)

                cost = 1 + min(a, b, c)
            # print([refs[i][:refs_len[i]] if refs_len[i] > 0 else '' for i in range(len(refs))], [hyps[i][:hyps_len[i]] if hyps_len[i] > 0 else '' for i in range(len(hyps))], cost, case)
            cache[key] = cost
            return cache[key]

    out = lev(tuple([len(r) for r in refs]), tuple([len(h) for h in hyps]))

    #     for k, v in assignments.items():
    #         print(k, v)

    # Backtrack through assignments to find the overall assignment
    # num_utts = [Counter(refs[i]).get('#', 0) for i in range(len(refs))]
    #
    # final_assignment = []
    # while sum(num_utts) != 0:
    #     assignment = assignments[tuple(num_utts)]
    #     final_assignment.append(assignment[1])
    #     num_utts[assignment[1][0]] -= 1

    return out, None # final_assignment[::-1]


def mimo_matching_v3(
        refs: Reference,
        hyps: Hypothesis,
):
    import itertools
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


