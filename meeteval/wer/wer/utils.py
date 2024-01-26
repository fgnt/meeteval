from meeteval.io import SegLST


def check_single_filename(reference: SegLST, hypothesis: SegLST):
    try:
        reference_session_ids = reference.unique('session_id')
    except KeyError:
        reference_session_ids = set()
    try:
        hypothesis_session_ids = hypothesis.unique('session_id')
    except KeyError:
        hypothesis_session_ids = set()

    if len(reference_session_ids) > 1:
        raise ValueError(
            f"Expected a single session ID, but got "
            f"{len(reference_session_ids)} in the reference: "
            f"{reference_session_ids}"
        )

    if len(hypothesis_session_ids) > 1:
        raise ValueError(
            f"Expected a single session ID, but got "
            f"{len(hypothesis_session_ids)} in the hypothesis: "
            f"{hypothesis_session_ids}"
        )

    if len(reference) > 0 and len(hypothesis) > 0 and reference_session_ids != hypothesis_session_ids:
        raise ValueError(
            f"Expected the same session ID in reference and hypothesis, but "
            f"got {reference_session_ids} and {hypothesis_session_ids}"
        )
