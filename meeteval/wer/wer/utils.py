def _check_valid_input_files(reference, hypothesis):
    if type(reference) != type(hypothesis):
        raise ValueError(
            f'Both reference and hypothesis must be of the same type, but found {type(reference)} and {type(hypothesis)}.'
        )

    # Check single filename
    if len(reference.filenames()) > 1:
        raise ValueError(
            f'Reference must contain exactly one file, but found {len(reference.filenames())} files.'
        )
    if len(hypothesis.filenames()) > 1:
        raise ValueError(
            f'Hypothesis must contain exactly one file, but found {len(hypothesis.filenames())} files.'
        )

    # Check same filename
    if reference.filenames() != hypothesis.filenames():
        raise ValueError(
            f'Both reference and hypothesis must have the same filename, but found {reference.filenames()} '
            f'and {hypothesis.filenames()}.'
        )