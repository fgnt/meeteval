from meeteval.wer.__main__ import _save_results


def md_eval_22(
        reference,
        hypothesis,
        average_out='{parent}/{stem}_md_eval_22.json',
        per_reco_out='{parent}/{stem}_md_eval_22_per_reco.json',
        collar=0,
        regions='all',
        regex=None,
        uem=None,
):
    from meeteval.der.api import md_eval_22
    results = md_eval_22(
        reference,
        hypothesis,
        collar=collar,
        regex=regex,
        regions=regions,
        uem=uem,
    )
    _save_results(results, hypothesis, per_reco_out, average_out)


def cli():
    from meeteval.wer.__main__ import CLI

    class DerCLI(CLI):
        def add_argument(self, command_parser, name, p):
            if name == 'regions':
                command_parser.add_argument(
                    '--regions',
                    choices=['all', 'nooverlap'],
                    help='Only evaluate the selected region type.\n'
                         'Choices:\n'
                         '- all: Evaluate the whole recording.\n'
                         '- nooverlap: Evaluate only non-overlapping regions.'
                )
            elif name == 'collar':
                command_parser.add_argument(
                    '--collar', type=self.positive_number,
                    help='The no-score zone around reference speaker segment '
                         'boundaries. Speaker Diarization output is not '
                         'evaluated within +/- collar seconds of a reference '
                         'speaker segment boundary. '
                         'Be aware that a nonzero collar can distort the '
                         'results for overlapped speech because the collar '
                         'tends to exclude the most challenging regions. This '
                         'results in an overoptimistic score.'
                )
            else:
                super().add_argument(command_parser, name, p)

    cli = DerCLI()

    cli.add_command(md_eval_22)

    cli.run()


if __name__ == '__main__':
    cli()
