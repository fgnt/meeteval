from meeteval.wer.__main__ import _save_results


def md_eval_22(
        reference,
        hypothesis,
        average_out='{parent}/{stem}_md_eval_22.json',
        per_reco_out='{parent}/{stem}_md_eval_22_per_reco.json',
        collar=0,
        exclude_overlap=False,
        regex=None,
        uem=None,
):
    from meeteval.der.api import md_eval_22
    results = md_eval_22(
        reference,
        hypothesis,
        collar=collar,
        regex=regex,
        exclude_overlap=exclude_overlap,
        uem=uem,
    )
    _save_results(results, hypothesis, per_reco_out, average_out)


def cli():
    from meeteval.wer.__main__ import CLI

    class DerCLI(CLI):
        def add_argument(self, command_parser, name, p):
            if name == 'exclude_overlap':
                command_parser.add_argument(
                    '-1', '--exclude-overlap',
                    action='store_true',
                    help='Limits scoring to single-speaker regions. '
                         'This option appends `-1` to the md-eval-22.pl '
                         'command.'
                )
            else:
                super().add_argument(command_parser, name, p)

    cli = DerCLI()

    cli.add_command(md_eval_22)

    cli.run()


if __name__ == '__main__':
    cli()
