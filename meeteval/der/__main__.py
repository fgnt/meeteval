from meeteval.wer.__main__ import _save_results


def md_eval_22(
        reference,
        hypothesis,
        average_out='{parent}/{stem}_md_eval_22.json',
        per_reco_out='{parent}/{stem}_md_eval_22_per_reco.json',
        collar=0,
        regex=None,
        uem=None,
):
    from meeteval.der.api import md_eval_22
    results = md_eval_22(
        reference,
        hypothesis,
        collar=collar,
        regex=regex,
        uem=uem,
    )
    _save_results(results, hypothesis, per_reco_out, average_out)


def cli():
    from meeteval.wer.__main__ import CLI

    cli = CLI()

    cli.add_command(md_eval_22)

    cli.run()


if __name__ == '__main__':
    cli()
