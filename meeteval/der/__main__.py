
def cli():
    from meeteval.wer.__main__ import CLI
    from meeteval.der.md_eval import _md_eval_22

    cli = CLI()

    cli.add_command(_md_eval_22, 'md_eval_22')

    cli.run()


if __name__ == '__main__':
    cli()
