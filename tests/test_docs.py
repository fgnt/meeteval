import pytest
import re
from pathlib import Path
import ast

FENCED_CODE_BLOCK_REGEX = re.compile(r'```([^`\n]*)?\n((?:.|\n)*?)\n```')

# List of language blocks that are not tested
LANG_BLACKLIST = ['shell', 'bibtex', '']


def get_fenced_code_blocks(markdown_string: str):
    """
    Returns a list of tuples (lang, code, lineno) for each fenced code block in the markdown string.
    lineno corresponds to the line where the code starts (after the opening ```).
    """
    def get_lineno(offset):
        return markdown_string[:offset].count('\n') + 1

    return [
        (m.group(1), m.group(2), get_lineno(m.span()[0]))
        for m in FENCED_CODE_BLOCK_REGEX.finditer(markdown_string)
    ]


def split_code_block_comment_output(code):
    """Splits a code block where a line starts with `print` and the following
    line is a comment.
    The comment is expected to be the output of the print statement.

    Returns a list of tuples with the code block and the expected output.
    The print statement in the code block is replaced with `__output = ` so
    that the result can be inspected after `exec`.
    """
    def get_line_offset(offset):
        return code[:offset].count('\n')

    last_match = 0
    blocks = []
    for m in list(re.finditer(r'print\((.*)\)\n((?:#.*\n)*#.*)', code)):
        expected_output = '\n'.join(s[2:] for s in m.group(2).split('\n'))

        blocks.append((
            code[last_match:m.span()[0]] + '__output = ' + m.group(1),
            expected_output,
            get_line_offset(last_match)
        ))
        last_match = m.span()[1]
    if last_match < len(code) - 1:
        blocks.append((code[last_match:], None, get_line_offset(last_match)))
    return blocks


def exec_with_source(code, filename, lineno, globals_=None, locals_=None):
    """
    Like `compile` followed by `exec`, but sets the correct line number for the code block.
    This is required for correct traceback display.
    """
    compiled = ast.parse(code, str(filename), 'exec')
    ast.increment_lineno(compiled, lineno)
    compiled = compile(compiled, str(filename), 'exec', optimize=0)
    exec(compiled, globals_, locals_)


README = Path(__file__).parent.parent / 'README.md'


@pytest.mark.parametrize('codeblock', get_fenced_code_blocks(README.read_text()))
def test_readme(codeblock):
    """Run fenced code blocks in readme isolated"""
    import os
    os.chdir(Path(__file__).parent.parent)
    lang, code, lineno = codeblock
    if lang in LANG_BLACKLIST:
        return

    try:
        if lang == 'python':
            globals_ = {}
            for code, expected_output, line_offset in split_code_block_comment_output(code):
                exec_with_source(code, str(README), lineno + line_offset, globals_)
                output = str(globals_.pop('__output', None))
                if expected_output is not None:
                    # Check that the output is equal to the expected output, but we want to ignore whitespace
                    # for formatting / clarity reasons.
                    # This is a very basic check that ignores all whitespace, but it should be
                    # sufficient for most cases.
                    output_ = output.replace(' ', '').replace('\n', '')
                    expected_output_ = expected_output.replace(' ', '').replace('\n', '')
                    assert output_ == expected_output_, f'Output mismatch: {output} != {expected_output}'
        elif lang == 'STM':
            # Test if the STM code block is valid.
            import meeteval
            meeteval.io.STM.parse(code)
        else:
            raise ValueError(f'Unsupported language: {lang}')
    except Exception:
        print(f'Error in {lang} code block:\n', code)
        raise
