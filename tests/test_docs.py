import pytest
import sys

if sys.version_info < (3, 11):
    pytest.skip(reason='algorithms.md requires Python 3.11+', allow_module_level=True)

import re
from pathlib import Path
import ast

MEETEVAL_ROOT = Path(__file__).parent.parent

FENCED_CODE_BLOCK_REGEX = re.compile(r'```([^`\n]*)?\n((?:.|\n)*?)\n```')

# List of language blocks that are not tested
LANG_BLACKLIST = ['shell', 'bibtex', '']

# Markdown files for which the context is kept between code blocks
KEEP_CONTEXT = ['doc/algorithms.md']


@pytest.fixture(scope='session')
def global_state():
    """Used to track global state across code blocks in the files listed in `KEEP_CONTEXT`."""
    return {}


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

    >>> split_code_block_comment_output(r'''
    ... # this is a comment
    ... print('hello')
    ... print('world')
    ... # hello
    ... # world
    ...
    ... # Here starts the second block
    ... a = 2
    ...
    ... print(
    ...     a
    ... )
    ... # 2
    ... ''')
    [("\\n# this is a comment\\nprint('hello')\\nprint('world')", ' hello\\n world', 0), ('\\n# Here starts the second block\\na = 2\\n\\nprint(\\n    a\\n)', ' 2', 6)]
    """
    c = ast.parse(code)
    lines = code.splitlines()
    last_match = 0
    blocks = []
    l = 0
    for s in c.body:
        if l > s.end_lineno:
            continue

        l = s.end_lineno

        if l < len(lines) and lines[l].startswith('#'):
            # Collect any lines that follow directly and start with a #
            output = []
            
            while l < len(lines) and lines[l].startswith('#'):
                output.append(lines[l][1:])
                l += 1
            blocks.append(('\n'.join(lines[last_match:s.end_lineno]), '\n'.join(output), last_match))
            last_match = l
    if last_match < len(lines):
        blocks.append(('\n'.join(lines[last_match:]), '', last_match))
    return blocks


def exec_with_source(code, filename, lineno, globals_=None, locals_=None):
    """
    Like `compile` followed by `exec`, but sets the correct line number for the code block.
    This is required for correct traceback display.
    Captures stdout and returns it as a string.
    """
    compiled = ast.parse(code, str(filename), 'exec')
    ast.increment_lineno(compiled, lineno)
    compiled = compile(compiled, str(filename), 'exec', optimize=0)
    from io import StringIO
    from contextlib import redirect_stdout

    f = StringIO()
    with redirect_stdout(f):
        exec(compiled, globals_, locals_)
    return f.getvalue()


@pytest.mark.parametrize(
        ('filename', 'codeblock'), 
        [
            (str(filename.relative_to(MEETEVAL_ROOT)), codeblock)
            for filename in MEETEVAL_ROOT.glob('**/*.md')
            if 'build' not in str(filename) and '/.' not in str(filename)
            for codeblock in get_fenced_code_blocks(filename.read_text())
        ]
)
def test_docs(filename, codeblock, global_state, monkeypatch):
    """Run fenced code blocks in markdown files in the MeetEval repository."""
    # Some code blocks in the readme file must run in the meeteval root directory
    # because they access the example files in `MEETEVAL_ROOT/example_files`
    monkeypatch.chdir(MEETEVAL_ROOT)

    lang, code, lineno = codeblock
    if lang in LANG_BLACKLIST:
        return

    try:
        if lang == 'python':
            if filename in KEEP_CONTEXT:
                globals_ = global_state.setdefault(filename, {})
            else:
                globals_ = {}
            for code, expected_output, line_offset in split_code_block_comment_output(code):
                output = exec_with_source(code, str(filename), lineno + line_offset, globals_)
                if expected_output is not None:
                    # Check that the output is equal to the expected output, but we want to ignore whitespace
                    # for formatting / clarity reasons.
                    # This is a very basic check that ignores all whitespace, but it should be
                    # sufficient for most cases.
                    output_ = output.replace(' ', '').replace('\n', '')
                    expected_output_ = expected_output.replace(' ', '').replace('\n', '')
                    if output_ != expected_output_:
                        raise AssertionError(
                            f'Output mismatch in {filename} at line {lineno + line_offset}:\n'
                            f'Output: {output}\nExpected: {expected_output}'
                        )
        elif lang == 'STM':
            # Test if the STM code block is valid.
            import meeteval
            meeteval.io.STM.parse(code)
        else:
            raise ValueError(f'Unsupported language: {lang}')
    except Exception:
        print(f'Error in {lang} code block:\n', code)
        raise
