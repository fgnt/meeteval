import pytest
import re
from pathlib import Path
import ast

MEETEVAL_ROOT = Path(__file__).parent.parent

FENCED_CODE_BLOCK_REGEX = re.compile(r'```([^`\n]*)?\n((?:.|\n)*?)\n```')

# List of language blocks that are not tested
LANG_BLACKLIST = ['shell', 'bibtex', '']

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

    Returns a list of tuples with the code block and the expected output.
    The print statement in the code block is replaced with `__output = ` so
    that the result can be inspected after `exec`.
    """
    c = ast.parse(code)
    lines = code.splitlines()
    last_match = 0
    blocks = []
    for s in c.body:
        # If we parsed a print statement at the root level
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and isinstance(s.value.func, ast.Name) and s.value.func.id == 'print':
            # Collect any lines that follow directly and start with a #
            output = []
            l = s.end_lineno
            while l < len(lines) and lines[l].startswith('#'):
                output.append(lines[l][1:])
                l += 1
            blocks.append(('\n'.join(lines[last_match:s.end_lineno]), '\n'.join(output), last_match))
            last_match = l
    blocks.append(('\n'.join(lines[last_match:]), '', last_match))
    return blocks


def exec_with_source(code, filename, lineno, globals_=None, locals_=None):
    """
    Like `compile` followed by `exec`, but sets the correct line number for the code block.
    This is required for correct traceback display.
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
            for codeblock in get_fenced_code_blocks(filename.read_text())
        ]
)
def test_readme(filename, codeblock, global_state):
    """Run fenced code blocks in markdown files in the MeetEval repository."""
    import os
    os.chdir(MEETEVAL_ROOT)
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
