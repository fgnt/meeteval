from setuptools import setup

from Cython.Build import cythonize
ext_modules = cythonize(
    [
        'meeteval/wer/matching/cy_orc_matching.pyx',
        'meeteval/wer/matching/cy_mimo_matching.pyx',
        'meeteval/wer/matching/cy_levenshtein.pyx',
     ]
)

setup(
    name="meeteval",
    python_requires=">=3.5",
    author="Thilo von Neumann",
    ext_modules=ext_modules,
    packages=["meeteval"],
    install_requires=[
        'editdistance',
        'kaldialign',
        'scipy',  # scipy.optimize.linear_sum_assignment
        "typing_extensions; python_version<'3.8'",  # Missing Literal in py37
        "cached_property; python_version<'3.8'",  # Missing functools.cached_property in py37
        'cython'
    ],
    extras_require={
        'cli': [
            'pyyaml',
        ],
        'test': [
            'pytest',
            'hypothesis',
            'click',
            'coverage',
            'pytest-cov',
            'ipython',  # IPython.lib.pretty.pprint
            'pyyaml',
        ]
    },
    entry_points={
        'console_scripts': [
            'meeteval-wer=meeteval.wer.__main__:cli'
        ]
    }
)
