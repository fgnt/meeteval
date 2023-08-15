from distutils.extension import Extension

from setuptools import setup, find_packages

from Cython.Build import cythonize
ext_modules = cythonize(
    [
        Extension(
            'meeteval.wer.matching.cy_orc_matching',
            ['meeteval/wer/matching/cy_orc_matching.pyx'],
            extra_compile_args=['-std=c++11'],
            extra_link_args=['-std=c++11'],
        ),
        Extension(
            'meeteval.wer.matching.cy_mimo_matching',
            ['meeteval/wer/matching/cy_mimo_matching.pyx'],
            extra_compile_args=['-std=c++11'],
            extra_link_args=['-std=c++11'],
        ),
        Extension(
            'meeteval.wer.matching.cy_levenshtein',
            ['meeteval/wer/matching/cy_levenshtein.pyx'],
            extra_compile_args=['-std=c++11'],
            extra_link_args=['-std=c++11'],
        ),
     ]
)

extras_require = {
    'cli': [
        'pyyaml',
        'fire',
    ],
    'test': [
        'editdistance',     # Faulty for long sequences, but useful for testing
        'pytest',
        'hypothesis',
        'click',
        'coverage',
        'pytest-cov',
        'ipython',  # IPython.lib.pretty.pprint
        'pyyaml',
    ],
    'all': [  # List only missing from the other lists
        'lazy_dataset',
    ],
}
extras_require['all'] = list(dict.fromkeys(sum(extras_require.values(), [])))

setup(
    name="meeteval",

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.0',

    python_requires=">=3.5",
    author="Thilo von Neumann",
    ext_modules=ext_modules,
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
        'kaldialign',
        'scipy',  # scipy.optimize.linear_sum_assignment
        "typing_extensions; python_version<'3.8'",  # Missing Literal in py37
        "cached_property; python_version<'3.8'",  # Missing functools.cached_property in py37
        'Cython'
    ],
    extras_require=extras_require,
    package_data={'meeteval': ['**/*.pyx', '**/*.h']},  # https://stackoverflow.com/a/60751886
    entry_points={
        'console_scripts': [
            'meeteval-wer=meeteval.wer.__main__:cli'
        ]
    }
)
