from distutils.extension import Extension
import numpy

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
        Extension(
            'meeteval.wer.matching.cy_time_constrained_orc_matching',
            ['meeteval/wer/matching/cy_time_constrained_orc_matching.pyx'],
            extra_compile_args=['-std=c++11', '-O3'],
            extra_link_args=['-std=c++11'],
        ),
        Extension(
            'meeteval.wer.matching.cy_greedy_combination_matching',
            ['meeteval/wer/matching/cy_greedy_combination_matching.pyx'],
            extra_compile_args=['-std=c++11', '-O3'],
            extra_link_args=['-std=c++11'],
        ),
        Extension(
            'meeteval.wer.matching.cy_time_constrained_mimo_matching',
            ['meeteval/wer/matching/cy_time_constrained_mimo_matching.pyx'],
            extra_compile_args=['-std=c++11'],
            extra_link_args=['-std=c++11'],
        ),
     ]
)

extras_require = {}
extras_require['cli'] = [
    'pyyaml',
    'fire',
    'simplejson',
    'aiohttp',
    'soundfile',
    'tqdm',  # Used in meeteval.viz.__main__.py
    'yattag',  # Used in meeteval.viz.__main__.py
    'platformdirs',  # Used in meeteval.viz.visualize.py
]
extras_require['test'] = [
    'editdistance',     # Faulty for long sequences, but useful for testing
    'pytest',
    'hypothesis',
    'coverage',
    'pytest-cov',
    'paderbox', # paderbox.utils.pretty.pprint
    *extras_require['cli'],
]
extras_require['all'] = [
    'lazy_dataset',
    'ipywidgets',  # Used to provide dropdown menu in ipynb
    *sum(extras_require.values(), []),
]
extras_require = {
    k: list(dict.fromkeys(v))  # Drop duplicates
    for k, v in extras_require.items()}

# Get the long description from the relevant file
try:
    from os import path

    here = path.abspath(path.dirname(__file__))
    with open(path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ''

setup(
    name="meeteval",

    # Versions should comply with PEP440.
    version='0.4.3',

    # The project's main homepage.
    url='https://github.com/fgnt/meeteval/',

    # Choose your license
    license='MIT',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],

    python_requires=">=3.5",
    author="Department of Communications Engineering, Paderborn University",
    author_email='sek@nt.upb.de',
    keywords='speech recognition, word error rate, evaluation, meeting, ASR, WER',
    long_description=long_description,
    long_description_content_type='text/markdown',  # Optional (see note above)

    ext_modules=ext_modules,
    packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
    install_requires=[
        'kaldialign',
        'numpy',
        'scipy',  # scipy.optimize.linear_sum_assignment
        "typing_extensions; python_version<'3.8'",  # Missing Literal in py37
        "cached_property; python_version<'3.8'",  # Missing functools.cached_property in py37
        'Cython',
        'packaging',  # commonly used to compare python versions, e.g., used by jupyter, matplotlib, pytest, ...
    ],
    extras_require=extras_require,
    package_data={'meeteval': ['**/*.pyx', '**/*.h', '**/*.js', '**/*.css', '**/*.html']},  # https://stackoverflow.com/a/60751886
    entry_points={
        'console_scripts': [
            'meeteval-wer=meeteval.wer.__main__:cli',
            'meeteval-der=meeteval.der.__main__:cli',
            'meeteval-viz=meeteval.viz.__main__:cli',
            'meeteval-io=meeteval.io.__main__:cli',
        ]
    },
    include_dirs=[numpy.get_include()],
)
