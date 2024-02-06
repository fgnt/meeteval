from distutils.extension import Extension
from pathlib import Path

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
     ]
)

extras_require = {
    'cli': [
        'pyyaml',
        'fire',
        'simplejson',
        'aiohttp',
        'soundfile',
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
        'simplejson',
        'aiohttp',
        'soundfile',
    ],
    'all': [  # List only missing from the other lists
        'lazy_dataset',
    ],
}
extras_require['all'] = list(dict.fromkeys(sum(extras_require.values(), [])))

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
    version='0.2.1',

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
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
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
        'scipy',  # scipy.optimize.linear_sum_assignment
        "typing_extensions; python_version<'3.8'",  # Missing Literal in py37
        "cached_property; python_version<'3.8'",  # Missing functools.cached_property in py37
        'Cython'
    ],
    extras_require=extras_require,
    package_data={'meeteval': ['**/*.pyx', '**/*.h', '**/*.js', '**/*.css']},  # https://stackoverflow.com/a/60751886
    entry_points={
        'console_scripts': [
            'meeteval-wer=meeteval.wer.__main__:cli',
            'meeteval-der=meeteval.der.__main__:cli',
        ]
    }
)
