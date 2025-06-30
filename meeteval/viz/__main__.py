import os
from pathlib import Path
import collections
import itertools
import json

import meeteval
from meeteval.viz.visualize import AlignmentVisualization
from meeteval.wer.api import _load_texts
import tqdm
import logging

from meeteval.viz.overview_table import dump_overview_table


def get_data_from_html(html_text: str):
    """
    Extracts the JSON visualization data from the html text by searching for
    the call to the function `alignment_visualization` and extracting the 
    first argument, which is always on a new line.
    """
    identifier = "if (typeof d3 !== 'undefined') alignment_visualization("
    data_start = html_text.find(identifier) + len(identifier) + 1
    data_end = html_text.find('\n', data_start) - 1 # Remove comma at end of line
    data = json.loads(html_text[data_start:data_end])
    return data



def create_viz_folder(
        reference,
        hypothesiss,
        out,
        alignments='tcp',
        regex=None,
        normalizer=None,
        js_debug=False,
        per_reco_file=None,
):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    if isinstance(alignments, str):
        alignments = alignments.split(',')

    if per_reco_file is not None:
        assert len(alignments) == len(per_reco_file), alignments

        error_rate_classes = {
            'tcp': meeteval.wer.CPErrorRate,
            'cp': meeteval.wer.CPErrorRate,
            'tcorc': meeteval.wer.OrcErrorRate,
            'orc': meeteval.wer.OrcErrorRate,
            'greedy_orc': meeteval.wer.OrcErrorRate,
            'greedy_tcorc': meeteval.wer.OrcErrorRate,
            'greedy_dicp': meeteval.wer.DICPErrorRate,
            'greedy_ditcp': meeteval.wer.DICPErrorRate,
        }

        def load_per_reco_file(alignment, f):
            from meeteval.wer.__main__ import _load

            error_rate_cls = error_rate_classes[alignment]

            return {
                session_id: error_rate_cls.from_dict(pr)
                for session_id, pr in _load(Path(f)).items()
            }
        per_reco = {
            alignment: load_per_reco_file(alignment, f)
            for alignment, f in zip(alignments, per_reco_file)
        }
    else:
        per_reco = collections.defaultdict(lambda: collections.defaultdict(lambda: None))

    avs = []
    for (system_name, hypothesis), alignment in tqdm.tqdm(list(itertools.product(
            hypothesiss.items(),
            alignments,
    ))):

        r, h = _load_texts(
            reference, hypothesis, regex=regex,
            normalizer=normalizer,
        )

        r = r.sorted('start_time')
        h = h.sorted('start_time')

        r = r.groupby('session_id')
        h = h.groupby('session_id')

        session_ids = set(r.keys()) & set(h.keys())
        xor = set(r.keys()) ^ set(h.keys())
        if xor:
            print(f'Ignore {xor}, because they are not available in reference and hypothesis.')

        for session_id in tqdm.tqdm(session_ids):
            av = AlignmentVisualization(
                r[session_id],
                h[session_id],
                alignment=alignment,
                js_debug=js_debug,
                sync_id=1,
                precomputed_error_rate=per_reco[alignment][session_id],
                system_name=system_name,
            )   
            save_name = _get_av_file_path(av.data)
            save_path = out / save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            av.dump(save_path)
            # Make the save_path relative to the index.html file such that
            # the links work when moving the folder
            av.data['save_path'] = str(save_name)
            av.data['absolute_path'] = save_path.absolute()
            avs.append(av.data)

    dump_overview_table(avs, out / 'index.html')
    _copy_side_by_side(out)


def _copy_side_by_side(out: Path):
    import shutil
    shutil.copy(Path(__file__).parent / 'side_by_side_sync.html', out / 'side_by_side_sync.html')


def _get_av_file_path(av_data):
    return (
        f'{av_data["info"]["session_id"]}'
        f'_{av_data["info"]["system_name"]}'
        f'_{av_data["info"]["alignment_type"]}.html'
    )

def index_html(
        folders: 'list[Path]',
        out: 'Path | str' = 'viz/index.html',
        copy: 'bool | str | Path' = False,
):
    """
    Creates an index.html file with an overview table for multiple visualizations.

    Uses the system names stored in the HTML files prefixed by the folder names.

    WARNING: Does not work with files generated with the --js-debug flag!

    Examples:
    
        # Re-generate the index_html file for a visualization folder
        meeteval-viz index_html viz

        # Generate an overview table for a sub-set of visualizations
        meeteval-viz index_html viz/*_tcp.html --out viz/index_tcp.html

        # Create a sharable folder with copies of the original visualizations
        meeteval-viz index_html viz/*_tcp.html --out share --copy

    """
    import shutil
    out = Path(out)

    if out.suffix != '.html':
        out = out / 'index.html'

    if copy:
        if isinstance(copy, (str, Path)):
            copy = Path(copy)
        else:
            copy = out.parent

    avs = []
    for folder in folders:
        folder = Path(folder)
        if folder.is_dir():
            folder = folder.glob('*.html')
        else:
            folder = [folder]
        for f in folder:
            # Skip helper files
            if f.name in ('index.html', 'side_by_side_sync.html'):
                continue

            with open(f, 'r') as text_file:
                html_text = text_file.read()
            try:
                data = get_data_from_html(html_text)
            except json.JSONDecodeError as e:
                logging.warning(f'Error extracting data from {f}. Skipping. {e}')
                continue
    
            data['absolute_path'] = f.absolute()

            avs.append(data)
    
    def resolve_system_names(avs):
        """
        Prepends parts of the file paths that differ between file storage paths
        to the system name to disambiguate visualizations for systems
        with the same name but in different folders
        """
        filenames = [av['absolute_path'].parts for av in avs]
        prefix = os.path.commonprefix(filenames)
        # Group by the first folder name that differs. Ignore filenames
        for group, grouped_avs in itertools.groupby(
            avs, 
            key=lambda x: x['absolute_path'].parts[len(prefix)] 
                if len(prefix) < len(x['absolute_path'].parts) - 1 
                else None
        ):
            grouped_avs = list(grouped_avs)
            if group is not None:
                resolve_system_names(grouped_avs)
                for av in grouped_avs:
                    av['info']['system_name'] = f'{group}_{av["info"]["system_name"]}'
    avs = sorted(avs, key=lambda x: x['absolute_path'])
    resolve_system_names(avs)
        
    if copy:
        copy.mkdir(parents=True, exist_ok=True)

    for av in avs:
        if copy:
            f_new = copy / _get_av_file_path(av)
            av.pop('path', None)
            shutil.copy(av['absolute_path'], f_new)
            av['absolute_path'] = f_new
        
        av['save_path'] = os.path.relpath(av['absolute_path'], out.parent)

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_overview_table(avs, out)
    _copy_side_by_side(out.parent)


def html(
        reference,
        hypothesis,
        alignment='tcp',
        regex=None,
        normalizer=None,
        out='viz',
        js_debug=False,
        per_reco_file=None,
):
    """
    Creates a visualization of the alignment between reference and hypothesis for the specified WER algorithm.

    The visualization is created in two steps.
    
    First, compute the WER and assignment, i.e. the mapping of utterances/segments to streams. Any WER algorithm 
    from meeteval can be used for this. Depending on the algorithm, the labels of the reference or 
    hypothesis utterances or streams are modified.
    
    Second, compute the alignment, i.e. the matching of words between reference and hypothesis (insertion, 
    deletion, substitution). This is done with a time-constrained algorithm if the assignment was 
    time-constrained, otherwise with a "classical" unconstrained algorithm.
    """
    def prepare(i: int, h: str):
        if ':' in h and not Path(h).exists():
            # inspired by tensorboard from the --logdir_spec argument.
            name, path = h.split(':', maxsplit=1)
            return name, path
        else:
            if len(hypothesis) > 1:
                return f'System {i}', h
            else:
                return f'System', h

    assert len(reference) == 1, (len(reference), 'At the moment only shared reference is supported.')

    hypothesis = dict([
        prepare(i, h)
        for i, h in enumerate(hypothesis)
    ])

    create_viz_folder(
        reference=reference,
        hypothesiss=hypothesis,
        out=out,
        alignments=alignment,
        regex=regex,
        normalizer=normalizer,
        js_debug=js_debug,
        per_reco_file=per_reco_file,
    )


def cli():
    from meeteval.wer.__main__ import CLI

    class VizCLI(CLI):

        def add_argument(self, command_parser, name, p, command_name):
            if name == 'alignment':
                command_parser.add_argument(
                    '--alignment',
                    choices=['tcp', 'cp', 'orc', 'greedy_orc', 'tcorc', 'greedy_tcorc', 'greedy_dicp', 'greedy_ditcp'],
                    nargs='+',
                    help='Specifies the algorithm used to obtain the alignment. \n'
                         'Multiple alignments can be specified to generate multiple visualizations with a single '
                         'merged overview table and side-by-side views.'
                )
            elif name == 'hypothesis':
                command_parser.add_argument(
                    '-h', '--hypothesis',
                    help='Hypothesis file(s) in SegLST, STM or CTM format. '
                         'Multiple files can be provided for a side by side view. '
                         'Optionally prefixed with system name, e.g. mysystem:/path/to/hyp.stm',
                    nargs='+', action=self.extend_action,
                    required=True,
                )
            elif name == 'js_debug':
                command_parser.add_argument(
                    '--js-debug',
                    action='store_true',
                    help='Add a debug flag to the HTML output to enable debugging in the browser.'
                )
            elif name == 'per_reco_file':
                command_parser.add_argument(
                    '--per-reco-file',
                    help='A precomputed per-reco file. Loads the WER and (stream) '
                         'assignment information from this file instead of computing it. '
                         'If supplied, the number of files must match the number of alignments specified '
                         'with --alignment.',
                    default=None,
                    nargs='+',
                )
            elif name == 'folders':
                command_parser.add_argument(
                    'folders',
                    help='A list of folders containing visualization html files. '
                         'Alternatively, files can be given explicitly which '
                         'allows for shell wildcards for filtering, e.g. '
                         '`viz/*_tcp.html`.',
                    nargs='+',
                    type=Path,
                )
            elif name == 'copy':
                command_parser.add_argument(
                    '--copy',
                    nargs='?',
                    const=True,
                    help='Copy the visualization HTML files to the specified directory. '
                         'If the --copy option is used without an argument, it defaults '
                         'to the parent directoy of the generated index html '
                         'specified with `--out`.'
                )
            elif name == 'out' and command_name == 'index_html':
                command_parser.add_argument(
                    '-o', '--out',
                    help='Path of the generated html file or a folder in which to '
                         'create the index.html file.'
                )
            else:
                return super().add_argument(command_parser, name, p, command_name)

    cli = VizCLI()
    cli.add_command(html)
    cli.add_command(index_html)
    cli.run()


if __name__ == '__main__':
    cli()
