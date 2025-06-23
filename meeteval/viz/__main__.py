from pathlib import Path
import collections
import itertools
import json

import meeteval
from meeteval.viz.visualize import AlignmentVisualization
from meeteval.wer.api import _load_texts
import tqdm

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
        file_format='{system_name}/{alignment}/{session_id}.html',
        # file_format='{session_id}_{system_name}_{alignment}.html',
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
            save_name = file_format.format(
                session_id=session_id,
                system_name=system_name,
                alignment=alignment,
            )
            save_path = out / save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            av.dump(save_path)
            # Make the save_path relative to the index.html file such that
            # the links work when moving the folder
            av.data['save_path'] = str(save_name)
            avs.append(av.data)

    create_index_html(avs, out)

def create_index_html(
    data: 'list[dict]',
    out,
    generate_side_by_side='auto',
    system_names=None,
    alignment_types=None,
    session_ids=None,
):
    """
    Args:
        data: A list of visualization data. The info must contain the keys
            - session_id
            - system
            - alignment_type
            - wer
    """
    dump_overview_table(data, out / 'index.html')
    import shutil
    shutil.copy(Path(__file__).parent / 'side_by_side_sync.html', out / 'side_by_side_sync.html')


def index_html(
        folders: 'list[Path]',
        out: Path,
        copy_files: bool = True,
):
    """
    Creates an index.html file with an overview table for multiple visualizations.

    Uses the system names stored in the HTML files prefixed by the folder names.

    WARNING: Does not work with files generated with the --js-debug flag!
    """
    import shutil
    out = Path(out)
    avs = []
    for folder_index, folder in enumerate(folders):
        if copy_files:
            (out / str(folder_index)).mkdir(parents=True, exist_ok=True)
        for f in folder.rglob('*.html'):
            # Skip helper files
            if f.name in ('index.html', 'side_by_side_sync.html'):
                continue

            with open(f, 'r') as text_file:
                html_text = text_file.read()
            try:
                data = get_data_from_html(html_text)
            except json.JSONDecodeError as e:
                print(f'Error extracting data from {f}. Skipping. {e}')
                continue

            # Add the file name to the data
            if copy_files:
                # Copy the file to the output folder
                # Create a new folder for each source folder to make sure that
                # the files are not overwritten
                f_new = out / str(folder_index) / f.name
                shutil.copy(f, f_new)
                data['save_path'] = str(f_new.relative_to(out))
            else:
                data['save_path'] = str(f.absolute())

            # Disambiguate system names by appending folder path
            if 'system_name' in data['info']:
                data['info']['system_name'] = f'{str(folder)}/{data["info"]["system_name"]}'
            else:
                data['info']['system_name'] = str(folder)
            avs.append(data)

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    create_index_html(avs, out)


def html(
        reference,
        hypothesis,
        alignment='tcp',
        regex=None,
        normalizer=None,
        out='viz',
        js_debug=False,
        per_reco_file=None,
        file_format='{system_name}/{alignment}/{session_id}.html',
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
        file_format=file_format,
    )


def cli():
    from meeteval.wer.__main__ import CLI

    class VizCLI(CLI):

        def add_argument(self, command_parser, name, p):
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
            elif name == 'file_format':
                command_parser.add_argument(
                    '--file-format',
                    help='The format of the file paths in the index.html file. '
                         'The following placeholders are available: '
                         '{system_name}, {alignment}, {session_id}.\n'
                         'The default splits by system name and alignment type to make it'
                         'easy to create comparisons between different systems and alignments. '
                         'using "meeteval-viz index_html".',
                )
            elif name == 'folders':
                command_parser.add_argument(
                    '--folders',
                    help='A list of folders containing the visualizations.',
                    nargs='+',
                    type=Path,
                    required=True,
                )
            elif name == 'copy_files':
                command_parser.add_argument(
                    '--copy-files',
                    type=bool,
                    help='If true, the HTML files are copied to the output '
                         'folder and links in index.html are relative. '
                         'If false, the original files are used and links are absolute.'
                )
            else:
                return super().add_argument(command_parser, name, p)

    cli = VizCLI()
    cli.add_command(html)
    cli.add_command(index_html)
    cli.run()


if __name__ == '__main__':
    cli()
