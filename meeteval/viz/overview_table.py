

import collections
from pathlib import Path
import meeteval
from meeteval.viz.visualize import AlignmentVisualization, dumps_json


def generate_overview_table(data: 'list[dict | AlignmentVisualization]') -> str:
    """
    Generates an overview table from the provided list of alignment 
    visualizations.

    Args:
        - data: A list of `AlignmentVisualization` objects or visualization 
            data, as returned by `AlignmentVisualization.data`. 
            The info must contain the keys
            - session_id
            - system_name
            - alignment_type
            - wer
    """
    data = [
        entry.data if isinstance(entry, AlignmentVisualization) else entry 
        for entry in data
    ]

    # Group by system and alignment type to get format ["<system>"]["<alignment_type>"]["<session_id>"]
    # The defaultdicts ensure that we get an empty cell when the data is missing
    avs = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
    for d in data:
        avs[d['info']['system_name']][d['info']['alignment_type']][d['info']['session_id']] = d
    
    def get_average_wer(data):
        error_rate = meeteval.wer.combine_error_rates(*[
            meeteval.wer.ErrorRate.from_dict(d['info']['wer'])
            for d in data
        ]).error_rate
        return error_rate
  
    html_data = []
    for system_name, alignments in avs.items():
        for alignment_type, sessions in alignments.items():
            html_data.append({
                'system_name': system_name,
                'metric': f'{alignment_type}WER',
                'average': get_average_wer(sessions.values()),
                'session_details': {
                    session_id: {
                        'error_rate': session_data['info']['wer']['error_rate'],
                        'link': str(session_data['save_path']),
                        'absolute_path': str(session_data['absolute_path']),
                    }
                    for session_id, session_data in sessions.items()
                }
            })
    
    indent = '            '
    html_data = (
        f'[\n{indent}    ' +
        f',\n{indent}    '.join([dumps_json(row, indent=None) for row in html_data]) +
        f',\n{indent}]'
    )
    html = (Path(__file__).parent / 'overview_table.html').read_text()

    import re
    html, n = re.subn(f'// DATA START((.|\n)*)// DATA END', f'const data = {html_data};', html)
    assert n == 1, (html, n)
    return html

def dump_overview_table(data, file: Path):
    html = generate_overview_table(data)
    with open(file, 'w') as f:
        f.write(html)
