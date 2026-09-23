"""Validate notebook placement, catalog coverage, JSON, and active Python cells."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    paths = sorted(ROOT.rglob('*.ipynb'))
    paths = [p for p in paths if not any(x in p.parts for x in ('.git', '.venv', '.ipynb_checkpoints'))]
    assert paths, 'No notebooks found'
    active_index = (ROOT / 'notebooks/README.md').read_text()
    archive_index = (ROOT / 'notebooks/archive/README.md').read_text()
    active_count = 0
    for path in paths:
        relative = path.relative_to(ROOT)
        assert relative.parts[0] == 'notebooks', f'Notebook outside notebooks/: {relative}'
        archived = 'archive' in relative.parts
        base = ROOT / ('notebooks/archive' if archived else 'notebooks')
        link = f']({path.relative_to(base).as_posix()})'
        assert link in (archive_index if archived else active_index), f'Missing catalog entry: {relative}'
        source = path.read_text()
        notebook = json.loads(source)
        assert notebook['nbformat'] == 4, relative
        assert not re.search(r'hf_[A-Za-z0-9]{24,}', source), f'Embedded HF token: {relative}'
        # Historical snapshots may use old IPython syntax; keep them intact.
        if archived:
            continue
        active_count += 1
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            lines = []
            for line in ''.join(cell['source']).splitlines():
                stripped = line.lstrip()
                lines.append(line[:len(line)-len(stripped)] + 'pass'
                             if stripped.startswith(('!', '%')) else line)
            compile('\n'.join(lines), f'{relative}:cell_{index}', 'exec')
    print(f'Validated {len(paths)} notebooks: {active_count} current, {len(paths)-active_count} archived.')


if __name__ == '__main__':
    main()
