"""Write only inside an explicitly selected local output root.

Network shares are inputs only. No write probe is made on an input share.
"""
from pathlib import Path
import json
import os


def _network_path(path):
    text = str(path)
    if text.startswith(('\\\\', '//')):
        return True
    if os.name == 'nt' and path.drive:
        import ctypes
        return ctypes.windll.kernel32.GetDriveTypeW(path.anchor) == 4
    return False


class OutputRoot:
    def __init__(self, root, source=None):
        self.root = Path(root).resolve()
        self.source = Path(source).resolve() if source else None
        if _network_path(self.root):
            raise ValueError('Output root must not be a network share')
        if self.source and (self.root == self.source or self.root.is_relative_to(self.source)):
            raise ValueError('Output root overlaps read-only source')
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, relative):
        path = (self.root / relative).resolve()
        if path == self.root or not path.is_relative_to(self.root) or _network_path(path):
            raise ValueError('Output escaped selected local root')
        if self.source and (path == self.source or path.is_relative_to(self.source)):
            raise ValueError('Write into read-only source forbidden')
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def json(self, relative, value):
        self.path(relative).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def self_test():
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        parent = Path(temp)
        source = parent / 'input'
        source.mkdir()
        original = source / 'sample.txt'
        original.write_text('immutable fixture', encoding='utf-8')
        before = original.read_bytes()
        out = OutputRoot(parent / 'output', source)
        out.json('valid.json', {'ok': True})
        rejected = []
        for name in ('../input/sample.txt', '../escape.json', str(original)):
            try:
                out.path(name)
            except ValueError:
                rejected.append(name)
            else:
                raise AssertionError('Unsafe destination accepted')
        try:
            OutputRoot(source, source)
        except ValueError:
            pass
        else:
            raise AssertionError('Source selected as output')
        assert original.read_bytes() == before
        return {'passed': True, 'escape_cases_rejected': len(rejected), 'source_unchanged': True, 'uses_real_share': False}


if __name__ == '__main__':
    print(json.dumps(self_test()))
