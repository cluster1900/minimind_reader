"""Read cited source bytes from immutable Git objects, never from a moving checkout."""
import hashlib
from pathlib import Path, PurePosixPath
import re
import subprocess

REPOSITORIES = {'minimind', 'minimind-v', 'minimind-o'}


class SourceSnapshot:
    def __init__(self, root):
        self.root = Path(root)
        self.files = {}

    def read(self, repository, commit, path):
        if repository not in REPOSITORIES or not re.fullmatch('[0-9a-f]{40}', commit):
            raise ValueError(f'Invalid source identity: {repository}@{commit}')
        relative = PurePosixPath(path)
        if relative.is_absolute() or '..' in relative.parts or not path:
            raise ValueError(f'Invalid source path: {path}')
        key = (repository, commit, path)
        if key not in self.files:
            result = subprocess.run(['git', '-C', str(self.root / repository),
                                     'show', f'{commit}:{path}'], capture_output=True)
            if result.returncode:
                raise ValueError(f'Cannot reconstruct cited source {repository}@{commit}:{path}: '
                                 + result.stderr.decode(errors='replace').strip())
            self.files[key] = result.stdout
        return self.files[key]

    def manifest(self):
        return [dict(repository=r, commit=c, path=p, sha256=hashlib.sha256(data).hexdigest())
                for (r, c, p), data in sorted(self.files.items())]
