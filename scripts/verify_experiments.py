#!/usr/bin/env python3
"""Run chapter experiments in isolated processes and preserve a dated input snapshot."""
import datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not __debug__:
        raise RuntimeError('Run experiments without -O or PYTHONOPTIMIZE; numerical assertions must remain enabled')
    folder = ROOT / 'build/final-audit'
    folder.mkdir(parents=True, exist_ok=True)
    report = {'started_at': datetime.datetime.now().astimezone().isoformat(),
              'working_directory': str(ROOT), 'hardware': platform.platform(),
              'status': 'running', 'experiments': [], 'environments': {}}
    for env in ['.venv', '.venv-omni']:
        python = ROOT / env / 'bin/python'
        report['environments'][env] = json.loads(subprocess.check_output(
            [python, '-c', 'import importlib.metadata as m,json; print(json.dumps(sorted('
             '[(d.metadata["Name"],d.version) for d in m.distributions()])))'], text=True))
    paths = [*ROOT.glob('book/**/*.md'), *ROOT.glob('glossary/*.md'),
             *ROOT.glob('experiments/*.py'), ROOT / 'publishing/book.yaml',
             ROOT / 'requirements-book.lock', ROOT / 'requirements-omni.lock']
    report['input_sha256'] = {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}
    repos = {}
    for name in ['minimind', 'minimind-v', 'minimind-o']:
        import os
        repo = Path(os.environ.get('MINIMIND_SOURCE_ROOT', ROOT.parent / 'minimind_resources')) / name
        def git(*args):
            return subprocess.check_output(['git', '-C', repo, *args], text=True).strip()
        repos[name] = {'commit': git('rev-parse', 'HEAD'), 'dirty': bool(git('status', '--porcelain')),
                       'remote': git('remote', 'get-url', 'origin')}
        if repos[name]['dirty']:
            raise ValueError(f'Final experiment audit requires a clean source workspace: {name}')
    report['sources'] = repos
    output = folder / 'experiments.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    for number in range(1, 21):
        script, = ROOT.glob(f'experiments/ch{number:02d}_*.py')
        env = '.venv' if number <= 15 else '.venv-omni'
        command = [str(ROOT / env / 'bin/python'), str(script)]
        started = datetime.datetime.now().astimezone()
        log = folder / f'ch{number:02d}.log'
        with log.open('w') as stream:
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        entry = {'chapter': f'ch{number:02d}', 'command': command,
                 'started_at': started.isoformat(), 'exit_code': result.returncode,
                 'log': str(log.relative_to(ROOT)), 'log_sha256': digest(log)}
        generated = ROOT / f'build/experiments/ch{number:02d}-results.json'
        if result.returncode == 0:
            entry['result'] = json.loads(generated.read_text())
            entry['result_sha256'] = digest(generated)
        report['experiments'].append(entry)
        if result.returncode:
            report['status'] = 'failed'
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f'ch{number:02d}: {"passed" if result.returncode == 0 else "FAILED"}', flush=True)
        if result.returncode:
            print(log.read_text()[-6000:])
            sys.exit(result.returncode)
    report['status'] = 'passed'
    report['completed_at'] = datetime.datetime.now().astimezone().isoformat()
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
