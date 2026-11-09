"""Run an offline experiment with affinity, memory, time and private-output limits."""
import argparse
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--name', required=True)
    parser.add_argument('--seconds', type=int, default=259200)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--memory-gib', type=int, default=32)
    parser.add_argument('--cache-root', help='Private execution directory measured against the 10GiB cap')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    out = Path(args.output).expanduser().resolve()
    module = Path(__file__).resolve().parent
    course = next((p for p in module.parents if (p / '课程大纲.md').is_file()), None)
    forbidden = [module] + ([course] if course is not None else [])
    if any(out == p or p in out.parents for p in forbidden) or str(out).startswith(('//', '\\\\')):
        raise ValueError('Experiment output must be a private local directory outside the course')
    if not (1 <= args.threads <= 8 and 1 <= args.memory_gib <= 32 and 1 <= args.seconds <= 259200):
        raise ValueError('Resource request exceeds approved maximum')
    if Path(args.name).name != args.name:
        raise ValueError('name must be a single filename component')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        raise ValueError('Missing command')
    out.mkdir(parents=True, exist_ok=True)
    cache_root = Path(args.cache_root).expanduser().resolve() if args.cache_root else out
    if not (cache_root == out or cache_root in out.parents):
        raise ValueError('Cache root must contain the private output directory')
    if any(cache_root == p or p in cache_root.parents for p in forbidden):
        raise ValueError('Cache root must be outside the public course')
    def cache_bytes():
        total = 0
        for directory, dirs, files in os.walk(cache_root, followlinks=False):
            dirs[:] = [d for d in dirs if not Path(directory, d).is_symlink()]
            for name in files:
                path = Path(directory, name)
                if not path.is_symlink():
                    try:
                        total += path.stat().st_size
                    except FileNotFoundError:
                        pass
        return total
    initial_cache = cache_bytes()
    if initial_cache > 10 * 1024**3:
        raise ValueError('Existing private execution directory exceeds 10GiB')
    record_path = out / (args.name + '_resource.json')
    if record_path.exists():
        raise FileExistsError('This execution name already has a resource record')
    import psutil
    env = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        env[key] = str(args.threads)
    env.update(PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='', JAX_PLATFORM_NAME='cpu',
               JAX_ENABLE_X64='true', MPLBACKEND='Agg')
    limit = args.memory_gib * 1024**3
    def restrict():
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        available = sorted(os.sched_getaffinity(0))
        os.sched_setaffinity(0, available[:args.threads])
    start = time.monotonic()
    record = {'name': args.name, 'started_unix': time.time(), 'status': 'running',
              'command': command, 'location': 'Wuying CPU', 'threads_limit': args.threads,
              'address_space_limit_gib_per_process': args.memory_gib,
              'process_tree_rss_limit_gib': args.memory_gib, 'observation_period_seconds': 0.2,
              'cache_root': str(cache_root), 'cache_limit_gib': 10, 'initial_cache_bytes': initial_cache}
    record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    peak = 0
    cause = None
    peak_cache = initial_cache
    last_cache_check = start
    with (out / (args.name + '.log')).open('w', encoding='utf-8') as log:
        proc = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                preexec_fn=restrict, start_new_session=True)
        root = psutil.Process(proc.pid)
        while proc.poll() is None:
            try:
                rss = root.memory_info().rss
                for child in root.children(recursive=True):
                    try:
                        rss += child.memory_info().rss
                    except psutil.Error:
                        pass
                peak = max(peak, rss)
                if rss > limit:
                    cause = 'process_tree_memory_limit'
            except psutil.Error:
                pass
            if time.monotonic() - start > args.seconds:
                cause = 'wall_time_limit'
            if time.monotonic() - last_cache_check >= 5:
                peak_cache = max(peak_cache, cache_bytes())
                last_cache_check = time.monotonic()
                if peak_cache > 10 * 1024**3:
                    cause = 'private_cache_limit'
            if cause:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                break
            time.sleep(.2)
    final_cache = cache_bytes()
    if final_cache > 10 * 1024**3:
        cause = 'private_cache_limit'
    record.update(status='success' if proc.returncode == 0 and not cause else 'failed',
                  stop_reason=cause, exit_code=proc.returncode, wall_seconds=time.monotonic()-start,
                  peak_sampled_process_tree_rss_bytes=peak,
                  peak_observed_cache_bytes=max(peak_cache, final_cache))
    record_path.write_text(json.dumps(record, indent=2), encoding='utf-8')
    print(json.dumps(record), flush=True)
    return 0 if record['status'] == 'success' else 1


if __name__ == '__main__':
    sys.exit(main())
