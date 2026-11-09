"""Explicit Linux execution entry; credentials stay in process memory.

No network call occurs unless --execute is supplied and independently reviewed
account evidence passes. Never use this entry to discover billing by execution.
"""
from __future__ import annotations
import argparse
import getpass
import hashlib
import json
import logging
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--evidence', required=True, help='Private official fee/capability evidence JSON')
    parser.add_argument('--ledger-root', required=True, help='Canonical private account ledger root')
    parser.add_argument('--account-scope', required=True, help='Stable non-secret account identifier')
    parser.add_argument('--window', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage', default='pilot8')
    parser.add_argument('--task', choices=['classification','regression'], default='classification')
    parser.add_argument('--method', choices=['qukernet','random'], default='qukernet')
    parser.add_argument('--seed', type=int, default=20261110)
    args = parser.parse_args(argv)
    if not args.execute:
        print('Cloud execution disabled. Use experiments --dry-run for resource accounting.')
        return {'status':'disabled','cloud_submissions':0}
    if sys.platform != 'linux':
        raise RuntimeError('Cloud transport requires the Linux bounded-request runtime; use the Wuying client.')
    from .preflight import validate_evidence
    evidence = json.loads(Path(args.evidence).read_text(encoding='utf-8'))
    account_hash = hashlib.sha256(args.account_scope.encode()).hexdigest()
    ledger = Path(args.ledger_root).expanduser().resolve()/('account-'+account_hash)/'ledger.json'
    validate_evidence(evidence, account_hash, ledger_path_hash=hashlib.sha256(str(ledger).encode()).hexdigest())
    # Bound classical libraries before importing them. No shared environment is modified.
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]='8'
    import resource
    memory=32*1024**3
    resource.setrlimit(resource.RLIMIT_AS,(memory,memory))
    import importlib.metadata
    if importlib.metadata.version('cqlib') != '1.3.11':
        raise RuntimeError('Use the reviewed Cqlib1.3.11 environment; no implicit SDK upgrade.')
    from cqlib import TianYanPlatform
    if TianYanPlatform.SCHEME != 'https' or TianYanPlatform.DOMAIN != 'qc.zdxlz.com':
        raise RuntimeError('Official HTTPS domain required; certificate verification is not disabled.')
    logging.getLogger('cqlib').setLevel(logging.CRITICAL)
    secret = os.environ.get('TIANYAN_API_KEY') or getpass.getpass('TianYan API key (hidden input): ')
    if not secret:
        raise RuntimeError('Missing process credential')
    try:
        platform = TianYanPlatform(login_key=secret, machine_name='tianyan_sw')
    except Exception:
        raise RuntimeError('TianYan authentication failed; credentials and raw response suppressed') from None
    finally:
        secret=None
    from .cloud import create_provider
    provider=create_provider(platform=platform, account_ledger_root=args.ledger_root,
                             account_scope=args.account_scope, evidence=evidence)
    from .experiments import main as run
    return run(['--execute','--stage',args.stage,'--window',args.window,'--output',args.output,
                '--task',args.task,'--method',args.method,'--seed',str(args.seed)],provider=provider)


if __name__=='__main__':
    main()
