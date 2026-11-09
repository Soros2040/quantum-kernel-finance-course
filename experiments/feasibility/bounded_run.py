"""Linux bounded experiment entrypoint: private output, affinity and memory limit."""
import argparse
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from safe_io import OutputRoot


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--name',required=True)
    p.add_argument('--seconds',type=int,default=7200);p.add_argument('--threads',type=int,default=8);p.add_argument('--memory-gib',type=int,default=32)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();out=OutputRoot(a.output)
    if a.command and a.command[0]=='--':a.command=a.command[1:]
    if not a.command:raise ValueError('Missing command')
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):env[key]=str(a.threads)
    env['PYTHONDONTWRITEBYTECODE']='1';env['EXECUTION_LOCATION']='Wuying CPU'
    def restrict():
        resource.setrlimit(resource.RLIMIT_AS,(a.memory_gib*1024**3,)*2)
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,available[:a.threads])
    started=time.perf_counter();peak=0;timed_out=False
    import psutil
    with out.path(a.name+'.log').open('w',encoding='utf-8') as log:
        process=subprocess.Popen(a.command,env=env,stdout=log,stderr=subprocess.STDOUT,preexec_fn=restrict,start_new_session=True)
        monitored=psutil.Process(process.pid)
        while process.poll() is None:
            try:peak=max(peak,monitored.memory_info().rss+sum(c.memory_info().rss for c in monitored.children(recursive=True) if c.is_running()))
            except psutil.Error:pass
            if time.perf_counter()-started>a.seconds:
                import signal
                os.killpg(process.pid,signal.SIGTERM);timed_out=True
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
                break
            time.sleep(.2)
    record={'name':a.name,'execution_location':'Wuying','exit_code':process.returncode,'timed_out':timed_out,'wall_seconds':time.perf_counter()-started,
            'peak_sampled_process_tree_rss_bytes':peak,'memory_metric':'sampled every 0.2 s, not exact peak','threads_limit':a.threads,'address_space_limit_gib':a.memory_gib,
            'command':a.command}
    out.json(a.name+'_resource.json',record);print(record,flush=True)


if __name__=='__main__':main()
