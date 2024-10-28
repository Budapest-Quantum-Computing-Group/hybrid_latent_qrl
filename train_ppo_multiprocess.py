from __future__ import print_function
from select     import select
from subprocess import Popen, PIPE

import os
import sys    
    
import numpy as np
from datetime import datetime

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--config-file', type=str, required=True)
parser.add_argument('--script-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--n-agents', type=int, required=True)
parser.add_argument('--continued', action='store_true')

args = parser.parse_args()

if args.continued == True:
    commands = [
        ['python', f"{args.script_path}", f"--config-file={args.config_file}", f"--save-path={args.save_path}", f"--agent-id={i}", f"--continued"]
        for i in range(args.n_agents)
    ]
else :
    commands = [
        ['python', f"{args.script_path}", f"--config-file={args.config_file}", f"--save-path={args.save_path}", f"--agent-id={i}"]
        for i in range(args.n_agents)
    ]

[print("Running command " + " ".join(command)) for command in commands]
sys.stdout.flush()

processes = [
    Popen(
        command, 
        stdout=PIPE,
        bufsize=1, 
        close_fds=True,
        universal_newlines=True
    )
    for command in commands
]

# read output
timeout = 0.5 # seconds
while processes:
    # remove finished processes from the list (O(N**2))
    for p in processes[:]:
        if p.poll() is not None: # process ended
            print(p.stdout.read(), end='') # read the rest
            p.stdout.close()
            processes.remove(p)

    # wait until there is something to read
    rlist = select([p.stdout for p in processes], [],[], timeout)[0]

    # read a line from each process that has output ready
    for f in rlist:
        print(f.readline(), end='') #NOTE: it can block
        sys.stdout.flush()