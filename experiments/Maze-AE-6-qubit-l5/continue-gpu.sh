#!/bin/bash
mkdir -p ./results
PYTHONPATH=../.. python ../../train_ppo_multiprocess_gpu.py --save-path=./results --config-file=./config.json --n-agents=8 --script-path=../../train_ppo_agent_tf_function.py --continued
