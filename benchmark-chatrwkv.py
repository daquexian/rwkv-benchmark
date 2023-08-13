########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, sys, types, json, math, time
import argparse
import functools
from collections import defaultdict
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{current_path}/rwkv_pip_package/src')
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='/fsx/BlinkDL/HF-MODEL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221110-ctx4096')
parser.add_argument('--strategy', type=str, default='cuda fp16')
args = parser.parse_args()
print(f'strategy is {args.strategy}')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
np.set_printoptions(precision=4, suppress=True, linewidth=200)

########################################################################################################

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_CUDA_ON"] = "1"

MODEL_NAME = args.model

########################################################################################################

print(f'\nLoading ChatRWKV https://github.com/BlinkDL/ChatRWKV')
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

from torch.nn import functional as F
import rwkv
from rwkv.model import RWKV

print(f'Loading model - {MODEL_NAME}')
model = RWKV(model=MODEL_NAME, strategy=args.strategy)

print('Benchmark speed...')

# Warmup
for i in range(10):
    out, state = model.forward([0], None if i == 0 else state)

start = time.time()

num_tokens = 100
for i in range(num_tokens):
    out, state = model.forward([0], None if i == 0 else state)

end = time.time()

print(f'Speed: {num_tokens / (end - start):.2f} tokens/sec')

print(f"GPU memory usage: {torch.cuda.max_memory_reserved() / 1024 / 1024} MB")
