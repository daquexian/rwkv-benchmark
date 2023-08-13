## Benchmark RWKV backends

It rents GPUs from vast.ai, so make sure you have the vast.ai command line tool installed.

Example:

```
python3 benchmark.py --model xxx.pth --verbose -n 1 --branch daquexian/test --log-dir log
```

### How to add a new backend

Please refer to `ChatRWKV` backend implementation in benchmark.py. Typically, a benchmark script (like benchmark_chatrwkv.py) is also needed.
