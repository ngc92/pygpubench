[![PyPI version](https://img.shields.io/pypi/v/pygpubench)](https://pypi.org/project/pygpubench/)

# PyGPUBench

PyGPUBench is a CUDA microbenchmark harness for **untrusted kernels**.

Most benchmarking scripts assume the kernel under test is cooperative: it computes the right result, runs on the expected stream, and does not introspect the harness. That assumption breaks if submissions are generated to maximize benchmark score instead of correctness.

This project focuses on that adversarial setting.

## What this evaluates (and what it does not)

PyGPUBench answers a narrow question:

> "Given this kernel API and these test generators, can this kernel produce correct outputs while measured honestly under short-latency CUDA timing?"

It does **not** sandbox arbitrary code execution. User code still runs in-process and can execute Python, PyTorch, and CUDA runtime calls.

So the goal is not "perfect isolation". The goal is to make common reward-hacking strategies fail, and to make successful cheating require substantially more sophistication.

## Real attack vectors in this setting

When kernel code is untrusted, these are practical attack classes:

- **Answer leakage / answer copying**
  - Read expected outputs from Python objects and copy them into the output tensor.
- **Validator tampering**
  - Corrupt or reset state used by correctness checking (for example, a GPU-side error counter).
- **Result-channel tampering**
  - Rewrite benchmark output after measurement but before the parent process reads it.
- **Timing-shape manipulation**
  - Detect warmup/measurement phases and bias iteration count or timing windows.
- **Stream/order races**
  - Launch work on different streams so measured events do not reflect true compute.

We document these hacks under [exploits/](exploits/)

## What PyGPUBench does differently

Compared with a typical Python+CUDA benchmarking loop, PyGPUBench adds defenses in C++ around the measurement path:

1. **Benchmark core in compiled extension**
   - Timing, cache management, and validation orchestration run in `csrc/manager.cpp`, not pure Python.

2. **Subprocess boundary (`do_bench_isolated`)**
   - The public API runs untrusted code in a spawned child process and only parses a result file in the parent.

3. **Expected-output shadowing**
   - Expected tensors are copied to raw `cudaMalloc` memory and original tensor storage is zeroed before user code is imported.
   - This blocks straightforward `gc.get_objects()` answer-sheet attacks.

4. **Input shadowing + canary repair window**
   - Inputs are staged in hidden device buffers, restored before each run, L2 is cleared, then sparse canaries are repaired immediately before launch.
   - This reduces opportunities to exploit stale-cache behavior.

5. **Immediate post-kernel validation**
   - Correctness check kernels launch right after the user kernel on the benchmark stream.
   - Test order is randomized to make selective late-write strategies less reliable.

6. **Basic anti-introspection hardening**
   - `PR_SET_DUMPABLE=0` and `PR_SET_NO_NEW_PRIVS=1` are set before user kernel generation.

## Current limits (important)

This suite is hardened, not airtight. Known limitations include:

- User code still shares process and GPU address space with the harness.
- The GPU error counter is currently a plain device allocation; this is a meaningful tampering target.
- Output-file integrity is not cryptographically signed.
- Warmup uses repeated calls on the same warmup input, which can be pattern-detected.


## Quick start

```python
import torch
import pygpubench


def generate_test_case(size: int, seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    x = torch.rand(size, size, 3, device="cuda", dtype=torch.float32, generator=gen).contiguous()
    y = torch.empty(size, size, device="cuda", dtype=torch.float32).contiguous()

    expected = torch.empty_like(y)
    w = torch.tensor([0.2989, 0.5870, 0.1140], device=x.device, dtype=x.dtype)
    expected[...] = torch.sum(x * w, dim=-1)

    # kernel inputs, then (expected, rtol, atol)
    return (y, x), (expected, 1e-6, 1e-6)


def kernel_generator():
    import submission  # import untrusted module here
    return submission.kernel


res = pygpubench.do_bench_isolated(
    kernel_generator,
    generate_test_case,
    {"size": 1024},
    repeats=100,
    seed=5,
    discard=True,
)

print("❌" if res.errors else "✅", pygpubench.basic_stats(res.time_us))
```

See `test/grayscale.py` for a complete runnable example.

## API surface

- `do_bench_isolated(...)`
  - Recommended path. Spawns a subprocess and returns parsed benchmark results.
- `do_bench_impl(...)`
  - Lower-level call that runs in the current process.
- `basic_stats(time_us)`
  - Convenience stats for completed runs.

## Running exploit regression tests

Use the exploit suite to validate assumptions on your machine:

```bash
cd exploits
python run_all.py
```

This reports which cheating strategies are currently blocked vs still viable.
