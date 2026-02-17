# PyGPUBench

Utilities for benchmarking low-latency CUDA kernels in an _adversarial_ setting.
Contrary to many existing benchmarking tools, which generally assume a cooperative kernel that
can be tested and benchmarked independently, this library tries to defend against kernels that
try to exploit benchmarking flaws to receive higher scores.

Unfortunately, any benchmarking tool written in python is inherently vulnerable to monkeypatching
and `inpect`-based manipulation of its variables by its callees. 
Therefore, PyGPUBench implements its main benchmarking logic in a compiled C++ extension. 
While this still leaves vulnerabilities - the code is running in the same address space, after all –
it makes attacks require much more sophistication. Running in a separate process fundamentally
clashes with the desire to benchmark very short kernels; cuda events must be recorded in the same
process as the kernel. Fortunately, we can assume that a reward-hacking LLM is still rather 
unlikely to produce a compiled extension that runs sophisticated low-level exploits.

Note that, as soon as any user code is executed, the entire python runtime becomes untrustworthy.
Consequently, benchmark results are not returned to python, but instead written to a file. The
name of this file is passed as an argument to the benchmarking function, and the file is unlinked
before the user code is called, making it impossible to reopen this file.
The `do_bench_isolated` function is designed to streamline this process: It automates creating
the temporary file, spawning a new python process to handle the benchmarking, and reading the
results back into python (the original, untainted process).
