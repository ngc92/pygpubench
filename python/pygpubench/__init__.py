import dataclasses
import math
import multiprocessing as mp
import tempfile

from pathlib import Path
from typing import Optional

from . import _pygpubench
from ._types import *
from .utils import DeterministicContext

__all__ = [
    "do_bench_impl",
    "do_bench_isolated",
    "basic_stats",
    "BenchmarkResult",
    "BenchmarkSummary",
    "DeterministicContext",
    "KernelFunction",
    "KernelGeneratorInterface",
    "TestGeneratorInterface",
    "ExpectedResult",
]


def do_bench_impl(out_file: str, kernel_generator: KernelGeneratorInterface, test_generator: TestGeneratorInterface,
                  test_args: dict, repeats: int, seed: int, stream: int = None, discard: bool = True,
                  unlink: bool = False, nvtx: bool = False):
    """
    Benchmarks the kernel returned by `kernel_generator` against the test case returned by `test_generator`.
    :param out_file: File in which to write the benchmark results.
    :param kernel_generator: A function that takes no arguments and returns a kernel function.
    :param test_generator: A function that takes the test arguments (including a seed) and returns a test case; i.e., a tuple of (input, expected)
    :param test_args: keyword arguments to be passed to `test_generator`. Seed will be generated automatically.
    :param repeats: Number of times to repeat the benchmark. `test_generator` will be called `repeat` times.
    :param stream: Cuda stream on which to run the benchmark. If not given, torch's current stream is selected
    :param discard: If true, then cache lines are discarded as part of cache clearing before each benchmark run.
    :param unlink: Whether to unlink the output file before calling `kernel_generator`. Unlinking makes it impossible to
    open the file again, protecting it against malicious kernels.
    :param nvtx: Whether to enable NVTX markers for the benchmark. Mostly useful for debugging.
    """
    assert repeats > 1
    if stream is None:
        import torch
        stream = torch.cuda.current_stream().cuda_stream

    with DeterministicContext():
        _pygpubench.do_bench(out_file, kernel_generator, test_generator, test_args, repeats, seed, stream, discard, unlink, nvtx)


@dataclasses.dataclass
class BenchmarkResult:
    event_overhead_us: float
    time_us: list[float]
    errors: Optional[int]


@dataclasses.dataclass
class BenchmarkSummary:
    fastest: float
    slowest: float
    median: float
    mean: float
    std: float

    def __str__(self):
        return f"{self.mean:.1f} ± {self.std:.2f} µs [{self.fastest:.1f} - {self.median:.1f} - {self.slowest:.1f}]"


def basic_stats(time_us: list[float]) -> BenchmarkSummary:
    fastest = min(time_us)
    slowest = max(time_us)
    median = sorted(time_us)[len(time_us) // 2]
    mean = sum(time_us) / len(time_us)
    std = math.sqrt(sum(map(lambda x: (x - mean) ** 2, time_us)) / len(time_us))
    return BenchmarkSummary(fastest, slowest, median, mean, std)

def do_bench_isolated(
        kernel_generator: KernelGeneratorInterface,
        test_generator: TestGeneratorInterface,
        test_args: dict,
        repeats: int,
        seed: int,
        *,
        discard: bool = True,
        nvtx: bool = False
) -> BenchmarkResult:
    """
    Runs kernel benchmark (`do_bench_impl`) in a subprocess for proper isolation.
    """
    assert repeats > 1

    # Create a named temporary file for the C++ extension to store the results in
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tsv') as f:
        result_file = f.name

    try:
        # open file before running subprocess; process will unlink
        with open(result_file, 'r') as f:
            # Spawn testing process
            ctx = mp.get_context('spawn')
            process = ctx.Process(
                target=do_bench_impl,
                args=(result_file, kernel_generator, test_generator,
                      test_args, repeats, seed, None, discard, True, nvtx)  # unlink=True
            )

            process.start()
            process.join()

            if process.exitcode != 0:
                raise RuntimeError(f"Benchmark subprocess failed with exit code {process.exitcode}")

            # Read results from file
            results = BenchmarkResult(None, [None] * repeats, None)
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2 and parts[0].isdigit():
                    iteration = int(parts[0])
                    time_us = float(parts[1])
                    results.time_us[iteration] = time_us
                elif parts[0] == "event-overhead":
                    results.event_overhead_us = float(parts[1].split()[0])
                elif parts[0] == "error-count":
                    results.errors = int(parts[1])

        return results

    finally:
        Path(result_file).unlink(missing_ok=True)
