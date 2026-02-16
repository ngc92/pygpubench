from typing import Callable, Tuple

from . import _pygpubench


KernelGeneratorInterface = Callable[[], Callable[[Tuple], None]]
TestGeneratorInterface = Callable[[Tuple], Tuple[Tuple, Tuple]]

def do_bench(out_file: str, kernel_generator: KernelGeneratorInterface, test_generator: TestGeneratorInterface,
             test_args: tuple, repeats: int, stream: int = None, unlink: bool = False, nvtx: bool = False):
    """
    Benchmarks the kernel returned by `kernel_generator` against the test case returned by `test_generator`.
    :param out_file: File in which to write the benchmark results.
    :param kernel_generator: A function that takes no arguments and returns a kernel function.
    :param test_generator: A function that takes a tuple of test arguments and returns a test case; i.e., a tuple of (input, expected)
    :param test_args: arguments to be passed to `test_generator`
    :param repeats: Number of times to repeat the benchmark. `test_generator` will be called `repeat` times.
    :param stream: Cuda stream on which to run the benchmark. If not give, torch's current stream is selected
    :param unlink: Whether to unlink the output file before calling `kernel_generator`. Unlinking makes it impossible to
    open the file again, protecting it against malicious kernels.
    :param nvtx: Whether to enable NVTX markers for the benchmark. Mostly useful for debugging.
    :return:
    """
    assert repeats > 1
    if stream is None:
        import torch
        stream = torch.cuda.current_stream().cuda_stream

    _pygpubench.do_bench(out_file, kernel_generator, test_generator, test_args, repeats, stream, unlink, nvtx)

