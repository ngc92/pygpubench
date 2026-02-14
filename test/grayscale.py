import torch
import pygpubench

def reference_kernel(data):
    output, data = data
    weights = torch.tensor([0.2989, 0.5870, 0.1140],
                           device=data.device,
                           dtype=data.dtype)
    output[...] = torch.sum(data * weights, dim=-1)


_weights = torch.tensor([0.2989, 0.5870, 0.1140],
                       device="cuda:0",
                       dtype=torch.float32)

def custom_kernel(data):
    output, data = data
    torch.sum(data * _weights, dim=-1, out=output)


def generate_input(size: int, seed: int):
    """
    Generates random RGB image tensor of the specified size.
    Returns:
        Tensor of shape (size, size, 3) with values in [0, 1]
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    x = torch.rand(
        size, size, 3, device="cuda", dtype=torch.float32, generator=gen
    ).contiguous()

    y = torch.empty(size, size, device="cuda", dtype=torch.float32).contiguous()

    return x, y


def generate_test_case(args):
    x, y = generate_input(*args)
    expected = torch.empty_like(y)
    reference_kernel((expected, x))
    return (y, x), (expected, 0.0, 0.0)


def kernel_generator():
    return custom_kernel

#void do_bench(std::string target_file, const nb::callable& kernel_generator, const nb::callable& test_generator, const nb::tuple& test_args, int repeats, std::uintptr_t stream) {
pygpubench._pygpubench.do_bench("bm.txt", kernel_generator, generate_test_case,  (128, 5), 100, torch.cuda.current_stream().cuda_stream)
