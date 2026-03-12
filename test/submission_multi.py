import torch


_weights = torch.tensor([0.2989, 0.5870, 0.1140],
                        device="cuda:0",
                        dtype=torch.float32)


stream = torch.cuda.Stream(device="cuda:0")
event = torch.cuda.Event(enable_timing=False)


def valid_custom_kernel_eager(output_gray, output_red, data):
    torch.sum(data * _weights, dim=-1, out=output_gray)
    output_red.copy_(data[..., 0])


@torch.compile
def valid_custom_kernel_compiled(output_gray, output_red, data):
    torch.sum(data * _weights, dim=-1, out=output_gray)
    output_red.copy_(data[..., 0])


def wrong_custom_kernel_backward_race(output_gray, output_red, data):
    with torch.cuda.stream(stream):
        torch.sum(data * _weights, dim=-1, out=output_gray)
        output_red.copy_(data[..., 0])
        event.record()
    event.synchronize()


def wrong_custom_kernel_forward_race(output_gray, output_red, data):
    event.record()
    with torch.cuda.stream(stream):
        event.synchronize()
        torch.sum(data * _weights, dim=-1, out=output_gray)
        output_red.copy_(data[..., 0])


def valid_custom_kernel_stream(output_gray, output_red, data):
    event.record()
    with torch.cuda.stream(stream):
        event.synchronize()
        torch.sum(data * _weights, dim=-1, out=output_gray)
        output_red.copy_(data[..., 0])
        event.record()
    event.synchronize()


def wrong_custom_kernel_sneaky(output_gray, output_red, data):
    event.record()
    with torch.cuda.stream(stream):
        event.synchronize()
        torch.sum(data * _weights, dim=-1, out=output_gray)
        output_red.copy_(data[..., 0])
        event.record()
    event.synchronize()