from typing import Callable, Tuple

Tensor = "torch.Tensor"
ExpectedResult = Tuple[Tensor] | Tuple[Tensor, float, float]

KernelFunction = Callable[..., None]
KernelGeneratorInterface = Callable[[], KernelFunction]
TestGeneratorInterface = Callable[..., Tuple[Tuple, ExpectedResult]]

__all__ = ["KernelFunction", "KernelGeneratorInterface", "TestGeneratorInterface", "ExpectedResult"]
