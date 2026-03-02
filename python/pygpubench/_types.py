from typing import Callable, Tuple

Tensor = "torch.Tensor"
ExpectedSpec = Tensor | Tuple[Tensor] | Tuple[Tensor, float, float]
ExpectedResult = Tuple[ExpectedSpec, ...]

KernelFunction = Callable[..., None]
KernelGeneratorInterface = Callable[[], KernelFunction]
TestGeneratorInterface = Callable[..., Tuple[Tuple, Tuple, ExpectedResult]]

__all__ = ["KernelFunction", "KernelGeneratorInterface", "TestGeneratorInterface", "ExpectedSpec", "ExpectedResult"]
