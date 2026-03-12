from typing import Callable, Tuple

Tensor = "torch.Tensor"
ExpectedSpec = Tensor | Tuple[Tensor] | Tuple[Tensor, float, float]
ExpectedResult = Tuple[ExpectedSpec, ...]

KernelFunction = Callable[..., None]
TestGeneratorInterface = Callable[..., Tuple[Tuple, Tuple, ExpectedResult]]

__all__ = ["KernelFunction", "TestGeneratorInterface", "ExpectedSpec", "ExpectedResult"]
