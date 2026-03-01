from typing import Callable, Tuple

Tensor = "torch.Tensor"
KernelArgs = Tuple
ExpectedResult = Tuple[Tensor] | Tuple[Tensor, float, float]

KernelFunction = Callable[[KernelArgs], None]
KernelGeneratorInterface = Callable[[], KernelFunction]
TestGeneratorInterface = Callable[..., Tuple[KernelArgs, ExpectedResult]]

__all__ = ["KernelFunction", "KernelGeneratorInterface", "TestGeneratorInterface", "ExpectedResult"]
