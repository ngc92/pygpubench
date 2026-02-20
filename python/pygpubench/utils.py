import os

class DeterministicContext:
    def __init__(self):
        self.allow_tf32 = None
        self.deterministic = None
        self.cublas = None

    def __enter__(self):
        import torch
        self.cublas = os.environ.get('CUBLAS_WORKSPACE_CONFIG', '')
        self.allow_tf32 = torch.backends.cudnn.allow_tf32
        self.deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import torch
        torch.backends.cudnn.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.deterministic = self.deterministic
        torch.use_deterministic_algorithms(False)
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = self.cublas
