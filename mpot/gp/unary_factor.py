import torch


class UnaryFactor:

    def __init__(
            self,
            dim: int,
            sigma: float,
            mean: torch.Tensor = None,
            tensor_args=None,
    ):
        self.sigma = sigma
        if mean is None:
            self.mean = torch.zeros(dim, **tensor_args)
        else:
            self.mean = mean
        self.tensor_args = tensor_args
        self.K = torch.eye(dim, **tensor_args) / sigma**2  # weight matrix
        self.dim = dim

    def get_error(self, X: torch.Tensor, calc_jacobian: bool = True) -> torch.Tensor:
        error = self.mean - X

        if calc_jacobian:
            H = torch.eye(self.dim, **self.tensor_args).unsqueeze(0).repeat(X.shape[0], 1, 1)
            return error.view(X.shape[0], self.dim, 1), H
        else:
            return error

    def set_mean(self, X: torch.Tensor) -> torch.Tensor:
        self.mean = X.clone().detach()
