import torchsde
from flow_matching.solver.solver import Solver
import torch
from torch import Tensor
from typing import Union, Callable, Sequence, Tuple, Optional
from flow_matching.utils import ModelWrapper

class SDESolver(Solver):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, 
                 velocity_model: Union[ModelWrapper, Callable], 
                 score: Union[ModelWrapper, Callable], 
                 reverse: bool = False, 
                 sigma: float = 0.1):
        super().__init__()
        self.velocity_model = velocity_model
        self.score = score
        self.reverse = reverse
        self.sigma = sigma
        self._input_shape = None

    # Drift
    def f(self, t, y):
        y = y.view(-1, *self._input_shape)
        if self.reverse:
            t = 1 - t
            return -self.velocity_model(y, t) + self.score(y, t)
        else:
            return self.velocity_model(y, t) + self.score(y, t)  # shape (batch_size, state_size)

    # Diffusion
    def g(self, t, y):
        y = y.view(-1, *self._input_shape)
        return torch.ones_like(y) * self.sigma

    def sample(
            self, 
            x_init: Tensor, 
            dt: float = 0.01,
            method: str = "ito",
            atol: float = 1e-5,
            rtol: float = 1e-5,
            time_grid: Tensor = torch.tensor([0.0, 1.0]),
            return_intermediates: bool = False,
            ) -> Union[Tensor, Sequence[Tensor]]:
        self.sde_type = method
        self._input_shape = tuple(x_init.shape[1:])
        batch_size = x_init.shape[0]
        with torch.no_grad():
            sol = torchsde.sdeint(self, 
                                   x_init.view(batch_size, -1), 
                                   time_grid.to(x_init.device), 
                                   dt=dt, 
                                   atol=atol, 
                                   rtol=rtol)
        sol = sol.view(time_grid.shape[0], batch_size, *self._input_shape)
        if return_intermediates:
            return sol
        else:
            return sol[-1]


