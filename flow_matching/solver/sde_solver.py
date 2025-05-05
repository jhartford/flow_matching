import torchsde
from flow_matching.solver.solver import Solver
import torch
from torch import Tensor
from typing import Union, Callable, Sequence, Tuple, Optional
from flow_matching.utils import ModelWrapper
from math import prod

class SDESolver(Solver):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, 
                 velocity_model: Union[ModelWrapper, Callable] = None, 
                 score_model: Union[ModelWrapper, Callable] = None,
                 combined_model: Union[ModelWrapper, Callable] = None,
                 reverse: bool = False, 
                 sigma: float = 0.1):
        super().__init__()
        assert (velocity_model is not None and score_model is not None) or combined_model is not None, \
            "Either velocity_model and score_model or combined_model must be provided"
        self.velocity_model = velocity_model
        self.score_model = score_model
        self.combined_model = combined_model
        self.reverse = reverse
        self.sigma = sigma
        self._input_shape = None
        self._model_inputs = None

    # Drift
    def f(self, t, x):
        inputs = self._model_inputs
        x = x.view(*self._input_shape)
        if self.combined_model is not None:
            # Handle extra inputs without crashing if the model doesn't take them
            if len(inputs) > 0:
                velocity, score = torch.chunk(self.combined_model(t, x, **inputs), 2, dim=1)
            else:
                velocity, score = torch.chunk(self.combined_model(t, x), 2, dim=1)
        else:
            if len(inputs) > 0:
                velocity = self.velocity_model(t,x, **inputs)
                score = self.score_model(t, x, **inputs)
            else:
                velocity = self.velocity_model(t, x)
                score = self.score_model(t, x)
        if self.reverse:
            t = 1 - t
            output = (-velocity + score).view(x.shape[0], -1)
            return output
        else:
            output = (velocity + score).view(x.shape[0], -1)
            return output

    # Diffusion
    def g(self, t, x):
        return torch.ones_like(x) * self.sigma

    def sample(
            self,
            x_init: Tensor, 
            dt: float = 0.01,
            method: str = "ito",
            atol: float = 1e-5,
            rtol: float = 1e-5,
            time_grid: Tensor = torch.tensor([0.0, 1.0]),
            return_intermediates: bool = False,
            **model_inputs
            ) -> Union[Tensor, Sequence[Tensor]]:
        self.sde_type = method
        self._input_shape = x_init.shape

        with torch.no_grad():
            self._model_inputs = model_inputs
            sol = torchsde.sdeint(self, x_init.flatten(start_dim=1), time_grid.to(x_init.device),dt=dt, atol=atol, rtol=rtol)
        sol = sol.view(time_grid.shape[0], *self._input_shape)
        if return_intermediates:
            return sol
        else:
            return sol[-1]
