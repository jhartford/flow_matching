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

    # Drift
    def f(self, t, x):
        inputs = self._unflatten_input(x)
        x = inputs['x']
        del inputs['x']
        if self.combined_model is not None:
            if len(inputs) > 0:
                velocity, score = torch.chunk(self.combined_model(x, t, **inputs), 2, dim=1)
            else:
                velocity, score = torch.chunk(self.combined_model(x, t), 2, dim=1)
        else:
            if len(inputs) > 0:
                velocity = self.velocity_model(x, t, **inputs)
                score = self.score_model(x, t, **inputs)
            else:
                velocity = self.velocity_model(x, t)
                score = self.score_model(x, t)
        if self.reverse:
            t = 1 - t
            return -velocity + score

        else:
            return velocity + score


    # Diffusion
    def g(self, t, x):
        inputs = self._unflatten_input(x)
        x = inputs['x']
        del inputs['x']
        return torch.ones_like(x) * self.sigma

    def _flatten_input(self, x: Tensor, **model_inputs):
        model_inputs['x'] = x
        model_inputs = {k: v.to(x.device) for k, v in model_inputs.items() if isinstance(v, Tensor)}
        self._input_shape = {k: tuple(v.shape[1:]) for k, v in model_inputs.items()}
        batch_size = x.shape[0]
        return torch.cat([v.view(batch_size, -1) for v in model_inputs.values()], dim=-1)
    
    def _unflatten_input(self, x: Tensor):
        tensors = torch.split(x, [sum(shape) for shape in self._input_shape.values()], dim=-1)
        return {k: v.view(-1, *self._input_shape[k]) for k, v in zip(self._input_shape.keys(), tensors)}

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
        self._input_shape = tuple(x_init.shape[1:])
        batch_size = x_init.shape[0]

        with torch.no_grad():
            x_init = self._flatten_input(x_init, **model_inputs)
            sol = torchsde.sdeint(self, 
                                   x_init, 
                                   time_grid.to(x_init.device), 
                                   dt=dt, 
                                   atol=atol, 
                                   rtol=rtol)
        sol = sol.view(time_grid.shape[0], batch_size, *self._input_shape)
        if return_intermediates:
            return sol
        else:
            return sol[-1]


