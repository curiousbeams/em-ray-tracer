"""Optimizer and scheduler management for a model that owns learnable parameters.

The owner implements ``get_optimization_parameters()`` returning ``{group_key: [tensors]}``,
whose keys match those of ``optimizer_params``. A single group uses ``DEFAULT_OPTIMIZER_KEY``;
several groups give per-group learning rates. ``step_optimizer`` takes a closure, so LBFGS
works.
"""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any

import torch

from em_ray_tracer.optimize.params import (
    OptimizerParams,
    OptimizerParamsType,
    SchedulerParams,
    SchedulerParamsType,
)


class OptimizerMixin:
    DEFAULT_OPTIMIZER_KEY = "default"

    def __init__(self):
        self._optimizer: torch.optim.Optimizer | None = None
        self._scheduler: Any = None
        self._optimizer_params: dict[str, OptimizerParamsType] = {
            self.DEFAULT_OPTIMIZER_KEY: OptimizerParams.NoneOptimizer()
        }
        self._scheduler_params: SchedulerParamsType = SchedulerParams.NoneScheduler()

    @property
    def optimizer(self) -> torch.optim.Optimizer | None:
        return self._optimizer

    @property
    def scheduler(self):
        return self._scheduler

    @property
    def optimizer_params(self) -> dict[str, OptimizerParamsType]:
        return self._optimizer_params

    @optimizer_params.setter
    def optimizer_params(self, params: OptimizerParamsType | dict[str, Any]) -> None:
        if isinstance(params, OptimizerParamsType):
            self._optimizer_params = {self.DEFAULT_OPTIMIZER_KEY: params}
        elif isinstance(params, dict) and ("name" in params or "type" in params):
            self._optimizer_params = {
                self.DEFAULT_OPTIMIZER_KEY: OptimizerParams.parse_dict(params)
            }
        elif isinstance(params, dict):
            self._optimizer_params = {
                k: v if isinstance(v, OptimizerParamsType) else OptimizerParams.parse_dict(v)
                for k, v in params.items()
            }
        else:
            raise TypeError(f"optimizer_params must be OptimizerParamsType or dict, got {params}")

    @property
    def scheduler_params(self) -> SchedulerParamsType:
        return self._scheduler_params

    @scheduler_params.setter
    def scheduler_params(self, params: SchedulerParamsType | dict) -> None:
        if isinstance(params, dict):
            params = SchedulerParams.parse_dict(params)
        if not isinstance(params, SchedulerParamsType):
            raise TypeError(f"scheduler_params must be SchedulerParamsType, got {params}")
        self._scheduler_params = params

    @abstractmethod
    def get_optimization_parameters(self) -> dict[str, list[torch.Tensor]]:
        """Tensors to optimize, grouped by the keys of ``optimizer_params``."""

    def set_optimizer(self, opt_params: OptimizerParamsType | dict | None = None) -> None:
        if opt_params is not None:
            self.optimizer_params = opt_params
        specs = {
            k: v
            for k, v in self.optimizer_params.items()
            if not isinstance(v, OptimizerParams.NoneOptimizer)
        }
        if not specs:
            self.remove_optimizer()
            return
        spec_types = {type(v) for v in specs.values()}
        if len(spec_types) > 1:
            raise ValueError(f"All parameter groups must use one optimizer type, got {spec_types}")

        groups = self.get_optimization_parameters()
        if set(groups) != set(specs):
            raise ValueError(
                f"optimizer_params keys {set(specs)} do not match parameter groups "
                f"{set(groups)} from {type(self).__name__}.get_optimization_parameters()"
            )
        param_groups = [{"params": groups[k], **specs[k].params()} for k in groups]
        self._optimizer = self._build_optimizer(next(iter(specs.values())), param_groups)

    @staticmethod
    def _build_optimizer(spec: OptimizerParamsType, param_groups: list[dict]):
        match spec:
            case OptimizerParams.Adam():
                return torch.optim.Adam(param_groups)
            case OptimizerParams.AdamW():
                return torch.optim.AdamW(param_groups)
            case OptimizerParams.SGD():
                return torch.optim.SGD(param_groups)
            case OptimizerParams.LBFGS():
                hyper = [{k: v for k, v in g.items() if k != "params"} for g in param_groups]
                if any(h != hyper[0] for h in hyper):
                    raise ValueError("LBFGS does not support per-group hyperparameters")
                params = [p for g in param_groups for p in g["params"]]
                return torch.optim.LBFGS(params, **hyper[0])
            case _:
                raise NotImplementedError(f"Unknown optimizer type: {spec}")

    def set_scheduler(
        self,
        scheduler_params: SchedulerParamsType | dict | None = None,
        num_iter: int | None = None,
    ) -> None:
        if scheduler_params is not None:
            self.scheduler_params = scheduler_params
        if self._optimizer is None:
            self._scheduler = None
            return
        base_lr = max(g["lr"] for g in self._optimizer.param_groups)
        kwargs = self._scheduler_params.params(base_lr, num_iter=num_iter)
        sched = torch.optim.lr_scheduler
        match self._scheduler_params:
            case SchedulerParams.NoneScheduler():
                self._scheduler = None
            case SchedulerParams.Plateau():
                self._scheduler = sched.ReduceLROnPlateau(self._optimizer, **kwargs)
            case SchedulerParams.Exponential():
                self._scheduler = sched.ExponentialLR(self._optimizer, **kwargs)
            case SchedulerParams.CosineAnnealing():
                self._scheduler = sched.CosineAnnealingLR(self._optimizer, **kwargs)
            case _:
                raise ValueError(f"Unknown scheduler type: {self._scheduler_params}")

    def step_optimizer(self, closure: Callable[[], torch.Tensor] | None = None):
        """Step the optimizer; returns the closure's loss when a closure is given."""
        if self._optimizer is None:
            return None
        return self._optimizer.step(closure)

    def zero_optimizer_grad(self) -> None:
        if self._optimizer is not None:
            self._optimizer.zero_grad()

    def step_scheduler(self, loss: float | None = None) -> None:
        if self._scheduler is None:
            return
        if isinstance(self._scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if loss is not None:
                self._scheduler.step(loss)
        else:
            self._scheduler.step()

    def has_optimizer(self) -> bool:
        return self._optimizer is not None

    def get_current_lr(self) -> float:
        return self._optimizer.param_groups[0]["lr"] if self._optimizer is not None else 0.0

    def remove_optimizer(self) -> None:
        self._optimizer = None
        self._optimizer_params = {self.DEFAULT_OPTIMIZER_KEY: OptimizerParams.NoneOptimizer()}
        self._scheduler = None
        self._scheduler_params = SchedulerParams.NoneScheduler()

    def reset_optimizer(self) -> None:
        """Rebuild the optimizer and scheduler from their stored params (clears momentum)."""
        self.set_optimizer(self._optimizer_params)
        self.set_scheduler(self._scheduler_params)
