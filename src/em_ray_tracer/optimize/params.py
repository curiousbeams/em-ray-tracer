"""Optimizer and scheduler configs as dataclasses.

Each namespace holds one dataclass per torch optimizer or scheduler. A config can be given as
the dataclass or as a plain dict (e.g. from YAML), which ``parse_dict`` turns into the
dataclass by its ``name`` (or ``type``) key.

Examples
--------
>>> OptimizerParams.LBFGS()
>>> OptimizerParams.parse_dict({"name": "adam", "lr": 1e-2})
>>> SchedulerParams.Plateau(factor=0.5, patience=10)
"""

from dataclasses import dataclass
from typing import ClassVar, Literal


def _parse(namespace, d: dict, default: str | None = None):
    d = dict(d)
    name = d.pop("name", None) or d.pop("type", None) or default
    if name is None:
        raise ValueError("Must provide either a 'name' or a 'type' key")
    name = name.__name__ if isinstance(name, type) else str(name)
    for variant in vars(namespace).values():
        if isinstance(variant, type) and name.lower() in (variant._name, variant.__name__.lower()):
            return variant(**d)
    raise ValueError(f"Unknown {namespace.__name__} type: {name}")


class OptimizerParams:
    """Namespace of optimizer configs, passed to ``OptimizerMixin.set_optimizer``.

    LBFGS is the usual choice here: lens-parameter problems have few, smooth parameters, where
    a quasi-Newton method converges in a handful of iterations. Adam/AdamW suit noisy or
    many-parameter problems (e.g. fitting a sampled field).
    """

    @dataclass
    class Adam:
        """``torch.optim.Adam``."""

        lr: float = 1e-3
        betas: tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-8
        weight_decay: float = 0
        _name: ClassVar[str] = "adam"

        def params(self) -> dict:
            return {
                "lr": self.lr,
                "betas": self.betas,
                "eps": self.eps,
                "weight_decay": self.weight_decay,
            }

    @dataclass
    class AdamW:
        """``torch.optim.AdamW`` (decoupled weight decay)."""

        lr: float = 1e-3
        betas: tuple[float, float] = (0.9, 0.999)
        eps: float = 1e-8
        weight_decay: float = 0
        _name: ClassVar[str] = "adamw"

        def params(self) -> dict:
            return {
                "lr": self.lr,
                "betas": self.betas,
                "eps": self.eps,
                "weight_decay": self.weight_decay,
            }

    @dataclass
    class SGD:
        """``torch.optim.SGD``."""

        lr: float = 1e-3
        momentum: float = 0
        dampening: float = 0
        weight_decay: float = 0
        nesterov: bool = False
        _name: ClassVar[str] = "sgd"

        def params(self) -> dict:
            return {
                "lr": self.lr,
                "momentum": self.momentum,
                "dampening": self.dampening,
                "weight_decay": self.weight_decay,
                "nesterov": self.nesterov,
            }

    @dataclass
    class LBFGS:
        """``torch.optim.LBFGS``. Needs a closure, which ``OptimizerMixin.step_optimizer``
        passes through.

        The tolerances are absolute, and much tighter than torch's defaults (1e-7, 1e-9).
        Losses here are squared SI lengths: an image plane 2 um off gives a loss of 4e-12,
        which torch's defaults would already call converged.
        """

        lr: float = 1.0
        max_iter: int = 20
        max_eval: int | None = None
        tolerance_grad: float = 1e-14
        tolerance_change: float = 1e-24
        history_size: int = 100
        line_search_fn: Literal["strong_wolfe"] | None = "strong_wolfe"
        _name: ClassVar[str] = "lbfgs"

        def params(self) -> dict:
            return {
                "lr": self.lr,
                "max_iter": self.max_iter,
                "max_eval": self.max_eval,
                "tolerance_grad": self.tolerance_grad,
                "tolerance_change": self.tolerance_change,
                "history_size": self.history_size,
                "line_search_fn": self.line_search_fn,
            }

    @dataclass
    class NoneOptimizer:
        """Disables optimization; ``set_optimizer`` removes the optimizer."""

        _name: ClassVar[str] = "none"

        def params(self) -> dict:
            return {}

    @classmethod
    def parse_dict(cls, d: dict) -> "OptimizerParamsType":
        """Build a config from ``{"name": "adam", "lr": 1e-3, ...}`` (``"type"`` also works)."""
        return _parse(cls, d)


OptimizerParamsType = (
    OptimizerParams.Adam
    | OptimizerParams.AdamW
    | OptimizerParams.SGD
    | OptimizerParams.LBFGS
    | OptimizerParams.NoneOptimizer
)


class SchedulerParams:
    """Namespace of learning-rate scheduler configs, passed to
    ``OptimizerMixin.set_scheduler``."""

    @dataclass
    class Plateau:
        """``ReduceLROnPlateau``. ``min_lr`` defaults to ``min_lr_factor * base_lr``."""

        mode: Literal["min", "max"] = "min"
        min_lr_factor: float = 1 / 20
        min_lr: float | None = None
        factor: float = 0.5
        patience: int = 10
        threshold: float = 1e-5
        cooldown: int = 50
        _name: ClassVar[str] = "plateau"

        def params(self, base_LR: float, num_iter: int | None = None) -> dict:
            min_lr = self.min_lr if self.min_lr is not None else self.min_lr_factor * base_LR
            return {
                "mode": self.mode,
                "factor": self.factor,
                "patience": self.patience,
                "threshold": self.threshold,
                "min_lr": min_lr,
                "cooldown": self.cooldown,
            }

    @dataclass
    class Exponential:
        """``ExponentialLR``. If ``factor`` is set, gamma is chosen so the LR falls by that
        factor over ``num_iter`` steps."""

        gamma: float = 0.9
        factor: float | None = None
        num_iter: int | None = None
        _name: ClassVar[str] = "exponential"

        def params(self, base_LR: float, num_iter: int | None = None) -> dict:
            n = self.num_iter if self.num_iter is not None else num_iter
            if self.factor is not None:
                if n is None:
                    raise ValueError("num_iter must be set when factor is used")
                return {"gamma": self.factor ** (1.0 / n)}
            return {"gamma": self.gamma}

    @dataclass
    class CosineAnnealing:
        """``CosineAnnealingLR``. ``T_max`` defaults to the number of iterations."""

        eta_min: float = 1e-7
        T_max: int | None = None
        _name: ClassVar[str] = "cosine_annealing"

        def params(self, base_LR: float, num_iter: int | None = None) -> dict:
            T_max = self.T_max if self.T_max is not None else num_iter
            if T_max is None:
                raise ValueError("T_max must be set if num_iter is not provided")
            return {"T_max": T_max, "eta_min": self.eta_min}

    @dataclass
    class NoneScheduler:
        """Disables LR scheduling."""

        _name: ClassVar[str] = "none"

        def params(self, base_LR: float, num_iter: int | None = None) -> dict:
            return {}

    @classmethod
    def parse_dict(cls, d: dict) -> "SchedulerParamsType":
        """Build a config from ``{"name": "plateau", ...}``; no name means no scheduler."""
        return _parse(cls, d, default="none")


SchedulerParamsType = (
    SchedulerParams.Plateau
    | SchedulerParams.Exponential
    | SchedulerParams.CosineAnnealing
    | SchedulerParams.NoneScheduler
)
