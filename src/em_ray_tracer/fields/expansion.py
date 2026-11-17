"""Off-axis expansion of an axisymmetric field from its axial profile. STUB.

Laplace's equation fixes the off-axis field term by term from the axial profile F(z):

    F_z = sum_{n=0..N}  (-1)^n / (2^(2n) (n!)^2)       r^(2n)   F^(2n)(z)
    F_x = sum_{n=1..N}  (-1)^n / (2^(2n-1) (n-1)! n!)  x r^(2n-2) F^(2n-1)(z)
    F_y = the same with y in place of x

Truncating at N leaves exactly one unpaired term in the divergence,
(-1)^N / (2^(2N) (N!)^2) r^(2N) F^(2N+1)(z); asserting that identity is a good test.

Order 1 is the paraxial field and order 3 is where spherical aberration appears. Truncating the
longitudinal sum one order below the transverse one gives the strictly linearized field (the
``r^2 B''`` term in B_z is most of a magnetic lens's spherical aberration). Reference
implementation: ``laplaceExpansion`` in the em-widgets kit (kit/nonparaxial.js).
"""

from __future__ import annotations
from itertools import product
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from em_ray_tracer.fields.base import AxialField, Field3D
from em_ray_tracer.constants import DTYPE
if TYPE_CHECKING:
    from em_ray_tracer.components.field_lenses import FieldComponent


class LaplaceExpansion(Field3D):
    """The 3D field of ``axial`` expanded to ``order`` (transverse) and
    ``longitudinal_order`` (defaults to ``order``). Phi can also be returned using phi = True.
    The convention was used such that for order k the highest r term present in the force field expansion 
    is r^k and in the energy r^k+1 
    """

    # Horner ratios, cached. With c_k = (-1/4)^k/(k!)^2:
    #   axial series      (B_z, phi_z, phi):  c_{k+1}/c_k = -1/(4 (k+1)^2)
    #   transverse series (B_x/B_y, phi_x/phi_y): ratio = -1/(4 (k+1)(k+2)), prefactor -1/2

    
    @lru_cache(maxsize=None)
    def _ratios_axial(self,n: int) -> tuple:
        return tuple(-0.25 / (k + 1) ** 2 for k in range(n))

    @lru_cache(maxsize=None)
    def _ratios_transverse(self,n: int) -> tuple:
        return tuple(-0.25 / ((k + 1) * (k + 2)) for k in range(n))

    def _horner(self,d: list, u: torch.Tensor, r: tuple) -> torch.Tensor:
        if not d:
            return torch.zeros_like(u)
        acc = d[-1]
        for k in range(len(d) - 2, -1, -1):
            acc = torch.addcmul(d[k], acc, u, value=r[k])
        if acc.shape != u.shape:  # single term and z shared by all rays -> derivative is a scalar
            acc = acc.expand(u.shape)
        return acc


    # ---- inside your class ----
    def __init__(self, axial: AxialField, 
                 order: int = 3, 
                 longitudinal_order: int | None = None, 
                 component: FieldComponent|None = None):
        super().__init__()
        self.axial: AxialField = axial
        self.order = order
        self.longitudinal_order = order if longitudinal_order is None else longitudinal_order
        self.component=component
        self.cache={}
        self.z_grid=None
        self._precalc=False

    # every (n, kind) axial derivative is evaluated at most once per z
    def _memo(self, z):
        cache = {}
        if self._precalc:
            cache_grid: dict = self.cache_grid
            index=self.estimate_index(z)
            if index is None:
                pass
            else:
                cache={k:cache_grid[k][index] for k in cache_grid.keys()}

        derivative = self.axial.derivative
        def d(n, kind):
            key = (n,kind)
            v = cache.get(key)
            if v is None:
                if self._precalc: print("missing derivative",n,z)
                v = cache[key] = derivative(n, z, kind_requested=kind)
            return v
        return d

    # --- fused series (u = x^2+y^2 passed in) ---
    def _Bz_series(self, u, d):
        n = len(range((self.longitudinal_order) // 2 + 1))
        return self._horner([d(2 * k, "magnetic") for k in range(n)], u, self._ratios_axial(n))

    def _Bt_series(self, u, d):  # B_x = x * this, B_y = y * this
        n = len(range((self.order - 1) // 2 + 1))
        acc = self._horner([d(2 * k + 1, "magnetic") for k in range(n)], u, self._ratios_transverse(n))
        return acc * -0.5 if n else acc

    def _phiz_series(self, u, d):
        n = len(range((self.longitudinal_order + 2) // 2))
        return self._horner([d(2 * k + 1, "electrostatic") for k in range(n)], u, self._ratios_axial(n))

    def _phit_series(self, u, d):  # phi_x = x * this, phi_y = y * this
        n = max(0, (self.order + 1) // 2)
        acc = self._horner([d(2 * k + 2, "electrostatic") for k in range(n)], u, self._ratios_transverse(n))
        return acc * -0.5 if n else acc

    def _phi_series(self, u, d, voltage):
        n = len(range((self.order + 1) // 2 + 1))
        if n == 0:
            return torch.ones_like(u) * voltage
        return self._horner([d(2 * k, "electrostatic") for k in range(n)], u, self._ratios_axial(n)) + voltage

    # --- public API: same names/signatures ---
    def B_z(self, x, y, z):   return self._Bz_series(x * x + y * y, self._memo(z))
    def B_x(self, x, y, z):   return x * self._Bt_series(x * x + y * y, self._memo(z))
    def B_y(self, x, y, z):   return y * self._Bt_series(x * x + y * y, self._memo(z))
    def phi_z(self, x, y, z): return self._phiz_series(x * x + y * y, self._memo(z))
    def phi_x(self, x, y, z): return x * self._phit_series(x * x + y * y, self._memo(z))
    def phi_y(self, x, y, z): return y * self._phit_series(x * x + y * y, self._memo(z))
    def phi(self, x, y, z, voltage: float):
        return self._phi_series(x * x + y * y, self._memo(z), voltage)

    def evaluate(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, voltage: float|None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        
        kind = self.axial.kind
        d = self._memo(z)
        u = torch.addcmul(x * x, y, y)
        n = len(x)
        match kind:
            case "magnetic":
                E = x.new_zeros(n, 3)
                t = self._Bt_series(u, d)
                B = torch.stack((x * t, y * t, self._Bz_series(u, d)), dim=-1)
            case "electrostatic":
                t = self._phit_series(u, d)
                E = torch.stack((x * t, y * t, self._phiz_series(u, d)), dim=-1)
                B = x.new_zeros(n, 3)
            case "static-electromagnetic":
                t = self._phit_series(u, d)
                E = torch.stack((x * t, y * t, self._phiz_series(u, d)), dim=-1)
                t = self._Bt_series(u, d)
                B = torch.stack((x * t, y * t, self._Bz_series(u, d)), dim=-1)
            case _:
                raise ValueError(f"Unknown field kind: {kind!r}")
        if voltage is not None:
            return (E, B, self._phi_series(u, d, voltage))
        return (E, B)
    
    def pre_calc(self,n_steps:int):
        if self.component is None:
            raise ValueError("please supply component in order to use precalc")
        z0=float(self.component.z_start-self.component.z)
        z1=float(self.component.z_end-self.component.z)
        z_step=(z1-z0)/(2*n_steps)
        
        z_space=torch.linspace(z0,z1,2*n_steps+1,dtype=DTYPE)
        n_space=range(0,self.order+2)
        match self.axial.kind:
            case "magnetic": kind_space=["magnetic",]
            case "electrostatic": kind_space=["electrostatic",]
            case "static-electromagnetic": kind_space=["electrostatic","magnetic"]


        cache_grid={(n,k): self.axial.derivative(n,z_space,kind_requested=k) for n in n_space for k in kind_space}

        self.z0=z0
        self.z_step=z_step
        self.z_grid=z_space
        self.cache_grid=cache_grid
        self._precalc=True
    
    def estimate_index(self,z):
        raw=float((z-self.z0)/self.z_step)
        index=round(raw)
        abs_error=raw-index
        if abs(abs_error)>1e-8 or not (0 <= index <= len(self.z_grid)) :  
            print(abs_error,":abs error is larger then 1e-8 for z:",z.item())
            return None
        return index
