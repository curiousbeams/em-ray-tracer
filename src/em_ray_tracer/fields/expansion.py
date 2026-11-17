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
from typing import TYPE_CHECKING

import torch
from math import factorial

from em_ray_tracer.fields.base import AxialField, Field3D
from em_ray_tracer.constants import DTYPE
if TYPE_CHECKING:
    from em_ray_tracer.components.field_lenses import FieldComponent


def _c_axial(k: int) -> float:
    return (-0.25) ** k / factorial(k) ** 2

def _c_trans(k: int) -> float:          # includes the -1/2 prefactor
    return -0.5 * (-0.25) ** k / (factorial(k) * factorial(k + 1))

class LaplaceExpansion(Field3D):
    """The 3D field of ``axial`` expanded to ``order`` (transverse) and
    ``longitudinal_order`` (defaults to ``order``). Phi can also be returned using phi = True.
    The convention was used such that for order k the highest r term present in the force field expansion 
    is r^k and in the energy r^k+1 
    """

    # Horner ratios, cached. With c_k = (-1/4)^k/(k!)^2:
    #   axial series      (B_z, phi_z, phi):  c_{k+1}/c_k = -1/(4 (k+1)^2)
    #   transverse series (B_x/B_y, phi_x/phi_y): ratio = -1/(4 (k+1)(k+2)), prefactor -1/2



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
        self.z_grid=None
        self._precalc=False

    _NEEDED = {
        "magnetic": ("Bz", "Bt"),
        "electrostatic": ("phiz", "phit", "phi"),
        "static-electromagnetic": ("Bz", "Bt", "phiz", "phit", "phi"),
    }

    def _specs(self):
        # name: (axial kind, derivative order of term k, number of terms, constant)
        N, L = self.order, self.longitudinal_order
        return {
            "Bz":   ("magnetic",      lambda k: 2 * k,     L // 2 + 1,               _c_axial),
            "Bt":   ("magnetic",      lambda k: 2 * k + 1, max(0, (N - 1) // 2 + 1), _c_trans),
            "phiz": ("electrostatic", lambda k: 2 * k + 1, (L + 2) // 2,             _c_axial),
            "phit": ("electrostatic", lambda k: 2 * k + 2, max(0, (N + 1) // 2),     _c_trans),
            "phi":  ("electrostatic", lambda k: 2 * k,     (N + 1) // 2 + 1,         _c_axial),
        }

    @staticmethod
    def _horner(a, u):
        """sum_k a[k] u^k, with a[k] 0-d tensors that carry grad."""
        n = len(a)
        if n == 0:
            return torch.zeros_like(u)
        if n == 1:
            return torch.zeros_like(u) + a[0]
        acc = torch.addcmul(a[-2], u, a[-1])
        for k in range(n - 3, -1, -1):
            acc = torch.addcmul(a[k], acc, u)
        return acc

    def _grid_index(self, z):
        raw = (float(z) - self.z0) / self.z_step
        i = round(raw)
        if abs(raw - i) > 1e-8 or not (0 <= i < len(self.z_grid)):   # < not <=
            return None
        return i

    def _coeffs(self, z):
        if self._precalc:
            i = self._grid_index(z)
            if i is not None:
                return {nm: (T[i].unbind(0) if T is not None else ())
                        for nm, T in self.coeff_grid.items()}
            self.misses += 1
        return self._coeffs_at(z)

    def _coeffs_at(self, z):                       # slow path, same maths
        specs, out = self._specs(), {}
        for name in self._NEEDED[self.axial.kind]:
            kind, deriv, n, c = specs[name]
            out[name] = tuple(c(k) * self.axial.derivative(deriv(k), z, kind_requested=kind)
                              for k in range(n))
        return out

    def pre_calc(self, n_steps: int):
        if self.component is None:
            raise ValueError("please supply component in order to use precalc")
        z0 = float(self.component.z_start - self.component.z)
        z1 = float(self.component.z_end - self.component.z)
        z_space = torch.linspace(z0, z1, 2 * n_steps + 1, dtype=DTYPE)
        specs = self._specs()

        grid = {}
        for name in self._NEEDED[self.axial.kind]:
            kind, deriv, n, c = specs[name]
            if n == 0:
                grid[name] = None
                continue
            terms = [c(k) * self.axial.derivative(deriv(k), z_space, kind_requested=kind)
                     for k in range(n)]
            grid[name] = torch.stack(terms, dim=1)      # (G, n), part of the graph

        self.coeff_grid = grid
        self.z0, self.z_step, self.z_grid = z0, (z1 - z0) / (2 * n_steps), z_space
        self.misses = 0
        self._precalc = True

    

    # --- fused series (u = x^2+y^2 passed in) ---
    def _Bz_series(self, u, d):   return self._horner(d["Bz"], u)
    def _Bt_series(self, u, d):   return self._horner(d["Bt"], u)
    def _phiz_series(self, u, d): return self._horner(d["phiz"], u)
    def _phit_series(self, u, d): return self._horner(d["phit"], u)
    def _phi_series(self, u, d, voltage): return self._horner(d["phi"], u) + voltage



    def evaluate(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, voltage: float|None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        
        kind = self.axial.kind
        d = self._coeffs(z)
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
    
    
