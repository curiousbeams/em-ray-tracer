"""Off-axis expansion of an axisymmetric field from its axial profile.

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

Implementation notes
--------------------
``axial.derivative(n, z)`` returns a dict ``{"magnetic": ..., "electrostatic": ...}``; there is no
field-kind attribute any more. Which of the two fields exist is read once from that dict (keys that
are present and not None). All the series (Bz, Bt, phiz, phit, phi) are polynomials in u = r^2
with coefficients ``const_k * d^(m_k)/dz^(m_k)`` of the axial profile, so they are evaluated together:

  * a one-off *plan* lists the distinct (kind, derivative order) pairs that are needed and a
    gather-index / constant matrix that turns those derivatives into a coefficient tensor
    ``(K, S, 1)`` (K = max number of terms, S = number of series; short series are zero padded);
  * ``axial.derivative`` is called once per distinct order (not once per series and term);
  * one fused Horner pass (K-1 ``addcmul`` calls) evaluates all S series at once.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, NamedTuple

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


class _Plan(NamedTuple):
    names: tuple            # series evaluated, in row order of the fused Horner result
    K: int                  # max number of terms over those series (rows of the coefficient tensor)
    orders: tuple           # distinct derivative orders -> one axial.derivative call each
    kinds_at: dict          # order -> kinds ("magnetic"/"electrostatic") read from that call
    idx: torch.Tensor       # (K, S) long: which derivative value feeds coefficient (k, s)
    cmat: torch.Tensor      # (K, S): constant of coefficient (k, s), 0 for padding


class LaplaceExpansion(Field3D):
    """The 3D field of ``axial`` expanded to ``order`` (transverse) and
    ``longitudinal_order`` (defaults to ``order``). Phi can also be returned using phi = True.
    The convention was used such that for order k the highest r term present in the force field expansion
    is r^k and in the energy r^k+1
    """

    # field kind -> series it needs
    _KIND_SERIES = {
        "magnetic": ("Bz", "Bt"),
        "electrostatic": ("phiz", "phit", "phi"),
    }

    # series -> (axial kind it is built from, derivative order of term k, constant of term k)
    _SERIES = {
        "Bz":   ("magnetic",      lambda k: 2 * k,     _c_axial),
        "Bt":   ("magnetic",      lambda k: 2 * k + 1, _c_trans),
        "phiz": ("electrostatic", lambda k: 2 * k + 1, _c_axial),
        "phit": ("electrostatic", lambda k: 2 * k + 2, _c_trans),
        "phi":  ("electrostatic", lambda k: 2 * k,     _c_axial),
    }

    def __init__(self, axial: AxialField,
                 order: int = 3,
                 longitudinal_order: int | None = None,
                 component: FieldComponent | None = None):
        super().__init__()
        self.axial: AxialField = axial
        self.order = order
        self.longitudinal_order = order if longitudinal_order is None else longitudinal_order
        self.component = component
        self.z_grid = None
        self.coeff_grid = None
        self.misses = 0
        self._precalc = False
        self._plan_cache = None

    # ------------------------------------------------------------------ plan
    def _present_kinds(self) -> tuple:
        """Which of the two fields the axial profile provides (one probe call, order 0)."""
        probe = self.axial.derivative(0, torch.zeros(1, dtype=DTYPE))
        kinds = tuple(k for k in self._KIND_SERIES if probe.get(k) is not None)
        if not kinds:
            raise ValueError("axial.derivative returned neither 'magnetic' nor 'electrostatic'")
        return kinds

    def _build_plan(self) -> _Plan:
        N, L = self.order, self.longitudinal_order
        counts = {                                   # number of terms of each series
            "Bz":   L // 2 + 1,
            "Bt":   max(0, (N - 1) // 2 + 1),
            "phiz": (L + 2) // 2,
            "phit": max(0, (N + 1) // 2),
            "phi":  (N + 1) // 2 + 1,
        }
        names = tuple(nm for kd in self._present_kinds() for nm in self._KIND_SERIES[kd])
        S, K = len(names), max(counts[nm] for nm in names)

        used = {(self._SERIES[nm][0], self._SERIES[nm][1](k))
                for nm in names for k in range(counts[nm])}
        sources = sorted(used, key=lambda t: (t[1], t[0]))      # (kind, order), by order
        pos = {key: i for i, key in enumerate(sources)}

        idx = torch.full((K, S), len(sources), dtype=torch.long)   # len(sources) -> the zero pad
        cmat = torch.zeros(K, S, dtype=DTYPE)
        for s, nm in enumerate(names):
            skind, order_of, c = self._SERIES[nm]
            for k in range(counts[nm]):
                idx[k, s] = pos[(skind, order_of(k))]
                cmat[k, s] = c(k)

        orders = tuple(sorted({n for _, n in sources}))
        kinds_at = {n: [kd for kd, m in sources if m == n] for n in orders}
        return _Plan(names, K, orders, kinds_at, idx, cmat)

    def _plan(self) -> _Plan:
        key = (self.order, self.longitudinal_order)
        if self._plan_cache is None or self._plan_cache[0] != key:
            self._plan_cache = (key, self._build_plan())
            self._precalc = False          # grid belongs to the old plan: call pre_calc again
        return self._plan_cache[1]

    # ---------------------------------------------------------- coefficients
    def _coefficients(self, z: torch.Tensor) -> torch.Tensor:
        """Coefficient tensor at z: (K, S, 1) for 0-d z, (G, K, S, 1) for z of shape (G,)."""
        p = self._plan()
        vals = []
        for n in p.orders:                         # one derivative call per distinct order
            d = self.axial.derivative(n, z)
            vals.extend(d[kd] for kd in p.kinds_at[n])
        vals.append(torch.zeros_like(vals[0]))     # padding entry (constant is 0 there)
        V = torch.stack(vals)                      # (M+1, *z.shape)
        extra = (1,) * (V.dim() - 1)
        cm = p.cmat.to(dtype=V.dtype, device=V.device)
        coef = V[p.idx.to(V.device)] * cm.reshape(*cm.shape, *extra)   # (K, S, *z.shape)
        if coef.dim() > 2:
            coef = coef.movedim(2, 0)              # (G, K, S)
        return coef.unsqueeze(-1)

    def _grid_index(self, z):
        raw = (float(z) - self.z0) / self.z_step
        i = round(raw)
        if abs(raw - i) > 1e-8 or not (0 <= i < self._G):   # < not <=
            return None
        return i

    def _coeffs(self, z):
        """K tensors of shape (S, 1) holding the Horner coefficients at z."""
        if self._precalc:
            i = self._grid_index(z)
            if i is not None:
                return self._grid_a[i]
            self.misses += 1
        zv = torch.as_tensor(z, dtype=DTYPE).reshape(1)
        return self._coefficients(zv)[0].unbind(0)

    def pre_calc(self, n_steps: int):
        if self.component is None:
            raise ValueError("please supply component in order to use precalc")
        z0 = float(self.component.z_start - self.component.z)
        z1 = float(self.component.z_end - self.component.z)
        z_space = torch.linspace(z0, z1, 2 * n_steps + 1, dtype=DTYPE)

        coef = self._coefficients(z_space)         # (G, K, S, 1), part of the graph
        self.coeff_grid = coef
        # Python-side split once: no select/unbind autograd ops in the hot path
        # (a select on the big tensor costs an O(G) zero-fill in backward per call).
        self._grid_a = tuple(row.unbind(0) for row in coef.unbind(0))
        self.z0, self.z_step, self.z_grid = z0, (z1 - z0) / (2 * n_steps), z_space
        self._G = len(z_space)
        self.misses = 0
        self._precalc = True

    # ------------------------------------------------------- fused evaluation
    @staticmethod
    def _horner(a, u):
        """All series at once. a: K tensors (S, 1), u: (n,) -> (S, n), row s = sum_k a[k][s] u^k."""
        K = len(a)
        if K == 1:
            return a[0].expand(-1, u.shape[0])
        acc = torch.addcmul(a[-2], a[-1], u)
        for k in range(K - 3, -1, -1):
            acc = torch.addcmul(a[k], acc, u)
        return acc

    def evaluate(
        self, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, voltage: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:

        p = self._plan()
        u = torch.addcmul(x * x, y, y)
        s = dict(zip(p.names, self._horner(self._coeffs(z), u).unbind(0)))
        n = len(x)
        if "phiz" in s:
            E = torch.stack((x * s["phit"], y * s["phit"], s["phiz"]), dim=-1)
        else:
            E = x.new_zeros(n, 3)
        if "Bz" in s:
            B = torch.stack((x * s["Bt"], y * s["Bt"], s["Bz"]), dim=-1)
        else:
            B = x.new_zeros(n, 3)
        if voltage is None:
            return (E, B, None)
        # no electrostatic part: the potential is just the constant voltage
        phi = s["phi"] + voltage if "phi" in s else x.new_zeros(n) + voltage
        return (E, B, phi)