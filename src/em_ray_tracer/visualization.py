"""Ray diagrams and spot diagrams (matplotlib), and 3D rays (pyvista, ``viz`` extra)."""

import matplotlib.pyplot as plt
import numpy as np

from em_ray_tracer.components.thin import Aperture, Biprism, Deflector, Plane, ThinLens
from em_ray_tracer.rays import Rays, X, Y
from em_ray_tracer.solvers.base import Trace
from em_ray_tracer.system import OpticalSystem


def plot_rays(
    trace: Trace,
    system: OpticalSystem | None = None,
    ax: plt.Axes | None = None,
    coordinate: str = "x",
    color: str | None = "C2",
    linewidth: float = 0.8,
    **kwargs,
) -> plt.Axes:
    """Ray diagram (z vs x or y) from a trace recorded with ``record=True``.

    Rays stop where they die (at an aperture). Pass ``system`` to draw the components.
    """
    if trace.trajectory is None:
        raise ValueError("Trace with record=True to plot trajectories")
    ax = ax or plt.subplots(figsize=(8, 4))[1]
    traj = trace.trajectory
    z = traj.z.detach().cpu().numpy()
    column = {"x": X, "y": Y}[coordinate]
    r = traj.state[..., column].detach().cpu().numpy().reshape(len(z), -1)
    alive = traj.alive.detach().cpu().numpy().reshape(len(z), -1)
    # keep the first dead sample so a ray visibly ends at the aperture plane
    was_alive = np.vstack([alive[:1], alive[:-1]])
    r = np.where(was_alive, r, np.nan)
    ax.plot(z, r, color=color, linewidth=linewidth, **kwargs)
    if system is not None:
        draw_components(system, ax)
    ax.set_xlabel("z [m]")
    ax.set_ylabel(f"{coordinate} [m]")
    return ax


def draw_components(system: OpticalSystem, ax: plt.Axes) -> None:
    """Mark lenses, apertures, deflectors and planes on a ray diagram."""
    lo, hi = ax.get_ylim()
    for c in system:
        z = float(c.z.detach())
        if isinstance(c, ThinLens):
            ax.axvline(z, color="k", linewidth=1.5, alpha=0.6)
            ax.annotate(c.name, (z, hi), fontsize=8, ha="center", va="bottom")
        elif isinstance(c, Aperture):
            r = float(c.radius.detach())
            far = 10 * max(abs(lo), abs(hi), r)  # past any later ylim change
            ax.vlines([z, z], [-far, r], [-r, far], color="k", linewidth=4)
        elif isinstance(c, Plane):
            ax.axvline(z, color="k", linestyle="--", linewidth=0.8)
            ax.annotate(c.name, (z, lo), fontsize=8, ha="center", va="top")
        elif isinstance(c, (Deflector, Biprism)):
            ax.plot([z], [0.0], marker="|", markersize=12, color="C3")
    ax.set_ylim(lo, hi)


def plot_spot(rays: Rays, ax: plt.Axes | None = None, **kwargs) -> plt.Axes:
    """Spot diagram of the alive rays in a plane."""
    ax = ax or plt.subplots(figsize=(4, 4))[1]
    alive = rays.alive.detach().cpu().numpy()
    x = rays.x.detach().cpu().numpy()[alive]
    y = rays.y.detach().cpu().numpy()[alive]
    ax.scatter(x, y, s=kwargs.pop("s", 4), **kwargs)
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    return ax


def plot_rays_3d(trace: Trace, plotter=None, **kwargs):
    """3D rays with pyvista. Requires the ``viz`` extra: ``uv sync --extra viz``."""
    try:
        import pyvista as pv
    except ImportError as err:
        raise ImportError("plot_rays_3d needs pyvista: uv sync --extra viz") from err
    if trace.trajectory is None:
        raise ValueError("Trace with record=True to plot trajectories")
    traj = trace.trajectory
    z = traj.z.detach().cpu().numpy()
    x = traj.state[..., X].detach().cpu().numpy().reshape(len(z), -1)
    y = traj.state[..., Y].detach().cpu().numpy().reshape(len(z), -1)
    plotter = plotter or pv.Plotter()
    for i in range(x.shape[1]):
        points = np.column_stack([x[:, i], y[:, i], z])
        plotter.add_mesh(pv.lines_from_points(points), **kwargs)
    return plotter
