"""Ray diagrams and spot diagrams (matplotlib), and 3D rays (pyvista, ``viz`` extra)."""
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from em_ray_tracer.components.thin import Aperture, Biprism, Deflector, Plane, ThinLens
from em_ray_tracer.rays import Rays, X, Y
from em_ray_tracer.solvers import IntegratorParams, ODESolver, Trace
from em_ray_tracer.system import OpticalSystem
from em_ray_tracer.tracer import RayTracer


def plot_rays(
    trace: Trace,
    system: OpticalSystem | None = None,
    ax: plt.Axes | None = None,
    coordinate: str = "x",
    color: str | None = "C2",
    linewidth: float = 0.8,
    off_axis_scale="m",
    on_axis_scale="m",
    invert_orientation=False,
    max_off_axis=False,
    **kwargs,
) -> plt.Axes | tuple[plt.Axes, float]:
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
    _UNIT_SCALE = {"m": 1.0, "mm": 1e3}
    

    def _scale(unit: str) -> float:
        try:
            if unit is None:
                return _UNIT_SCALE["m"]
            return _UNIT_SCALE[unit]
        except KeyError:
            raise ValueError(f"Unknown scale {unit!r}, expected one of {list(_UNIT_SCALE)}") from None

    # ---- inside the plotting function, replacing the if/else block ----
    r = r * _scale(off_axis_scale)
    z = z * _scale(on_axis_scale)

    off_label = f"{coordinate} [{off_axis_scale}]"
    on_label = f"z [{on_axis_scale}]"

    if invert_orientation:           # z vertical, r horizontal
        ax.plot(r, z, color=color, linewidth=linewidth, **kwargs)
        if off_axis_scale : 
            ax.set_xlabel(off_label)
        if on_axis_scale : 
            ax.set_ylabel(on_label)
    else:                            # z horizontal, r vertical
        ax.plot(z, r, color=color, linewidth=linewidth, **kwargs)
        if on_axis_scale : 
            ax.set_xlabel(on_label)
        if off_axis_scale : 
            ax.set_ylabel(off_label)
    
    if system is not None:
        draw_components(system, ax,invert_orientation=invert_orientation,r_scale=_scale(off_axis_scale))
    if max_off_axis:
        return ax, np.nanmax(r)
    return ax


def draw_components(system: OpticalSystem, ax: plt.Axes,invert_orientation: bool = False ,r_scale : float = 1) -> None:
    """Mark lenses, apertures, deflectors and planes on a ray diagram."""
    if invert_orientation:
        lo, hi = ax.get_xlim()
        for c in system:
            z = float(c.z.detach())
            if isinstance(c, ThinLens):
                ax.axhline(z, color="k", linewidth=1.5, alpha=0.6)
                ax.annotate(c.name, (hi, z), fontsize=8, ha="center", va="bottom")
            elif isinstance(c, Aperture):
                r = r_scale*float(c.radius.detach())
                far = 10 * max(abs(lo), abs(hi), r)  # past any later ylim change
                ax.hlines([z, z], [-far, r], [-r, far], color="k", linewidth=4)
            elif isinstance(c, Plane):
                ax.axhline(z, color="k", linestyle="--", linewidth=0.8)
                ax.annotate(c.name, (lo,z), fontsize=8, ha="center", va="top")
            elif isinstance(c, (Deflector, Biprism)):
                ax.plot([z], [0.0], marker="|", markersize=12, color="C3")
        ax.set_xlim(lo, hi)
    else:
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


def _run_case(i,step_size,system,rays,z_end):
    torch.set_num_threads(1)
    if i==0: 
        solver=ODESolver.paraxial(integrator=IntegratorParams.RK4(dz=step_size))
    else: 
        solver=ODESolver.nonparaxial(expansion_order=2*i-1,integrator=IntegratorParams.RK4(dz=step_size))

    trace=RayTracer(system=system,rays=rays,solver=solver,z_end=z_end).trace(record=True)
    return trace

class Standardplots:

    @staticmethod
    def single(trace: Trace,system: OpticalSystem, coordinate: str = "x",title: str|None = None) -> None:
        """single plot of system trajectory along coordinate, default "x"."""
        ax, r_max = plot_rays(trace, system,coordinate=coordinate,max_off_axis=True)
        try: 
            ax.set_ylim(-r_max*1.5, r_max*1.5)
        except ValueError as err:
            raise Warning("the axis limits could not be set, the rays blow up to inf") from err
        
        if title :
            ax.set_title(title)
        if len(sys.argv) > 1:
            plt.savefig(sys.argv[1], dpi=150, bbox_inches="tight")
        else:
            plt.show()

    @staticmethod
    def multi_order(
        system: OpticalSystem,
        rays : Rays, 
        highest_order: int = 5, 
        step_size: float = 1e-4,
        show_fields: bool =False,
        z_end: float|None = None,
        fname: str|None = None,
        POLC_domain: tuple[float,float]|None=None
        ) -> None:

        # --- setup ---
        N=(highest_order+1)//2+1
        t0=time()
        

        if show_fields:
            offset=1
            z_space, phi_space,b_space=system.fields(z_start=rays.z,z_end=z_end)
            z_space=z_space.detach().cpu().numpy()
            phi_space=phi_space.detach().cpu().numpy()
            b_space=b_space.detach().cpu().numpy()
            #--- normalize ---
            if np.max(abs(phi_space)) != 0 :
                phi_space=phi_space/np.max(abs(phi_space)) 
            if np.max(abs(b_space)) !=0 :
                b_space=b_space/np.max(abs(b_space)) 
            

            _,axis=plt.subplots(1,N+1,sharey=True,figsize=(19.3,15),constrained_layout=True)
            axis[0].plot(phi_space,z_space,"r--",label="Phi")
            axis[0].plot(b_space,z_space,"b--",label="B_z")
            axis[0].set_xlabel("normalized fields")
            axis[0].set_title("on axis fields")
            axis[0].legend()

        else:
            offset=0
            fig, axis=plt.subplots(1,N,sharey=True,figsize=(19.3,15),constrained_layout=True)

        axis[0].set_ylabel("z[m]")
        axis[0].invert_yaxis()
        

        # --- seperate tasks across multiple cores ---
        worker=partial(_run_case,step_size=step_size,system=system,rays=rays,z_end=z_end)
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
            traces: list[Trace,]=list(ex.map(worker,range(N)))

        # --- plot ---
        r_set=[]
        for i,trace in enumerate(traces):
            if i == 0 :
                axis[i+offset].set_title("paraxial")
            else : 
                axis[i+offset].set_title(f"non paraxial order: {2*i-1}")
            if POLC_domain is not None:
                z_POLC, r2_POLC=trace.trajectory.plane_of_least_confusion(domain=POLC_domain)
                r_POLC= r2_POLC.item()*1e9 #mu
                POLC=Plane(z=float(z_POLC),name=f"POLC, radius : {round(r_POLC)} $nm$")
                system.add(POLC)
                axis[i+offset], r = plot_rays(trace,system=system,color="b",ax=axis[i+offset],off_axis_scale="mm",on_axis_scale=None,invert_orientation=True,max_off_axis=True)
                system.remove(f"POLC, radius : {round(r_POLC)} $nm$")
            else:
                axis[i+offset], r = plot_rays(trace,system=system,color="b",ax=axis[i+offset],off_axis_scale="mm",on_axis_scale=None,invert_orientation=True,max_off_axis=True)
            r_set.append(r)
        
        rmax=min(max(r_set),3*r_set[0]) 
        for i in range(offset,len(r_set)+offset): 
            axis[i].set_xlim(-1.5*rmax,1.5*rmax) 
        t1=time()
        print(f"Trajectory computation time: {round(t1-t0,1)}s")
        if fname is not None:
            plt.savefig(fname=fname,dpi=600,bbox_inches="tight")
        else:
            plt.show()
