from functools import partial

import jax
import MDAnalysis as mda
import mdtraj as md
import numpy as np

from jax import numpy as jnp
from scipy.constants import gas_constant


def scale_force(force, boxsize, temperature=298.0):
    """Scale force to unitless DM.

    Parameters
    ----------
    force : float
        Force in kJ/mol/nm
    boxsize : float
        Box size in nm
    temperature : float
        Temperature in K

    Returns
    -------
    float
        Scaled force.

    """
    return force * boxsize / (temperature * gas_constant * 1e-3)


def load_gromacs_pullx(pullx_file, gro_file, stride=1, nrows_max=None):
    """Load Gromacs pullx file and convert to unitless DM.

    Parameters
    ----------
    pullx_file : str
        Path to the pullx file.
    gro_file : str
        Path to the gro file.
    stride : int
        Stride for the trajectory. Default is 1.
    nrows_max : int, optional
        Maximum number of rows to load. If None, load all data. Default is None.

    Returns
    -------
    tuple
        Xs : np.ndarray
            Scaled trajectory.
        traj : np.ndarray
            Trajectory in nm.
        dt : float
            Time step in ns.
        boxsize : float
            Box size in nm.

    """
    t, traj = np.loadtxt(pullx_file, skiprows=17, max_rows=nrows_max).T
    gro = md.load(gro_file)
    boxsize = gro.unitcell_lengths[0][-1]

    # load pullf
    pullf_file = pullx_file.replace('pullx', 'pullf')
    tf, traj_f = np.loadtxt(pullf_file, skiprows=17, max_rows=nrows_max).T
    if not np.allclose(t, tf):
        raise ValueError(f'Time vectors of {pullx_file} and {pullf_file} do not match.')

    t, traj = np.loadtxt(pullx_file, skiprows=17, max_rows=nrows_max).T

    # stride
    traj = traj[::stride]
    dt = t[stride] * 1e-3  # convert to ns

    # shift pbc
    traj = ((traj + 0.5 * boxsize) % boxsize) - 0.5 * boxsize

    # fix reference system pulling in -z instead of z
    traj *= -1
    traj_f *= -1

    Xs = ((traj / boxsize + 0.5) % 1).reshape(-1, 1)
    return Xs, traj, traj_f, dt, boxsize


def load_trajs(
    *,
    directory,
    ext_forces,
    pullx_basename,
    gro_basename,
    stride=1,
    temperature=298.0,
    nrows_max=None,
):
    """Load Gromacs pullx files.

    Parameters
    ----------
    directory : str
        Directory containing the pullx and gro files.
    ext_forces : list[float]
        List of external forces.
    pullx_basename : str
        Basename of the pullx files. Should contain {vel} placeholders.
    gro_basename : str
        Basename of the gro files. Should contain {vel} placeholders.
    stride : int
        Stride for the trajectory. Default is 1.
    temperature : float
        Temperature in K. Default is 298.0.
    nrows_max : int, optional
        Maximum number of rows to load. If None, load all data. Default is None.

    Returns
    -------
    tuple
        Xs : dict
            Dictionary of scaled trajectories for each external force.
        ys : dict
            Dictionary of scaled forces for each external force.
        boxsizes : dict
            Dictionary of box sizes for each external force.
        dt : float
            Time step in ns.
    """
    boxsizes = {}
    Xs = {}
    ys = {}

    for ext_force in ext_forces:
        # time in ps, traj in nm
        pullx_file = pullx_basename.format(vel=ext_force)
        gro_file = gro_basename.format(vel=ext_force)
        Xs_force, _, ys_force, dt, boxsize = load_gromacs_pullx(
            pullx_file=f'{directory}/{pullx_file}',
            gro_file=f'{directory}/{gro_file}',
            stride=stride,
            nrows_max=nrows_max,
        )
        Xs[ext_force] = Xs_force
        boxsizes[ext_force] = boxsize
        print(
            f'force: {ext_force:.2f} with {len(Xs_force):.0f}'
            f' frames and boxsize {boxsize:.4f} nm'
        )

        ys[ext_force] = scale_force(
            force=ys_force,
            boxsize=boxsize,
            temperature=temperature,
        )

    return Xs, ys, boxsizes, dt


def load_gromacs_trr(
    trr_file,
    gro_file,
    lipid='POPC',
    molecule='N3',
    unitless=True,
):
    """Load Gromacs trr file and convert to unitless DM.

    Parameters
    ----------
    trr_file : str
        Path to the trr file.
    gro_file : str
        Path to the gro file.
    lipid : str
        Lipid name. Default is 'POPC'.
    molecule : str
        Molecule name. Default is 'N3'.
    unitless : bool
        If True, return unitless DM. Default is True.

    Returns
    -------
    tuple
        zs : np.ndarray
            Scaled trajectory or in [nm].
        v_zs : np.ndarray
            Scaled velocities or in [nm/ns].
        fs : np.ndarray
            Scaled forces or in kJ/mol/nm.
        dt : float
            Time step in ns.
        boxsize : float
            Box size in nm.
    """
    universe = mda.Universe(gro_file, trr_file)
    lipid = universe.select_atoms(f'resname {lipid}')
    molecule = universe.select_atoms(f'resname {molecule}')
    n_frames = universe.trajectory.n_frames

    ts = np.zeros(n_frames, dtype=float)
    zs = np.zeros_like(ts)
    v_zs = np.zeros_like(ts)
    fs = np.zeros_like(ts)
    for idx, frame in enumerate(universe.trajectory):
        ts[idx] = frame.time
        zs[idx] = (molecule.center(None) - lipid.center(None))[-1] / 10  # in nm
        v_zs[idx] = molecule.velocities[0, -1] * 1e-4  # in nm/ns
        fs[idx] = molecule.forces[0, -1] * 10  # in kJ/mol/nm

    # get boxsize
    boxsize = universe.dimensions[2] / 10  # in nm
    dt = ts[1] - ts[0] * 1e-3  # in ns

    # shift pbc
    zs = ((zs + 0.5 * boxsize) % boxsize) - 0.5 * boxsize  # / boxsize

    if unitless:
        return (
            zs / boxsize + 0.5,
            v_zs * dt * boxsize,
            scale_force(fs, boxsize),
            ts,
            boxsize,
        )

    return zs, v_zs, fs, dt, boxsize


def unwrap_traj(traj, boxsize=1):
    """Unwrap a trajectory according to periodic boundary conditions.

    Parameters
    ----------
    traj : array_like
        The wrapped trajectory, values in [0,1].
    boxsize : float, optional
        The size of the box. Default is 1.

    Returns
    -------
    unwrapped : np.ndarray
        The unwrapped trajectory (in box units, not scaled by boxsize).
    """
    unwrapped = traj.copy().flatten()
    deltas = np.diff(unwrapped)
    jumps = np.zeros_like(unwrapped)
    jumps[1:][deltas > 0.3 * boxsize] = -boxsize
    jumps[1:][deltas < -0.3 * boxsize] = boxsize
    unwrapped += np.cumsum(jumps)
    return unwrapped * boxsize


@partial(jax.jit, static_argnames=['tau', 'bins'])
def estimate_diff_x(x, tau, dt=1.0, bins=200):
    """Compute position-dependent diffusion coefficient from traj.

    Parameters
    ----------
    x : List[jnp.ndarray]
        List of trajectories in [0, 1].
    tau : int
        The time lag in frames.
    dt : float, optional
        The time step in [ns]. Default is 1.0.
    bins : int, optional
        The number of bins for the histogram. Default is 200.

    Returns
    -------
    tuple
        diff_x : jnp.ndarray
            The diffusion coefficient for each bin.
        xs : jnp.ndarray
            The x values for each bin.

    """
    x_bins = jnp.linspace(0.0, 1, bins)
    xs = (x_bins[1:] + x_bins[:-1]) / 2

    diff_x = jnp.empty(len(x_bins) - 1)

    # check if x is single trajectory or multiple
    if isinstance(x, np.ndarray) and x.ndim == 1 or not isinstance(x, (list, tuple)):
        x = [x]

    dx = jnp.concatenate(
        [traj[tau:] - traj[:-tau] for traj in x],
        axis=0,
    )
    x = jnp.concatenate(
        [traj[:-tau] for traj in x],
        axis=0,
    )

    dx = jnp.where(
        jnp.abs(dx) < 0.5,
        dx,
        dx - jnp.sign(dx),
    )

    for idx_bin, (x_min, x_max) in enumerate(zip(x_bins[:-1], x_bins[1:])):
        mask_bin = jnp.logical_and(x >= x_min, x < x_max)
        mask_count = mask_bin.astype(int).sum()
        dx_mask = jnp.where(mask_bin, dx, 0.0)
        dx_mask_mean = jnp.where(mask_bin, dx_mask.sum() / mask_count, 0.0)

        diff_x = diff_x.at[idx_bin].set(
            jnp.sum(
                (dx_mask - dx_mask_mean) ** 2,
            )
            / mask_count
            / (2 * tau * dt)  # * boxsize ** 2
        )

    return diff_x, xs
