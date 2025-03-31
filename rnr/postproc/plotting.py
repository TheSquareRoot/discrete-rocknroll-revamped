import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray

from ..core.distribution import AdhesionDistribution, SizeDistribution
from ..core.flow import Flow
from ..postproc.results import TemporalResults, FractionVelocityResults
from ..utils.misc import read_exp_data, log_norm


# ======================================================================================================================
# DISTRIBUTION PLOTS
# ======================================================================================================================

def plot_size_distribution(size_distrib: SizeDistribution, name: str, scale: str = 'linear',**kwargs) -> None:
    """
    Basic bar plot of the size distribution.

    Args:
        size_distrib (SizeDistribution): Size distribution to plot.
        name (str): Name of the configuration file. Will be used to save the figure in the right folder.
        scale (str, optional): Scale of the x_axis. Defaults to 'linear'.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(size_distrib.radii, size_distrib.weights, **kwargs)

    ax.set_xscale(scale)
    ax.set_ylim([0.0, 1.1 * np.max(size_distrib.weights)])

    ax.set_xlabel('radius [µm]')
    ax.set_ylabel('weight')

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(f'figs/{name}/size_distrib.png', dpi=300)
    plt.close(fig)

def plot_adhesion_distribution(adh_distrib: AdhesionDistribution, name: str, i: int, norm: bool = True, scale: str = 'log', **kwargs) -> None:
    """
    Basic plot of the adhesion distribution of the i-th size bin.

    Args:
        adh_distrib (AdhesionDistribution): Adhesion distribution to plot.
        name (str): Name of the configuration file. Will be used to save the figure in the right folder.
        i (int): Index of the size bin.
        norm (bool, optional): Whether to normalize the adhesion forces. Defaults to True.
        scale (str, optional): Scale of the x-axis. Defaults to 'log'.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    if norm:
        ax.plot(adh_distrib.fadh_norm[i], adh_distrib.weights[i], **kwargs)
        ax.set_xlabel('Normalized adhesion force')
    else:
        ax.plot(adh_distrib.fadh[i], adh_distrib.weights[i], **kwargs)
        ax.set_xlabel('Adhesion force [N]')

    # Compute the median and display it
    med = adh_distrib.median(i, norm=norm)
    mean = adh_distrib.mean(i, norm=norm)

    ax.axvline(med, color='r', linestyle='-', label=f'Median = {med:.2e}')
    ax.axvline(mean, color='r', linestyle='--', label=f'Mean = {mean:.2e}')

    ax.legend()
    ax.grid(True)

    # Set scale and limits
    ax.set_xscale(scale)
    ax.set_ylim(bottom=0)

    fig.tight_layout()

    fig.savefig(f"figs/{name}/adh_distrib.png", dpi=300)
    plt.close(fig)

# ======================================================================================================================
# VELOCITY AND AERODYNAMIC QUANTITIES PLOTS
# ======================================================================================================================

def plot_velocity_history(flow: Flow, scale: str = 'linear', **kwargs) -> None:
    """
    Basic plot of the time history of friction velocity.

    Args:
        flow (Flow): Flow containing the velocity time history.
        scale (str, optional): Scale of the x-axis. Defaults to 'linear'.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(flow.time, flow.velocity, **kwargs)

    ax.set_xscale(scale)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Friction velocity [m/s]')

    fig.tight_layout()

    fig.savefig('figs/velocity.png', dpi=300)
    plt.close(fig)

def plot_flow(flow: Flow, name: str, i: int, scale: str = 'linear', **kwargs) -> None:
    """
    Basic plot of all the time dependant quantities of a flow for the i-th size bin.

    Args:
        flow (Flow): Flow containing the velocity time history.
        name (str): Name of the configuration file. Will be used to save the figure in the right folder.
        i (int): Index of the size bin.
        scale (str, optional): Scale of the x-axis. Defaults to 'linear'.
    """
    # Create subplots
    fig, axs = plt.subplots(2, 2, sharex=True, )
    axs[1, 0].sharey(axs[1, 1])

    # plot all time histories
    axs[0, 0].plot(flow.time, flow.velocity, color='black', **kwargs)
    axs[0, 1].plot(flow.time, flow.burst, color='green', **kwargs)
    axs[1, 0].plot(flow.time, flow.lift[:, i], color='orange', **kwargs)
    axs[1, 1].plot(flow.time, flow.drag[:, i], color='red', **kwargs)

    # Axis labels
    axs[0, 0].set_ylabel('Friction velocity [m/s]')
    axs[0, 1].set_ylabel('Burst frequency [s-1]')
    axs[1, 0].set_ylabel('Lift [N]')
    axs[1, 1].set_ylabel('Drag [N]')

    # Setting shared properties
    for ax in axs.flat:
        # Limits
        ax.set_xlim(left=0, right=flow.time[-1])
        ax.set_ylim(bottom=0)

        ax.set_xscale(scale)

        # Grids
        ax.grid(True)

    fig.tight_layout()

    fig.savefig(f'figs/{name}/all_aero_forces.png', dpi=300)
    plt.close(fig)

# ======================================================================================================================
# POST-PROCESSING PLOTS
# ======================================================================================================================

def plot_resuspended_fraction(results: list[TemporalResults], name: str, scale: str = 'log', ) -> None:
    """
    Basic plot of the resuspended fraction with time. Can take several simulation results.

    Args:
        results (list[TemporalResults]): List of simulation result objects.
        name (str): Name of the configuration file. Will be used to save the figure in the right folder.
        scale (str, optional): Scale of the x-axis. Defaults to 'log'.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    for res in results:
        ax.plot(res.time, res.resuspended_fraction, label=f'{res.name}')

    # Draw resuspension milestones lines
    # fracs = [0.5, 0.9, 0.99]
    # x = [res.time_to_fraction(frac) for frac in fracs]
    # y = [frac * res.final_resus_frac for frac in fracs]
    #
    # ax.axvline(x=x[0], ymax=y[0], color='r', linestyle='-', )
    # ax.axvline(x=x[1], ymax=y[1], color='r', linestyle='--', )
    # ax.axvline(x=x[2], ymax=y[2], color='r', linestyle=':', )

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Resuspended fraction')
    ax.legend()

    ax.set_xscale(scale)
    ax.set_xlim(left=results[0].time[1], )
    ax.set_ylim(0.0, 1.0)

    ax.grid(axis='x', which='both')
    ax.grid(axis='y', which='major')

    fig.tight_layout()

    fig.savefig(f'figs/{name}/resuspended_fraction.png', dpi=300)
    plt.close(fig)

def plot_instant_rate(results: list[TemporalResults], name: str, xscale: str = 'log', yscale: str = 'log', ) -> None:
    """
    Basic plot of the instant resuspension rate with time. Can take several simulation results.

    Args:
        results (list[TemporalResults]): List of simulation result objects.
        name (str): Name of the configuration file. Will be used to save the figure in the right folder.
        xscale (str, optional): Scale of the x-axis. Defaults to 'log'.
        yscale (str, optional): Scale of the y-axis. Defaults to 'log'.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    for res in results:
        ax.plot(res.time[:-1], res.instant_rate, label=f'{res.name}')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Resuspension rate')
    ax.legend()

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(results[0].time[1], results[0].time[-1])
    ax.set_ylim(bottom=1e-10, top=1e0)

    ax.grid(axis='x', which='both')
    ax.grid(axis='y', which='major')

    fig.tight_layout()

    fig.savefig(f'figs/{name}/instant_rate.png', dpi=300)
    plt.close(fig)

# ======================================================================================================================
# OTHER PLOTS
# ======================================================================================================================

def plot_validity_domain(modes: list[float], target_vel: float, viscosity: float,) -> None:
    # Defined the radius and velocity ranges
    velocities = np.linspace(0.1, 1.0, 100)
    radii = np.logspace(np.log10(1.0), np.log10(50.0), 100)

    # Create mesh for contour plotting
    U, R = np.meshgrid(velocities, radii)

    # Compute the r+ grid
    R_plus = (R * 1e-6) * U / viscosity

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(U, R, R_plus, cmap='magma', levels=20)
    cbar = fig.colorbar(contour, ax=ax)

    # Add isolines for r+ = 1.8 and r+ = 2.5
    isolines = ax.contour(U, R, R_plus, levels=[1.8, 2.5], colors=['white', 'red'], linestyles=['dashed', 'solid'])
    ax.clabel(isolines, inline=True, fontsize=8, fmt={1.8: '1.8', 2.5: '2.5'})

    # Add modes
    for mode in modes:
        ax.scatter(target_vel, mode, color='white', marker='x',)

    # Labels and formatting
    ax.set_xlabel("Friction Velocity $u^*$ [m/s]")
    ax.set_ylabel("Particle Radius $r$ [µm]")
    cbar.set_label("$r^+$")

    #ax.set_yscale('log')  # Log scale for r (optional)

    fig.tight_layout()

    fig.savefig('figs/validity_domain.png', dpi=300)
    plt.close(fig)

def plot_fraction_velocity_curve(res: FractionVelocityResults, plot_exp=False, plot_stats=True,) -> None:
    """
    Basic plot of the fraction-velocity curve.

    Args:
        res (FractionVelocityResults): FractionVelocityResults object.
        plot_exp (bool, optional): If True, plot experimental values from Reeks and Hall (2001).
    """
    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(res.velocities, res.fraction, color='r', zorder=10)

    # Plot analysis quantities
    if plot_stats:
        vlow, vhigh = res.threshold_velocity(0.95), res.threshold_velocity(0.05)

        ax.axvline(vhigh, color='k', linestyle='--', linewidth=0.75, zorder=5)
        ax.axvline(vlow, color='k', linestyle='--', linewidth=0.75, zorder=5)
        ax.fill_betweenx(y=[0, 1.1], x1=vlow, x2=vhigh, color='grey', alpha=0.3, zorder=3)

    # Load Reeks and Hall data
    if plot_exp:
        exp_data = read_exp_data()
        ax.scatter(exp_data[10][9][0], exp_data[10][9][1], color='blue', marker='^', facecolors='none',
                   label='Exp. run 9')
        ax.scatter(exp_data[10][10][0], exp_data[10][10][1], color='black', marker='s', facecolors='none',
                   label='Exp. run 10')
        ax.scatter(exp_data[10][15][0], exp_data[10][15][1], color='red', marker='o', facecolors='none',
                   label='Exp. run 15')

    ax.set_xscale('log')
    ax.set_xlim(res.velocities[0], res.velocities[-1])
    ax.set_ylim(0, 1.1)

    ax.set_xlabel('Friction velocity [m/s]')
    ax.set_ylabel('Remaining fraction after 1s')

    ax.grid(axis='x', which='both', zorder=0)
    ax.grid(axis='y', which='major', zorder=0)

    fig.tight_layout()

    fig.savefig('figs/validation.png', dpi=300)
    plt.close(fig)

def plot_fraction_derivative(res: FractionVelocityResults,) -> None:
    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))

    med = 0.93
    spread = 1.72

    lognormfit = np.array([
        log_norm(vel, med, spread) for vel in res.velocities
    ])

    ax.plot(res.velocities, res.fraction_derivative, color='r', zorder=10)
    ax.plot(res.velocities, lognormfit, color='b', linestyle='--', zorder=10)

    #ax.set_xscale('log')

    fig.tight_layout()

    fig.savefig('figs/fraction_derivative.png', dpi=300)
    plt.close(fig)