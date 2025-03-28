import matplotlib.pyplot as plt
import numpy as np

from ..core.distribution import AdhesionDistribution, SizeDistribution
from ..core.flow import Flow
from ..postproc.results import Results

# ======================================================================================================================
# DISTRIBUTION PLOTS
# ======================================================================================================================

def plot_size_distribution(size_distrib: SizeDistribution, name: str, scale: str = 'linear',**kwargs) -> None:
    """Basic bar plot of the size distribution."""
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
    """Basic plot of the adhesion distribution of the i-th size bin."""
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
    """Basic plot of the time history of friction velocity."""
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(flow.time, flow.velocity, **kwargs)

    ax.set_xscale(scale)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Friction velocity [m/s]')

    fig.tight_layout()

    fig.savefig('figs/velocity.png', dpi=300)
    plt.close(fig)

def plot_flow(flow: Flow, name: str, i: int, scale: str = 'linear', **kwargs) -> None:
    """Basic plot of all the time dependant quantities of a flow for the i-th size bin."""
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

        # Grids
        ax.grid(True)

    fig.tight_layout()

    fig.savefig(f'figs/{name}/all_aero_forces.png', dpi=300)
    plt.close(fig)

# ======================================================================================================================
# POST-PROCESSING PLOTS
# ======================================================================================================================

def plot_resuspended_fraction(results: list[Results], name: str, scale: str = 'log',) -> None:
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

def plot_instant_rate(results: list[Results], name: str, scale: str = 'log',) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))

    for res in results:
        ax.plot(res.time[:-1], res.instant_rate, label=f'{res.name}')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Resuspension rate')
    ax.legend()

    ax.set_xscale(scale)
    ax.set_yscale('log')
    ax.set_xlim(results[0].time[1], results[0].time[-1])
    ax.set_ylim(bottom=1e-10, top=1e0)

    ax.grid(axis='x', which='both')
    ax.grid(axis='y', which='major')

    fig.tight_layout()

    fig.savefig(f'figs/{name}/instant_rate.png', dpi=300)
    plt.close(fig)