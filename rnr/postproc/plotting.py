import matplotlib.pyplot as plt
import numpy as np

from ..core.distribution import AdhesionDistribution, SizeDistribution
from ..core.flow import Flow


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