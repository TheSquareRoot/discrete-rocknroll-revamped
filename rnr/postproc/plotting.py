import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

from rnr.core.distribution import AdhesionDistribution, SizeDistribution
from rnr.core.flow import Flow
from rnr.core.model import ResuspensionModel
from rnr.postproc.results import FractionVelocityResults, TemporalResults
from rnr.utils.misc import read_exp_data

# ======================================================================================================================
# DISTRIBUTION PLOTS
# ======================================================================================================================


def plot_size_distribution(size_distrib: SizeDistribution, name: str, scale: str = "linear", **kwargs: dict) -> None:
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

    ax.set_xlabel("radius [µm]")
    ax.set_ylabel("weight")

    ax.grid(True)

    fig.tight_layout()

    fig.savefig(f"figs/{name}/size_distrib.png", dpi=300)
    plt.close(fig)


def plot_adhesion_distribution(
    adh_distrib: AdhesionDistribution,
    name: str,
    i: int,
    norm: bool = True,
    scale: str = "log",
    **kwargs: dict,
) -> None:
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
        ax.set_xlabel("Normalized adhesion force")
    else:
        ax.plot(adh_distrib.fadh[i], adh_distrib.weights[i], **kwargs)
        ax.set_xlabel("Adhesion force [N]")

    # Compute the median and display it
    med = adh_distrib.median(i, norm=norm)
    mean = adh_distrib.mean(i, norm=norm)

    ax.axvline(med, color="r", linestyle="-", label=f"Median = {med:.2e}")
    ax.axvline(mean, color="r", linestyle="--", label=f"Mean = {mean:.2e}")

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


def plot_velocity_history(flow: Flow, scale: str = "linear", **kwargs: dict) -> None:
    """
    Basic plot of the time history of friction velocity.

    Args:
        flow (Flow): Flow containing the velocity time history.
        scale (str, optional): Scale of the x-axis. Defaults to 'linear'.
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(flow.time, flow.velocity, **kwargs)

    ax.set_xscale(scale)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Friction velocity [m/s]")

    fig.tight_layout()

    fig.savefig("figs/velocity.png", dpi=300)
    plt.close(fig)


def plot_flow(
    flow: Flow,
    name: str,
    i: int,
    scale: str = "linear",
    **kwargs: dict,
) -> None:
    """
    Basic plot of all the time dependant quantities of a flow for the i-th size bin.

    Args:
        flow (Flow): Flow containing the velocity time history.
        name (str): Name of the configuration file. Will be used to save the figure in the right folder.
        i (int): Index of the size bin.
        scale (str, optional): Scale of the x-axis. Defaults to 'linear'.
    """
    # Create subplots
    fig, axs = plt.subplots(
        2,
        2,
        sharex=True,
    )
    axs[1, 0].sharey(axs[1, 1])

    # plot all time histories
    axs[0, 0].plot(flow.time, flow.velocity, color="black", **kwargs)
    axs[0, 1].plot(flow.time, flow.burst, color="green", **kwargs)
    axs[1, 0].plot(flow.time, flow.lift[:, i], color="orange", **kwargs)
    axs[1, 1].plot(flow.time, flow.drag[:, i], color="red", **kwargs)

    # Axis labels
    axs[0, 0].set_ylabel("Friction velocity [m/s]")
    axs[0, 1].set_ylabel("Burst frequency [s-1]")
    axs[1, 0].set_ylabel("Lift [N]")
    axs[1, 1].set_ylabel("Drag [N]")

    # Setting shared properties
    for ax in axs.flat:
        # Limits
        ax.set_xlim(left=0, right=flow.time[-1])
        ax.set_ylim(bottom=0)

        ax.set_xscale(scale)

        # Grids
        ax.grid(True)

    fig.tight_layout()

    fig.savefig(f"figs/{name}/all_aero_forces.png", dpi=300)
    plt.close(fig)


# ======================================================================================================================
# POST-PROCESSING PLOTS
# ======================================================================================================================


def plot_resuspended_fraction(
    results: list[TemporalResults],
    name: str,
    scale: str = "log",
) -> None:
    """
    Basic plot of the resuspended fraction with time. Can take several simulation results.

    Args:
        results (list[TemporalResults]): List of simulation result objects.
        name (str): Name of the configuration file. Will be used to save the figure in the right folder.
        scale (str, optional): Scale of the x-axis. Defaults to 'log'.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    for res in results:
        ax.plot(res.time, res.resuspended_fraction, label=f"{res.name}")

    # Draw resuspension milestones lines
    # fracs = [0.5, 0.9, 0.99]
    # x = [res.time_to_fraction(frac) for frac in fracs]
    # y = [frac * res.final_resus_frac for frac in fracs]
    #
    # ax.axvline(x=x[0], ymax=y[0], color='r', linestyle='-', )
    # ax.axvline(x=x[1], ymax=y[1], color='r', linestyle='--', )
    # ax.axvline(x=x[2], ymax=y[2], color='r', linestyle=':', )

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Resuspended fraction")
    ax.legend()

    ax.set_xscale(scale)
    ax.set_xlim(
        left=results[0].time[1],
    )
    ax.set_ylim(0.0, 1.0)

    ax.grid(axis="x", which="both")
    ax.grid(axis="y", which="major")

    fig.tight_layout()

    fig.savefig(f"figs/{name}/resuspended_fraction.png", dpi=300)
    plt.close(fig)


def plot_instant_rate(
    results: list[TemporalResults],
    name: str,
    xscale: str = "log",
    yscale: str = "log",
) -> None:
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
        ax.plot(res.time[:-1], res.instant_rate, label=f"{res.name}")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Resuspension rate")
    ax.legend()

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlim(results[0].time[1], results[0].time[-1])
    ax.set_ylim(bottom=1e-10, top=1e0)

    ax.grid(axis="x", which="both")
    ax.grid(axis="y", which="major")

    fig.tight_layout()

    fig.savefig(f"figs/{name}/instant_rate.png", dpi=300)
    plt.close(fig)


# ======================================================================================================================
# OTHER PLOTS
# ======================================================================================================================


def plot_validity_domain(
    modes: list[float],
    target_vel: float,
    viscosity: float,
) -> None:
    # Defined the radius and velocity ranges
    velocities = np.linspace(0.1, 1.0, 100)
    radii = np.logspace(np.log10(1.0), np.log10(50.0), 100)

    # Create mesh for contour plotting
    U, R = np.meshgrid(velocities, radii)

    # Compute the r+ grid
    R_plus = (R * 1e-6) * U / viscosity

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 5))
    contour = ax.contourf(U, R, R_plus, cmap="magma", levels=20)
    cbar = fig.colorbar(contour, ax=ax)

    # Add isolines for r+ = 1.8 and r+ = 2.5
    isolines = ax.contour(
        U,
        R,
        R_plus,
        levels=[1.8, 2.5],
        colors=["white", "red"],
        linestyles=["dashed", "solid"],
    )
    ax.clabel(isolines, inline=True, fontsize=8, fmt={1.8: "1.8", 2.5: "2.5"})

    # Add modes
    for mode in modes:
        ax.scatter(
            target_vel,
            mode,
            color="white",
            marker="x",
        )

    # Labels and formatting
    ax.set_xlabel("Friction Velocity $u^*$ [m/s]")
    ax.set_ylabel("Particle Radius $r$ [µm]")
    cbar.set_label("$r^+$")

    # ax.set_yscale('log')  # Log scale for r (optional)

    fig.tight_layout()

    fig.savefig("figs/validity_domain.png", dpi=300)
    plt.close(fig)


def plot_fraction_velocity_curve(
    results: list[FractionVelocityResults],
    *,
    plot_exp: bool = False,
    plot_stats: bool = True,
) -> None:
    """
    Basic plot of the fraction-velocity curve.

    Args:
        res (FractionVelocityResults): FractionVelocityResults object.
        plot_exp (bool, optional): If True, plot experimental values from Reeks and Hall (2001).
    """
    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))

    for res in results:
        ax.plot(
            res.velocities,
            res.fraction,
            color="r",
            label=res.name,
            zorder=10,
        )

    # Plot analysis quantities
    # Only if there is one set of results for now
    if plot_stats and (len(results) == 1):
        vlow, vhigh = (
            results[0].threshold_velocity(0.95),
            results[0].threshold_velocity(0.05),
        )

        ax.axvline(vhigh, color="k", linestyle="--", linewidth=0.75, zorder=5)
        ax.axvline(vlow, color="k", linestyle="--", linewidth=0.75, zorder=5)
        ax.fill_betweenx(
            y=[0, 1.1],
            x1=vlow,
            x2=vhigh,
            color="grey",
            alpha=0.3,
            zorder=3,
        )

    # Load Reeks and Hall data
    if plot_exp:
        # Load data
        exp_data = read_exp_data()

        # Extract a single experimental run for RMSE comparison
        velocities_exp = [exp_data[10][i][0] for i in [9, 10, 15]]
        fractions_exp = [exp_data[10][i][1] for i in [9, 10, 15]]

        # Interpolate simulation results at experimental velocities
        fractions_sim_interp = [np.interp(exp, results[0].velocities, results[0].fraction) for exp in velocities_exp]

        # Compute RMSE
        rmse = [np.sqrt(np.mean((fractions_sim_interp[i] - fractions_exp[i]) ** 2)) for i in range(3)]
        print("------ RMSE ------")
        print(f"Exp  9: {rmse[0]:.4f}")
        print(f"Exp 10: {rmse[1]:.4f}")
        print(f"Exp 15: {rmse[2]:.4f}")
        print(f"TOTAL: {np.sqrt((rmse[0] ** 2 + rmse[1] ** 2 + rmse[2] ** 2) / 3):.4f}")

        # Plot data
        ax.scatter(
            exp_data[10][9][0],
            exp_data[10][9][1],
            color="blue",
            marker="^",
            facecolors="none",
            label="Exp. run 9",
            zorder=20,
        )
        ax.scatter(
            exp_data[10][10][0],
            exp_data[10][10][1],
            color="black",
            marker="s",
            facecolors="none",
            label="Exp. run 10",
            zorder=20,
        )
        ax.scatter(
            exp_data[10][15][0],
            exp_data[10][15][1],
            color="green",
            marker="o",
            facecolors="none",
            label="Exp. run 15",
            zorder=20,
        )

    ax.set_xscale("log")
    ax.set_xlim(results[0].velocities[0], results[0].velocities[-1])
    ax.set_ylim(0, 1.1)

    ax.set_xlabel("Friction velocity [m/s]")
    ax.set_ylabel("Remaining fraction after 1s")

    ax.legend()

    ax.grid(axis="x", which="both", zorder=0)
    ax.grid(axis="y", which="major", zorder=0)

    fig.tight_layout()

    fig.savefig("figs/validation.png", dpi=300)
    plt.close(fig)


def plot_fraction_velocity_difference(results: list[FractionVelocityResults]) -> None:
    # Compute the difference between the each curve and the baseline (by default the last result)
    diffs = [np.abs(res.fraction - results[-1].fraction) for res in results]

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(6, 4))

    for diff, res in zip(diffs[:-1], results[:-1], strict=True):
        ax.plot(results[0].velocities, diff, label=res.name, zorder=10)

    ax.set_xscale("log")
    ax.set_xlim(results[0].velocities[0], results[0].velocities[-1])
    ax.set_ylim(0, 0.2)

    ax.set_xlabel("Friction velocity [m/s]")
    ax.set_ylabel("Remaining fraction after 1s")

    ax.legend()

    ax.grid(axis="x", which="both", zorder=0)
    ax.grid(axis="y", which="major", zorder=0)

    fig.tight_layout()

    fig.savefig("figs/diff.png", dpi=300)
    plt.close(fig)


def plot_resuspension_rate(
    model: ResuspensionModel,
    flow: Flow,
) -> None:
    # Get the resuspension at the last time step
    rate = model.rate(flow, -1)

    # Plot
    fig = plt.figure(figsize=(6, 4))
    ax1 = fig.add_subplot(1, 1, 1)
    ax2 = ax1.twinx()

    # Force ax1 above ax2
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)  # Hide ax1 background so bars from ax2 are visible

    ax1.plot(
        model.adh_distrib.fadh[0, :] / 1e-9,
        rate[0, :] / flow.burst[-1],
        color="k",
        label="Resuspension rate",
        zorder=10,
    )
    ax2.bar(
        model.adh_distrib.fadh[0, :] / 1e-9,
        model.adh_distrib.weights[0, :],
        width=2.15,  # Adjust width as needed for visual clarity
        color="r",
        alpha=0.5,
        edgecolor="r",
        label="Adhesion distribution",
        zorder=5,
    )

    # Axis limits
    ax1.set_xlim(left=0, right=1e2)
    ax1.set_ylim(bottom=0, top=1.05)
    ax2.set_ylim(bottom=0)

    # Labels
    ax1.set_xlabel(r"Adhesion force $f_a$ [nN]")
    ax1.set_ylabel(r"$\mathcal{T}_r / n_{\theta}$")
    ax2.set_ylabel(r"frequency")

    # Ticks settings
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax1.xaxis.set_minor_locator(ticker.MultipleLocator(5))

    ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    ax2.yaxis.set_major_locator(ticker.MultipleLocator(0.01))
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.0025))

    # Legend
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    fig.tight_layout()

    fig.savefig("figs/resusp_rate.png", dpi=300)
    plt.close(fig)
