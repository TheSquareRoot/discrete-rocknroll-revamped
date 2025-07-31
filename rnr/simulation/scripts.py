import logging
from pathlib import Path

import numpy as np

from rnr.core.aeromodel import AeroModel
from rnr.core.distribution import (
    AdhesionDistribution,
    AdhesionDistributionBuilder,
    SizeDistribution,
    SizeDistributionBuilder,
)
from rnr.core.flow import Flow, FlowBuilder
from rnr.core.model import (
    NonGaussianRocknRollModel,
    ResuspensionModel,
    RocknRollModel,
    StaticMomentBalance,
)
from rnr.postproc.plotting import (
    plot_adhesion_distribution,
    plot_flow,
    plot_fraction_velocity_curve,
    plot_fraction_velocity_difference,
    plot_instant_rate,
    plot_resuspended_fraction,
    plot_resuspension_rate,
    plot_size_distribution,
)
from rnr.postproc.results import FractionVelocityResults, TemporalResults
from rnr.simulation.simulation import Simulation
from rnr.utils.parameters import check_config, load_config

logger = logging.getLogger(__name__)


def _build_distribs(
    size_params: dict,
    adh_params: dict,
    name: str | None = None,
    *,
    plot: bool = False,
) -> tuple[SizeDistribution, AdhesionDistribution]:
    # Build the particle size distribution
    logger.info("Generating size distribution...")
    sizedistrib_builder = SizeDistributionBuilder(**size_params)
    size_distrib = sizedistrib_builder.generate()
    logger.debug(f"Size distribution generated: {size_distrib}")

    # Build the adhesion force distribution
    logger.info("Generating adhesion distribution...")
    adhesion_builder = AdhesionDistributionBuilder(size_distrib, **adh_params)
    adh_distrib = adhesion_builder.generate()
    logger.debug(f"Adhesion distribution generated: {adh_distrib}")

    # Plot the distributions if the user requested
    if plot:
        plot_size_distribution(size_distrib, name, scale="linear")
        plot_adhesion_distribution(adh_distrib, name, 0, norm=False, scale="linear")

    return size_distrib, adh_distrib


def _build_flow(
    size_distrib: SizeDistribution,
    flow_params: dict,
    name: str | None = None,
    *,
    plot: bool = False,
) -> Flow:
    # Instantiate force model
    aeromodel = AeroModel(**flow_params)

    # Build the flow
    logger.info("Generating friction velocity time history...")
    flow_builder = FlowBuilder(size_distrib, aeromodel, **flow_params)
    flow = flow_builder.generate()
    logger.debug(f"Flow generated: {flow}")

    # Plot the time histories if requested
    if plot:
        plot_flow(flow, name, 0)

    return flow


def _build_model(
    model: str,
    size_distrib: SizeDistribution,
    adh_distrib: AdhesionDistribution,
) -> ResuspensionModel:
    # Chose which resuspension model to use
    if model == "RnR":
        return RocknRollModel(size_distrib, adh_distrib)
    if model == "Static":
        return StaticMomentBalance(size_distrib, adh_distrib)
    if model == "NG_RnR":
        return NonGaussianRocknRollModel(size_distrib, adh_distrib)
    raise NotImplementedError


def single_run(config_file: str | Path) -> TemporalResults:
    # Load utils file
    config_path = Path("configs") / f"{config_file}.toml"
    config = load_config(config_path)

    # Check the values from the utils file
    logger.info("Checking parameters...")
    check_config(config)

    # Create the output folder for figures
    name = config["info"]["full_name"]
    Path(f"figs/{name}").mkdir(parents=True, exist_ok=True)

    # Compose the argument dicts for the builders
    size_params = config["sizedistrib"]
    adh_params = {**config["adhdistrib"], **config["physics"]}
    flow_params = {**config["simulation"], **config["physics"]}

    # Build the distributions, the flow and the resuspension model
    size_distrib, adh_distrib = _build_distribs(size_params, adh_params, name, plot=True)
    flow = _build_flow(size_distrib, flow_params, name, plot=True)
    resusp_model = _build_model(
        config["physics"]["resuspension_model"],
        size_distrib,
        adh_distrib,
    )
    plot_resuspension_rate(resusp_model, flow)

    # Build a simulation and run it
    logger.info("Running simulation...")
    sim = Simulation(size_distrib, adh_distrib, flow, resusp_model)
    res = sim.run(vectorized=config["simulation"]["vectorized"])
    res.name = config["info"]["short_name"]
    logger.info("Done.")

    # Plot basic results
    plot_resuspended_fraction(
        [res],
        name,
    )
    plot_instant_rate([res], name)

    return res


def multiple_runs(config_dir: str | Path) -> None:
    # Get the config files from the directory
    config_dir = Path("configs") / config_dir
    config_files = [f for f in config_dir.iterdir() if f.suffix == ".toml"]

    # Run the simulations
    results = []

    for config_file in config_files:
        relative_path = config_file.relative_to("configs").with_suffix("")
        results.append(single_run(relative_path))

    # Plot the results of all simulations on the same graph
    plot_resuspended_fraction(
        results,
        name="multi",
    )
    plot_instant_rate(
        results,
        name="multi",
    )


def fraction_velocity_curve(config_file: str | Path, *, plot: bool = True) -> FractionVelocityResults:
    # Load utils file
    config_path = Path("configs") / f"{config_file}.toml"
    config = load_config(config_path)

    # Check the values from the utils file
    logger.info("Checking parameters...")
    check_config(config)

    # The number of timesteps does not matter since we assume constant velocity
    if config["simulation"]["perturbation"]:
        config["simulation"]["duration"] = 10.0
        config["simulation"]["dt"] = 1e-3
        config["simulation"]["vectorized"] = True
    else:
        config["simulation"]["duration"] = 1.0
        config["simulation"]["dt"] = 0.5
        config["simulation"]["vectorized"] = True

    # Compose the argument dicts for the builders
    size_params = config["sizedistrib"]
    adh_params = {**config["adhdistrib"], **config["physics"]}

    # Build the distributions
    size_distrib, adh_distrib = _build_distribs(size_params, adh_params, plot=False)
    resusp_model = _build_model(
        config["physics"]["resuspension_model"],
        size_distrib,
        adh_distrib,
    )

    # Generate a range of velocities to build the validation fraction-velocity curve
    velocities = np.logspace(np.log10(0.05), np.log10(10), 60)
    fraction = np.zeros_like(velocities)
    flow_params = {**config["simulation"], **config["physics"]}

    for i in range(velocities.shape[0]):
        # Generate a flow with the given target velocity
        flow_params["target_vel"] = velocities[i]
        flow = _build_flow(size_distrib, flow_params, plot=False)

        # Run the simulation
        logger.info(f"Running simulation {i + 1}/{velocities.shape[0]}...")
        sim = Simulation(size_distrib, adh_distrib, flow, resusp_model)
        res = sim.run(vectorized=config["simulation"]["vectorized"])
        logger.info("Done.")

        # Store the final fraction
        fraction[i] = 1 - res.resuspended_fraction[-1]

    # Store results and plot the fraction-velocity curves
    res = FractionVelocityResults(
        adh_distrib,
        size_distrib,
        fraction,
        velocities,
    )
    res.name = config["info"]["short_name"]

    # Plot the curve
    if plot:
        plot_fraction_velocity_curve([res], plot_exp=(config_file == "reeks"))
        # plot_fraction_derivative(res,)
        print(f"Critical threshold velocity (50%): {res.threshold_velocity(0.5):.2f}m/s")
        print(f"Resuspension range: {res.resuspension_range:.2f}m/s")

    return res


def multiple_fraction_velocity_curves(config_dir: str) -> None:
    # Get the config files from the directory
    config_dir = Path("configs") / config_dir
    config_files = [f for f in config_dir.iterdir() if f.suffix == ".toml"]

    # Run the simulations
    results = []

    for config_file in config_files:
        relative_path = config_file.relative_to("configs").with_suffix("")
        results.append(fraction_velocity_curve(relative_path, plot=False))

    # Plot the results of all simulations on the same graph
    plot_fraction_velocity_curve(results, plot_exp=False, plot_stats=False)
    plot_fraction_velocity_difference(results)
