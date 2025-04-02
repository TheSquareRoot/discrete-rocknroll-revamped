import matplotlib.pyplot as plt
import numpy as np
import os

from rnr.utils.config import setup_logging
from rnr.utils.parameters import check_config, load_config

from rnr.core.aeromodel import BaseAeroModel
from rnr.core.distribution import SizeDistributionBuilder, AdhesionDistributionBuilder
from rnr.core.flow import FlowBuilder
from rnr.core.model import RocknRollModel, NonGaussianRocknRollModel
from rnr.simulation.simulation import Simulation
from rnr.postproc.plotting import (plot_adhesion_distribution,
                                   plot_size_distribution,
                                   plot_flow,
                                   plot_resuspended_fraction,
                                   plot_instant_rate,
                                   plot_fraction_velocity_curve,
                                   plot_fraction_derivative,
                                   )
from rnr.postproc.results import TemporalResults, FractionVelocityResults


logger = setup_logging(__name__, 'logs/log.log')

def _build_distribs(size_params, adh_params, name=None, plot=False,):
    # Build the particle size distribution
    logger.info('Generating size distribution...')
    sizedistrib_builder = SizeDistributionBuilder(**size_params)
    size_distrib = sizedistrib_builder.generate()
    logger.debug(f'Size distribution generated: {size_distrib}')

    # Build the adhesion force distribution
    logger.info('Generating adhesion distribution...')
    adhesion_builder = AdhesionDistributionBuilder(size_distrib, **adh_params)
    adh_distrib = adhesion_builder.generate()
    logger.debug(f'Adhesion distribution generated: {adh_distrib}')

    # Plot the distributions if the user requested
    if plot:
        plot_size_distribution(size_distrib, name, scale='linear')
        plot_adhesion_distribution(adh_distrib, name, 0, norm=False, scale='linear')

    return size_distrib, adh_distrib

def _build_flow(size_distrib, flow_params, name=None, plot=False):
    # Instantiate force model
    aeromodel = BaseAeroModel(**flow_params)

    # Build the flow
    logger.info('Generating friction velocity time history...')
    flow_builder = FlowBuilder(size_distrib, aeromodel, **flow_params)
    flow = flow_builder.generate()
    logger.debug(f'Flow generated: {flow}')

    # Plot the time histories if requested
    if plot:
        plot_flow(flow, name,0)

    return flow

def _build_model(model, size_distrib, adh_distrib, name=None, plot=False):
    # Chose which resuspension model to use
    if model == 'RnR':
        return RocknRollModel(size_distrib, adh_distrib)
    elif model == 'NG_RnR':
        return NonGaussianRocknRollModel(size_distrib, adh_distrib)
    else:
        raise NotImplementedError

def single_run(config_file: str,) -> TemporalResults:
    # Load utils file
    config = load_config(f"configs/{config_file}.toml")

    # Check the values from the utils file
    logger.info('Checking parameters...')
    check_config(config)

    # Create the output folder for figures
    name = config['info']['full_name']
    os.makedirs(f'figs/{name}', exist_ok=True)

    # Compose the argument dicts for the builders
    size_params = config['sizedistrib']
    adh_params = {**config['adhdistrib'], **config['physics']}
    flow_params = {**config['simulation'], **config['physics']}

    # Build the distributions, the flow and the resuspension model
    size_distrib, adh_distrib = _build_distribs(size_params, adh_params, name, plot=True)
    flow = _build_flow(size_distrib, flow_params, name, plot=True)
    resusp_model = _build_model(config['physics']['resuspension_model'], size_distrib, adh_distrib, name, plot=True)

    # Build a simulation and run it
    logger.info('Running simulation...')
    sim = Simulation(size_distrib, adh_distrib, flow, resusp_model)
    res = sim.run(config['simulation']['vectorized'])
    res.name = config['info']['short_name']
    logger.info('Done.')

    # Plot basic results
    plot_resuspended_fraction([res], name,)
    plot_instant_rate([res], name)

    return res

def multiple_runs(config_dir: str,) -> None:
    # Get the config files from the directory
    dir_path = f"configs/{config_dir}/"
    config_files = [f for f in os.listdir(dir_path) if f.endswith(".toml")]

    # Run the simulations
    results = []

    for config_file in config_files:
        path = f"{config_dir}/{config_file.split('.')[0]}"
        results.append(single_run(path))

    # Plot the results of all simulations on the same graph
    plot_resuspended_fraction(results, name='multi',)
    plot_instant_rate(results, name='multi',)

def fraction_velocity_curve(config_file: str) -> None:
    # Load utils file
    config = load_config(f"configs/{config_file}.toml")

    # Check the values from the utils file
    logger.info('Checking parameters...')
    check_config(config)

    # Compose the argument dicts for the builders
    size_params = config['sizedistrib']
    adh_params = {**config['adhdistrib'], **config['physics']}

    # Build the distributions
    size_distrib, adh_distrib = _build_distribs(size_params, adh_params, plot=False)
    resusp_model = _build_model(config['physics']['resuspension_model'], size_distrib, adh_distrib, plot=False)

    # Generate a range of velocities to build the validation fraction-velocity curve
    velocities = np.logspace(np.log10(0.05), np.log10(10), 60)
    fraction = np.zeros_like(velocities)
    flow_params = {**config['simulation'], **config['physics']}

    for i in range(velocities.shape[0]):
        # Generate a flow with the given target velocity
        flow_params['target_vel'] = velocities[i]
        flow = _build_flow(size_distrib, flow_params, plot=False)

        # Run the simulation
        logger.info(f'Running simulation {i+1}/{velocities.shape[0]}...')
        sim = Simulation(size_distrib, adh_distrib, flow, resusp_model)
        res = sim.run(config['simulation']['vectorized'])
        logger.info('Done.')

        # Store the final fraction
        fraction[i] = 1 - res.resuspended_fraction[-1]

    # Store results and plot the fraction-velocity curves
    res = FractionVelocityResults(adh_distrib, size_distrib, fraction, velocities,)
    res.name = config['info']['short_name']

    # Plot the curve
    plot_fraction_velocity_curve(res, plot_exp=(config_file == 'reeks'))
    plot_fraction_derivative(res,)
    print(f'Critical threshold velocity (50%): {res.threshold_velocity(0.5):.2f}m/s')
    print(f'Resuspension range: {res.resuspension_range:.2f}m/s')
