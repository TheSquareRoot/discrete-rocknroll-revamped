import matplotlib.pyplot as plt
import numpy as np
import os

from rnr.utils.config import setup_logging
from rnr.utils.parameters import check_config, load_config

from rnr.core.aeromodel import BaseAeroModel
from rnr.core.distribution import SizeDistributionBuilder, AdhesionDistributionBuilder
from rnr.core.flow import FlowBuilder
from rnr.simulation.simulation import Simulation
from rnr.postproc.plotting import (plot_adhesion_distribution,
                                   plot_size_distribution,
                                   plot_flow,
                                   )
from rnr.postproc.results import Results


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

def single_run(config_file: str,) -> Results:
    # Load utils file
    config = load_config(f"configs/{config_file}.toml")

    # Check the values from the utils file
    logger.info('Checking parameters...')
    check_config(config)

    # Create the output folder for figures
    name = config['info']['name']
    os.makedirs(f'figs/{name}', exist_ok=True)

    # Compose the argument dicts for the builders
    size_params = config['sizedistrib']
    adh_params = {**config['adhdistrib'], **config['physics']}
    flow_params = {**config['simulation'], **config['physics']}

    # Build the distributions and the flow
    size_distrib, adh_distrib = _build_distribs(size_params, adh_params, name, plot=True)
    flow = _build_flow(size_distrib, flow_params, name, plot=True)

    # Build a simulation and run it
    logger.info('Running simulation...')
    sim = Simulation(size_distrib, adh_distrib, flow)
    res = sim.run(config['simulation']['vectorized'])
    logger.info('Done.')

    # Plot basic results
    res.plot_resuspended_fraction()
    res.plot_instant_rate()

    return res

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

    # Generate a range of velocities to build the validation fraction-velocity curve
    target_velocities = np.logspace(np.log10(0.1), np.log10(10), 40)
    fraction = np.zeros_like(target_velocities)
    flow_params = {**config['simulation'], **config['physics']}

    for i in range(target_velocities.shape[0]):
        # Generate a flow with the given target velocity
        flow_params['target_vel'] = target_velocities[i]
        flow = _build_flow(size_distrib, flow_params, plot=False)

        # Run the simulation
        logger.info(f'Running simulation {i+1}/{target_velocities.shape[0]}...')
        sim = Simulation(size_distrib, adh_distrib, flow)
        res = sim.run()
        logger.info('Done.')

        # Store the final fraction
        fraction[i] = 1 - res.resuspended_fraction[-1]

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(target_velocities, fraction)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.1)

    ax.set_xlabel('Friction velocity [m/s]')
    ax.set_ylabel('Remaining fraction after 1s')

    ax.grid(axis='x', which='both')
    ax.grid(axis='y', which='major')

    fig.tight_layout()

    fig.savefig('figs/validation.png', dpi=300)
    plt.close(fig)
