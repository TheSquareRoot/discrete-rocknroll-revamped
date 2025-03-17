import numpy as np
from scipy.special import erf

from rnr.utils.config import setup_logging
from rnr.core.distribution import AdhesionDistribution, SizeDistribution
from rnr.core.flow import Flow


# Configure module logger from utils file
logger = setup_logging(__name__, 'logs/log.log')


def rocknroll_model(size_distrib: SizeDistribution,
                    adh_distrib: AdhesionDistribution,
                    flow: Flow,):

    # Construct the Fadh - Faero array
    adh_tiled = np.tile(adh_distrib.fadh, (flow.nsteps, 1, 1))
    aero_tiled = np.tile(flow.faero, (adh_distrib.nbins, 1, 1)).transpose(1,2,0)
    burst_tiled = np.tile(flow.burst[:, np.newaxis, np.newaxis], (1, size_distrib.nbins, adh_distrib.nbins))

    # plt.clf()
    # plt.matshow(adh_tiled[:,0,:],)
    # plt.colorbar()
    # plt.savefig('figs/test.png', dpi=300)

    diff = adh_tiled - aero_tiled
    fluct_tiled = 0.04 * (aero_tiled ** 2)

    rate = burst_tiled * np.exp(- (diff ** 2) / (2 * fluct_tiled)) / (0.5 * (1 + erf(diff / np.sqrt( 2 * fluct_tiled))))

    return rate