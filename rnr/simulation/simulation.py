import numpy as np

from rnr.utils.config import setup_logging
from rnr.core.distribution import AdhesionDistribution, SizeDistribution
from rnr.core.flow import Flow
from rnr.core.model import RocknRollModel
from rnr.postproc.results import Results


# Configure module logger from utils file
logger = setup_logging(__name__, 'logs/log.log')


class Simulation:
    def __init__(self,
                 size_distrib: SizeDistribution,
                 adh_distrib: AdhesionDistribution,
                 flow: Flow,
                 ) -> None:

        self.size_distrib = size_distrib
        self.adh_distrib = adh_distrib
        self.flow = flow

    def run(self, vectorized: bool = False) -> Results:
        # Build population array
        logger.info('Building population array...')
        counts = np.zeros([self.flow.nsteps, self.size_distrib.nbins, self.adh_distrib.nbins])
        counts[0,:,:] = self.size_distrib.weights[:, None] * self.adh_distrib.weights

        # Compute the rates for the whole simulation
        logger.info('Computing rates...')
        resusp_model = RocknRollModel(self.size_distrib, self.adh_distrib, self.flow)
        dt = self.flow.time[1] - self.flow.time[0]

        logger.info('Entering time loop...')

        # In the vectorized case, the resuspension rate array is computed before the loop. It is usually faster but
        # requires more memory
        if vectorized:
            logger.debug('Vectorized simulation...')
            rate = resusp_model.rate_vectorized()

            for t in range(self.flow.nsteps-1):
                counts[t+1,:,:] = np.maximum(counts[t,:,:] * (1 - rate[t,:,:] * dt), 0)

        # In the sequential case, the resuspension rate is computed at each time step
        else:
            logger.debug('Sequential simulation...')
            for t in range(self.flow.nsteps-1):
                rate = resusp_model.rate(t)
                counts[t+1,:,:] = np.maximum(counts[t,:,:] * (1 - rate * dt), 0)

        res = Results(self.adh_distrib,
                      self.size_distrib,
                      self.flow,
                      counts,
                      self.flow.time,)

        return res