import numpy as np

from rnr.utils.config import setup_logging, setup_progress_bar
from rnr.core.distribution import AdhesionDistribution, SizeDistribution
from rnr.core.flow import Flow
from rnr.core.model import RocknRollModel
from rnr.postproc.results import TemporalResults


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

    def run(self, vectorized: bool = False) -> TemporalResults:
        # Build population array
        logger.info('Building population array...')
        counts = np.zeros([self.flow.nsteps, self.size_distrib.nbins, self.adh_distrib.nbins])
        counts[0,:,:] = self.size_distrib.weights[:, None] * self.adh_distrib.weights

        # Instantiate the resuspension model
        resusp_model = RocknRollModel(self.size_distrib, self.adh_distrib, self.flow)

        dt = self.flow.time[1] - self.flow.time[0]

        logger.info('Entering time loop...')

        # Create the progress bar
        progress = setup_progress_bar()

        with progress:
            sim_task = progress.add_task('Running simulation...', total=self.flow.nsteps)
            # In the vectorized case, the resuspension rate array is computed before the loop. It is usually faster but
            # requires more memory
            if vectorized:
                logger.debug('Vectorized simulation chosen.')
                rate = resusp_model.rate_vectorized()

                for t in range(self.flow.nsteps-1):
                    counts[t+1,:,:] = np.maximum(counts[t,:,:] * (1 - rate[t,:,:] * dt), 0)

                    progress.advance(sim_task)

            # In the sequential case, the resuspension rate is computed at each time step
            else:
                logger.debug('Sequential simulation chosen.')
                for t in range(self.flow.nsteps-1):
                    rate = resusp_model.rate(t)
                    counts[t+1,:,:] = np.maximum(counts[t,:,:] * (1 - rate * dt), 0)

                    progress.advance(sim_task)


            res = TemporalResults(self.adh_distrib,
                                  self.size_distrib,
                                  self.flow,
                                  counts,
                                  self.flow.time, )

        return res