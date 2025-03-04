import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray

from .builder import Builder
from .config import setup_logging
from .distribution import AdhesionDistribution, SizeDistribution
from .flow import Flow


# Configure module logger from config file
logger = setup_logging(__name__, 'logs/log.log')


class Simulation:
    def __init__(self,
                 size_distrib: SizeDistribution,
                 adhesion_distrib: AdhesionDistribution,
                 flow: Flow,):
        self.size_distrib = size_distrib
        self.adhesion_distrib = adhesion_distrib
        self.flow = flow

    def run(self,):
        pass

