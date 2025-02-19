import numpy as np

from numpy.typing import NDArray
from scipy.integrate import quad

from .utils import biasi_params, log_norm


class Distribution:
    def __init__(self,
                 counts: NDArray[np.int_],
                 centers: NDArray[np.float64],
                 edges: NDArray[np.float64],
                 widths: NDArray[np.float64],
                 ) -> None:
        self.counts = counts
        self.centers = centers
        self.edges = edges
        self.widths = widths


class DistributionBuilder:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            if isinstance(value, dict):
                # Recursively convert dicts into DistributionBuilder instances
                setattr(self, key, DistributionBuilder(**value))
            else:
                setattr(self, key, value)

    def generate(self,) -> Distribution:
        """
        Generate an adhesion force distribution from the parameters loaded from the config file.
        A log-normal distribution is assumed.
        """
        # Get the number of distributions
        ndistribs = len(self.distribution.radii)

        # Get the log-normal median and spread parameters
        medians, spreads = [], []
        if self.distribution.params == 'biasi':
            medians, spreads = biasi_params(*self.distribution.radii)
        elif self.distribution.params == 'custom':
            medians = self.distribution.adh_medians
            spreads = self.distribution.adh_spreads

        # Create bins edges, widths and centers
        edges = np.empty([ndistribs, self.distribution.nbins + 1])

        for i, radius in enumerate(self.distribution.radii):
            edges[i] = np.linspace(0.0, medians[i]*10, self.distribution.nbins + 1)

        widths = edges[:,1:] - edges[:,:-1]
        centers =  (edges[:,1:] + edges[:,:-1])/2

        # Compute the bin probabilities
        counts = np.zeros_like(centers)

        for i in range(ndistribs):
            for j in range(self.distribution.nbins):
                counts[i,j] = quad(log_norm, edges[i,j], edges[i,j+1], args=(medians[i], spreads[i],))[0]

            counts[i,:] = counts[i,:] * self.distribution.nparts / np.sum(counts[i,:])

        counts = np.round(counts).astype(int)

        # Instantiate the distribution
        distrib = Distribution(counts, centers, edges, widths)

        return distrib



