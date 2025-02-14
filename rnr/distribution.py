import numpy as np

from numpy.typing import NDArray


class Distribution:
    def __init__(self,
                 count: NDArray[np.int_],
                 center: NDArray[np.float64],
                 edge: NDArray[np.float64],
                 width: NDArray[np.float64],
                 ) -> None:
        self.count = count
        self.center = center
        self.edge = edge
        self.width = width


class DistributionBuilder:
    def __init__(self, radius: float, spread):
        self.radius = radius