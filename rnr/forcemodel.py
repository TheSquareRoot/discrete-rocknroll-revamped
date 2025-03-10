import numpy as np

from numpy.typing import NDArray

from .config import setup_logging

# Configure module logger from config file
logger = setup_logging(__name__, 'logs/log.log')


class ForceModel:
    def __init__(self,):
        pass

    def lift(self,
             velocity: NDArray[np.float64],
             radii: NDArray[np.floating]
             ) -> NDArray[np.floating]:
        pass

    def drag(self,
             velocity: NDArray[np.float64],
             radii: NDArray[np.floating]
             ) -> NDArray[np.floating]:
        pass

    def burst(self,
             velocity: NDArray[np.float64],
             radii: NDArray[np.floating]
             ) -> NDArray[np.floating]:
        pass

class BaseForceModel(ForceModel):
    def __init__(self, density: float, viscosity: float) -> None:
        super().__init__()
        self.density = density
        self.viscosity = viscosity

    def lift(self,
             velocity: NDArray[np.float64],
             radii: NDArray[np.floating]
             ) -> NDArray[np.floating]:
        pass