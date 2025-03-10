import numpy as np

from numpy.typing import NDArray

from .config import setup_logging

# Configure module logger from config file
logger = setup_logging(__name__, 'logs/log.log')


class ForceModel:
    def lift(self,
             velocity: NDArray[np.floating],
             radii: NDArray[np.floating]
             ) -> NDArray[np.floating]:
        pass

    def drag(self,
             velocity: NDArray[np.floating],
             radii: NDArray[np.floating]
             ) -> NDArray[np.floating]:
        pass

    def burst(self,
             velocity: NDArray[np.floating],
             ) -> NDArray[np.floating]:
        pass

class BaseForceModel(ForceModel):
    """
    Estimates lift, drag and the burst frequency for a particle attached to a flat surface in a shear flow.
    All formulas are based on the results of O'Neill (1968) and Hall (1988).
    """
    def __init__(self, density: float, viscosity: float) -> None:
        self.density = density
        self.viscosity = viscosity

    def lift(self,
             velocity: NDArray[np.floating],
             radii: NDArray[np.floating]
             ) -> NDArray[np.floating]:
        # Reshape for broadcasting
        velocity = velocity.reshape(1, -1)
        radii = radii.reshape(-1, 1)

        return 20.9 * self.density * (self.viscosity ** 2) * ((radii * velocity / self.viscosity) ** 2.31)

    def drag(self,
             velocity: NDArray[np.floating],
             radii: NDArray[np.floating]
             ) -> NDArray[np.floating]:
        # Reshape for broadcasting
        velocity = velocity.reshape(1, -1)
        radii = radii.reshape(-1, 1)

        return 32.0 * self.density * (self.viscosity ** 2) * ((radii * velocity / self.viscosity) ** 2)

    def burst(self,
             velocity: NDArray[np.floating],
             ) -> NDArray[np.floating]:
        return 0.00658 * (velocity ** 2) / self.viscosity

