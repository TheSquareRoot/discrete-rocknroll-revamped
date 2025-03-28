import numpy as np

from numpy.typing import NDArray

from rnr.utils.config import setup_logging

# Configure module logger from utils file
logger = setup_logging(__name__, 'logs/log.log')


class AeroModel:
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

class BaseAeroModel(AeroModel):
    """
    Estimates lift, drag and the burst frequency for a particle attached to a flat surface in a shear flow.
    All formulas are based on the results of O'Neill (1968) and Hall (1988).
    """
    def __init__(self,
                 density: float,
                 viscosity: float,
                 drag_coeff: float,
                 drag_power: float,
                 lift_coeff: float,
                 lift_power: float,
                 burst_coeff: float,
                 **kwargs) -> None:
        self.density = density
        self.viscosity = viscosity
        self.drag_coeff = drag_coeff
        self.drag_power = drag_power
        self.lift_coeff = lift_coeff
        self.lift_power = lift_power
        self.burst_coeff = burst_coeff

    def lift(self,
             velocity: NDArray[np.floating],
             radii: NDArray[np.floating]
             ) -> NDArray[np.floating]:
        # Reshape for broadcasting
        velocity = velocity.reshape(-1, 1)
        radii = radii.reshape(1, -1)

        return self.lift_coeff * self.density * (self.viscosity ** 2) * ((radii * velocity / self.viscosity) ** self.lift_power)

    def drag(self,
             velocity: NDArray[np.floating],
             radii: NDArray[np.floating]
             ) -> NDArray[np.floating]:
        # Reshape for broadcasting
        velocity = velocity.reshape(-1, 1)
        radii = radii.reshape(1, -1)

        return self.drag_coeff * self.density * (self.viscosity ** 2) * ((radii * velocity / self.viscosity) ** self.drag_power)

    def burst(self,
             velocity: NDArray[np.floating],
             ) -> NDArray[np.floating]:
        return self.burst_coeff * (velocity ** 2) / self.viscosity

