import logging

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class AeroModel:
    """
    Aerodynamic model for computing lift, drag, and burst forces on particles in a flow.

    Attributes
    ----------
    density : float
        Fluid density [kg/m³].
    viscosity : float
        Fluid dynamic viscosity [Pa·s].
    drag_model : str
        The drag model to use ("stokes" or "liu").
    drag_coeff : float
        Scaling factor for the drag force.
    lift_coeff : float
        Scaling factor for the lift force.
    burst_coeff : float
        Scaling factor for the burst frequency.
    """

    def __init__(
        self,
        density: float,
        viscosity: float,
        drag_model: str,
        drag_coeff: float,
        lift_coeff: float,
        burst_coeff: float,
        **kwargs: dict,
    ) -> None:
        self.density = density
        self.viscosity = viscosity
        self.drag_model = drag_model
        self.drag_coeff = drag_coeff
        self.lift_coeff = lift_coeff
        self.burst_coeff = burst_coeff

    def lift(
        self,
        velocity: NDArray,
        radii: NDArray,
    ) -> NDArray:
        """
        Compute the lift force acting on particles according to Hall (1988)'s experiments.

        Parameters
        ----------
        velocity : NDArray (size: nv)
            Friction velocity at particle locations.
        radii : NDArray (size: nr)
            Particle radii in meters.

        Returns
        -------
        NDArray (size: nv x nr)
            2D array of lift forces.
        """
        # Reshape for broadcasting
        velocity = velocity.reshape(-1, 1)
        radii = radii.reshape(1, -1)

        return (
            self.lift_coeff * 20.9 * self.density * (self.viscosity**2) * ((radii * velocity / self.viscosity) ** 2.31)
        )

    def drag(
        self,
        velocity: NDArray,
        radii: NDArray,
    ) -> NDArray:
        """
        Compute the drag force acting on particles.
        Models available:
            - stokes: stokes law.
            - oneill: modified Stokes law to account for shear flow near the wall (O'Neill, 1968).
            - liu: oneill corrected with a reynolds dependent coefficient (Liu, 2011).

        The O'Neill model is recommended. There is little noticeable difference between the O'Neill and Liu models in practice.

        Parameters
        ----------
        velocity : NDArray (size: nv)
            Friction velocity at particle locations.
        radii : NDArray (size: nr)
            Particle radii in meters.

        Returns
        -------
        NDArray (size: nv x nr)
            2D array of drag forces.
        """
        # Reshape for broadcasting
        velocity = velocity.reshape(-1, 1)
        radii = radii.reshape(1, -1)
        drag = np.zeros([velocity.shape[0], radii.shape[1]])

        # Translate radii to in wall unit
        rplus = radii * velocity / self.viscosity

        # Compute drag according to the chosen model
        if self.drag_model == "stokes":
            drag = 6 * np.pi * self.density * (self.viscosity**2) * (rplus**2)

        if self.drag_model == "oneill":
            drag = 32.0 * self.density * (self.viscosity**2) * (rplus**2)

        if self.drag_model == "liu":
            # Masks
            mask_low = rplus < np.sqrt(2.5)
            mask_high = ~mask_low

            drag_low = (1 + 0.0961 * (rplus**2)) * 32.0 * self.density * (self.viscosity**2) * (rplus**2)
            drag_high = (1 + 0.158 * (rplus ** (4 / 3))) * 32.0 * self.density * (self.viscosity**2) * (rplus**2)

            drag[mask_low] = drag_low[mask_low]
            drag[mask_high] = drag_high[mask_high]

        return self.drag_coeff * drag

    def burst(
        self,
        velocity: NDArray,
    ) -> NDArray:
        """
        Compute the frequency of burst events according to O'Neill (1968).

        Parameters
        ----------
        velocity : NDArray
            Friction velocity at particle locations.

        Returns
        -------
        NDArray
            Array of burst event frequencies.
        """
        return self.burst_coeff * (velocity**2) / self.viscosity
