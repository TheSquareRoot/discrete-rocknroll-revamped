import logging

import numpy as np
from numpy.typing import NDArray

from rnr.core.aeromodel import AeroModel
from rnr.core.distribution import SizeDistribution

logger = logging.getLogger(__name__)


class Flow:
    def __init__(
        self,
        velocity: NDArray,
        lift: NDArray,
        drag: NDArray,
        faero: NDArray,
        fluct_var: NDArray,
        burst: NDArray,
        time: NDArray,
    ) -> None:
        self.velocity = velocity
        self.lift = lift
        self.drag = drag
        self.faero = faero
        self.fluct_var = fluct_var
        self.burst = burst
        self.time = time

    def __str__(self) -> str:
        return (
            f"Flow(\n"
            f"  velocity: {np.shape(self.velocity)}   - [{self.velocity[0]:.2e} ... {self.velocity[-1]:.2e}],\n"
            f"  lift: {np.shape(self.lift)} - [{self.lift[0, 0]:.2e} ... {self.lift[-1, -1]:.2e}],\n"
            f"  drag: {np.shape(self.drag)} - [{self.drag[0, 0]:.2e} ... {self.drag[-1, -1]:.2e}],\n"
            f"  burst: {np.shape(self.burst)} - [{self.burst[0]:.2e} ... {self.burst[-1]:.2e}]\n"
            f")"
        )

    @property
    def nsteps(
        self,
    ) -> int:
        return len(self.time)


class FlowBuilder:
    def __init__(
        self,
        size_distrib: SizeDistribution,
        aeromodel: AeroModel,
        duration: float,
        dt: float,
        target_vel: float,
        acc_time: float,
        transition: str,
        density: float,
        viscosity: float,
        frms: float,
        perturbation: bool,  # noqa: FBT001
        **kwargs: dict,
    ) -> None:
        # Objects
        self.size_distrib = size_distrib
        self.aeromodel = aeromodel

        # Simulation params
        self.duration = duration
        self.dt = dt
        self.target_vel = target_vel
        self.acc_time = acc_time
        self.transition = transition
        self.perturbation = perturbation

        # Physical quantities
        self.density = density
        self.viscosity = viscosity

        # Aerodynamic coeffs
        self.frms = frms

    def generate(
        self,
    ) -> Flow:
        # First generate the time array
        time = np.arange(0.0, self.duration, self.dt)

        # Then compute the velocity as a function of time
        if self.acc_time != 0.0:
            if self.transition == "smooth":
                scaled_time = np.minimum(time / self.acc_time, 1)
                velocity = self.target_vel * (scaled_time**2 * (3 - 2 * scaled_time))
            else:
                velocity = np.clip(
                    (self.target_vel / self.acc_time) * time,
                    0,
                    self.target_vel,
                )
        else:
            velocity = np.ones_like(time) * self.target_vel

        if self.perturbation:
            # Generate gaussian white noise
            # noise = np.random.normal(0.0, 1.0, size=len(time))
            # current_rmse = np.sqrt(np.mean(noise ** 2))
            # scaled_noise = noise * (0.5 * self.frms * np.max(velocity) / current_rmse)
            # velocity += scaled_noise
            rng = np.random.default_rng()

            local_std = 0.5 * self.frms * velocity

            noise = rng.normal(loc=0.0, scale=local_std)

            velocity += noise

        # Compute aerodynamic quantities
        lift = self.aeromodel.lift(velocity, self.size_distrib.radii_meter)
        drag = self.aeromodel.drag(velocity, self.size_distrib.radii_meter)
        faero = 0.5 * lift + 100 * drag
        fluct_var = (self.frms * faero) ** 2
        burst = 0.5 * self.aeromodel.burst(velocity) / np.pi

        # Instantiate the flow class
        flow = Flow(
            velocity,
            lift,
            drag,
            faero,
            fluct_var,
            burst,
            time,
        )

        return flow
