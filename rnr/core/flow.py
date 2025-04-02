import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray

from rnr.utils.config import setup_logging
from rnr.core.distribution import SizeDistribution
from rnr.core.aeromodel import AeroModel


# Configure module logger from utils file
logger = setup_logging(__name__, 'logs/log.log')


class Flow:
    def __init__(self,
                 velocity: NDArray[np.floating],
                 lift: NDArray[np.floating],
                 drag: NDArray[np.floating],
                 faero: NDArray[np.floating],
                 fluct_var: NDArray[np.floating],
                 burst: NDArray[np.floating],
                 time: NDArray[np.floating],
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
            f"  lift: {np.shape(self.lift)} - [{self.lift[0,0]:.2e} ... {self.lift[-1,-1]:.2e}],\n"
            f"  drag: {np.shape(self.drag)} - [{self.drag[0,0]:.2e} ... {self.drag[-1,-1]:.2e}],\n"
            f"  burst: {np.shape(self.burst)} - [{self.burst[0]:.2e} ... {self.burst[-1]:.2e}]\n"
            f")"
        )

    @property
    def nsteps(self,) -> int:
        return len(self.time)


class FlowBuilder:
    def __init__(self,
                 size_distrib: SizeDistribution,
                 aeromodel: AeroModel,
                 duration: float,
                 dt: float,
                 target_vel:  float,
                 acc_time: float,
                 transition: str,
                 density: float,
                 viscosity: float,
                 frms: float,
                 **kwargs,
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

        # Physical quantities
        self.density = density
        self.viscosity = viscosity

        # Aerodynamic coeffs
        self.frms = frms

    def generate(self,) -> Flow:
        # First generate the time array
        time = np.arange(0.0, self.duration, self.dt)

        # Then compute the velocity as a function of time
        if self.acc_time != 0.0:
            if self.transition == 'smooth':
                scaled_time = np.minimum(time / self.acc_time, 1)
                velocity = self.target_vel * (scaled_time**2 * (3 - 2 * scaled_time))
            else:
                velocity = np.clip((self.target_vel / self.acc_time) * time, 0, self.target_vel)
        else:
            velocity = np.ones_like(time) * self.target_vel

        # Compute aerodynamic quantities
        lift = self.aeromodel.lift(velocity, self.size_distrib.radii_meter)
        drag = self.aeromodel.drag(velocity, self.size_distrib.radii_meter)
        faero = 0.5 * lift + 100 * drag
        fluct_var = (self.frms * faero) ** 2
        burst = self.aeromodel.burst(velocity, )

        # Instantiate the flow class
        flow = Flow(velocity,
                    lift,
                    drag,
                    faero,
                    fluct_var,
                    burst,
                    time,
                    )

        return flow
