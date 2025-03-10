import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray

from .config import setup_logging
from .distribution import SizeDistribution
from .forcemodel import ForceModel


# Configure module logger from config file
logger = setup_logging(__name__, 'logs/log.log')


class Flow:
    def __init__(self,
                 velocity: NDArray[np.floating],
                 lift: NDArray[np.floating],
                 drag: NDArray[np.floating],
                 burst: NDArray[np.floating],
                 time: NDArray[np.floating],
                 ) -> None:

        self.velocity = velocity
        self.lift = lift
        self.drag = drag
        self.burst = burst
        self.time = time

    @property
    def nsteps(self,) -> int:
        return len(self.time)

    def plot(self, scale: str = 'linear', **kwargs) -> None:
        plt.clf()
        plt.plot(self.time, self.velocity, **kwargs)

        plt.xscale(scale)

        plt.xlabel('Time [s]')
        plt.ylabel('Friction velocity [m/s]')

        plt.savefig('figs/velocity.png', dpi=300)

class FlowBuilder:
    def __init__(self,
                 size_distrib: SizeDistribution,
                 forcemodel: ForceModel,
                 duration: float,
                 dt: float,
                 target_vel:  float,
                 acc_time: float,
                 density: float,
                 viscosity: float,
                 **kwargs,
                 ) -> None:

        # Objects
        self.size_distrib = size_distrib
        self.forcemodel = forcemodel

        # Simulation params
        self.duration = duration
        self.dt = dt
        self.target_vel = target_vel
        self.acc_time = acc_time

        # Physical quantities
        self.density = density
        self.viscosity = viscosity

    def generate(self,) -> Flow:
        # First generate the time array
        time = np.arange(0.0, self.duration, self.dt)

        lift = self.forcemodel.lift()
        drag = self.forcemodel.drag()
        burst = self.forcemodel.burst()

        # Then compute the velocity as a function of time
        if self.acc_time != 0.0:
            velocity = np.clip((self.target_vel / self.acc_time) * time, 0, self.target_vel)
        else:
            velocity = np.ones_like(time) * self.target_vel

        # Instantiate the flow class
        flow = Flow(velocity,
                    lift,
                    drag,
                    burst,
                    time,
                    )

        return flow
