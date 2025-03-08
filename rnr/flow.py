import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray

from .builder import Builder
from .config import setup_logging


# Configure module logger from config file
logger = setup_logging(__name__, 'logs/log.log')


class Flow:
    def __init__(self,
                 velocity: NDArray[np.float64],
                 time: NDArray[np.float64],
                 ) -> None:
        self.velocity = velocity
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

class FlowBuilder(Builder):
    def generate(self,) -> Flow:
        # First generate the time array
        time = np.arange(0.0, self.sim.duration, self.sim.dt)

        # Then compute the velocity as a function of time
        if self.sim.acc_time != 0.0:
            velocity = np.clip((self.sim.target_vel / self.sim.acc_time) * time, 0, self.sim.target_vel)
        else:
            velocity = np.ones_like(time) * self.sim.target_vel

        # Instantiate the flow class
        flow = Flow(velocity,
                    time,
                    )

        return flow
