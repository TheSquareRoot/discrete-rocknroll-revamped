import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray

from .config import setup_logging


# Configure module logger from config file
logger = setup_logging(__name__, 'logs/log.log')


class Flow:
    def __init__(self,
                 velocity: NDArray[np.floating],
                 time: NDArray[np.floating],
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

class FlowBuilder:
    def __init__(self,
                 duration: float,
                 dt: float,
                 target_vel:  float,
                 acc_time: float,
                 **kwargs,
                 ) -> None:

        self.duration = duration
        self.dt = dt
        self.target_vel = target_vel
        self.acc_time = acc_time

    def generate(self,) -> Flow:
        # First generate the time array
        time = np.arange(0.0, self.duration, self.dt)

        # Then compute the velocity as a function of time
        if self.acc_time != 0.0:
            velocity = np.clip((self.target_vel / self.acc_time) * time, 0, self.target_vel)
        else:
            velocity = np.ones_like(time) * self.target_vel

        # Instantiate the flow class
        flow = Flow(velocity,
                    time,
                    )

        return flow
