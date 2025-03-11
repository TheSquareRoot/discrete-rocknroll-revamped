import matplotlib.pyplot as plt
import numpy as np

from numpy.typing import NDArray

from .config import setup_logging
from .distribution import SizeDistribution
from .aeromodel import AeroModel


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

    def __str__(self) -> str:
        return (
            f"AdhesionDistribution(\n"
            f"  velocity: {np.shape(self.velocity)}   - [{self.velocity[0]:.2e} ... {self.velocity[-1]:.2e}],\n"
            f"  lift: {np.shape(self.lift)} - [{self.lift[0,0]:.2e} ... {self.lift[-1,-1]:.2e}],\n"
            f"  burst: {np.shape(self.burst)} - [{self.burst[0]:.2e} ... {self.burst[-1]:.2e}]\n"
            f")"
        )


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

    def plot_all(self, i: int, scale: str = 'linear', **kwargs) -> None:
        plt.clf()

        # Create subplots
        fig, axs = plt.subplots(2,2, sharex=True,)
        axs[1,0].sharey(axs[1,1])

        # plot all time histories
        axs[0,0].plot(self.time, self.velocity, color='black', **kwargs)
        axs[0,1].plot(self.time, self.burst, color='green', **kwargs)
        axs[1,0].plot(self.time, self.lift[i,:], color='orange',**kwargs)
        axs[1,1].plot(self.time, self.drag[i,:], color='red', **kwargs)

        # Axis labels
        axs[0,0].set_ylabel('Friction velocity [m/s]')
        axs[0,1].set_ylabel('Burst frequency [s-1]')
        axs[1,0].set_ylabel('Lift [N]')
        axs[1,1].set_ylabel('Drag [N]')

        # Setting shared properties
        for ax in axs.flat:
            # Limits
            ax.set_xlim(left=0, right=self.time[-1])
            ax.set_ylim(bottom=0)

            # Grids
            ax.grid(True)

        plt.savefig('figs/all_aero_forces.png', dpi=300)


class FlowBuilder:
    def __init__(self,
                 size_distrib: SizeDistribution,
                 aeromodel: AeroModel,
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
        self.aeromodel = aeromodel

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

        # Then compute the velocity as a function of time
        if self.acc_time != 0.0:
            velocity = np.clip((self.target_vel / self.acc_time) * time, 0, self.target_vel)
        else:
            velocity = np.ones_like(time) * self.target_vel

        # Compute aerodynamic quantities
        lift = self.aeromodel.lift(velocity, self.size_distrib.radii_meter)
        drag = self.aeromodel.drag(velocity, self.size_distrib.radii_meter)
        burst = self.aeromodel.burst(velocity, )

        # Instantiate the flow class
        flow = Flow(velocity,
                    lift,
                    drag,
                    burst,
                    time,
                    )

        return flow
