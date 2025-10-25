import logging

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import interp1d

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
        density: float,
        viscosity: float,
        frms: float,
        read_from_file: bool | None = None,  # noqa: FBT001
        input_file: str | None = None,
        duration: float | None = None,
        dt: float | None = None,
        target_vel: float | None = None,
        acc_time: float | None = None,
        transition: str | None = None,
        perturbation: bool | None = None,  # noqa: FBT001
        **kwargs: dict,
    ) -> None:
        # Objects
        self.size_distrib = size_distrib
        self.aeromodel = aeromodel

        # Simulation params
        self.read_from_file = read_from_file
        self.input_file = input_file
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

    def _generate_velocity_from_file(self) -> tuple[NDArray, NDArray]:
        with open(f"./input/{self.input_file}.npy", "rb") as f:
            time = np.load(f)
            velocity = np.load(f)

        # Shift the signal in case initial time is not 0
        time = time - time[0]

        # Up-sample the timeseries if a timsetep is provided by the user
        if self.dt:
            # If the new time step is smaller than the current one, no resampling is done.
            # This is to avoid having to deal with aliasing.
            dt_sample = np.median(np.diff(time))

            if dt_sample > self.dt:
                # Make a new time array
                new_time = np.arange(time[0], time[-1], self.dt)

                # Interpolate velocity to new grid
                interp = interp1d(time, velocity, axis=0, kind="linear", fill_value="extrapolate")
                new_velocity = interp(new_time)

                # Replacing original arrays
                time = new_time
                velocity = new_velocity

            else:
                logger.warning("New sampling frequency is lower than the input file. No resampling.")

        # Clean the signal in case of negative velocity values
        velocity = np.max(velocity, 0)

        return time, velocity

    def _generate_velocity_from_params(self) -> tuple[NDArray, NDArray]:
        """Generate a velocity time series from user defined parameters."""
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

        return time, velocity

    def _generate_velocity(self) -> tuple[NDArray, NDArray]:
        """Wrapper function to call the proper velocity generator."""
        if self.read_from_file:
            logger.info("Reading velocity signal from file...")
            time, velocity = self._generate_velocity_from_file()
        else:
            logger.info("Generating velocity signal...")
            time, velocity = self._generate_velocity_from_params()

        return time, velocity

    def generate(
        self,
    ) -> Flow:
        # Generate velocity time series
        time, velocity = self._generate_velocity()

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
