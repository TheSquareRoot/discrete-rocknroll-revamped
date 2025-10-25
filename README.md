# discrete-rocknroll-revamped
This is a python implementation of the Rock'n'Roll resuspension model. The model is described in [1]. The implementation uses the parametrization proposed by [2]. The main features of this implementation are:

- Discretized distributions for adhesion forces and particle size (all scipy.stats distributions are available for use!)
- Unsteady friction velocity inputs (both generated in place and read from .npy files.)
- Several drag force models.
- Plotting capabilities

Although the focus here is on the Rock'n'Roll model, any other kinetic models (models that compute a resuspension rate from a set of parameters) can be integrated into the code quite easily.

### Installation
The implementation is not available as a package as of today, so the whole repository has to be cloned. Note that the code is still WIP, so it might require a bit of troubleshooting to make it work on your platform. If you encounter issues, feel free to reach out at the mail adress given at the end.

Using uv, the project dependencies are installed with the following command:

`uv sync`

Or:

`uv pip install -r pyproject.toml`

If you are still using pip, a requirement file is also provided to be used as follows:

`pip install -r requirements.txt`

### Planned improvement

The code is still being actively developped. Here are some features that I plan to add in the future:
- Memory optimization using dask.
- Saving simulation results to plot later.
- More extensive control of the frequency content of friction velocity perturbations.
- Include the Rabinovich model for adhesion force calculation.

### Contact
Mail: victor.bourgin@universite-paris-saclay.fr \
RG: https://www.researchgate.net/profile/Victor-Bourgin-2

## References
[1] M. W. Reeks and D. Hall (2001). "Kinetic models for particle resuspension in turbulent flows: Theory and measurement." *Aerosol Science*. doi: https://doi.org/10.1016/S0021-8502(00)00063-X \
[2] L. Biasi (2001). "Use of a simple model for the interpretation of experimental data
on particle resuspension in turbulent flows." *Aerosol Science*. doi: https://doi.org/10.1016/S0021-8502(01)00048-9
