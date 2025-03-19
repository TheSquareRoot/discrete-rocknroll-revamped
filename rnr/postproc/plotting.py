import matplotlib.pyplot as plt
import numpy as np

from ..core.distribution import SizeDistribution


def plot_size_distribution(size_distrib: SizeDistribution, scale: str = 'linear', **kwargs) -> None:
    """Basic bar plot of the size distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.bar(size_distrib.radii, size_distrib.weights, **kwargs)

    ax.set_xscale(scale)
    ax.set_ylim([0.0, 1.1 * np.max(size_distrib.weights)])

    ax.set_xlabel('radius [µm]')
    ax.set_ylabel('weight')

    ax.grid(True)

    fig.tight_layout()

    fig.savefig('figs/size_distrib.png', dpi=300)
    plt.close(fig)