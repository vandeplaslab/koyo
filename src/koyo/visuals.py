"""Visuals."""
import numpy as np


def clear_axes(i: int, axs):
    """Clear axes that were not used."""
    if i < len(axs) - 1:
        for _i in range(i + 1, len(axs)):
            axs[_i].axis("off")


def auto_clear_axes(axs):
    """Automatically clear axes. This checks whether there are any lines or collections."""
    if isinstance(axs, np.ndarray):
        axs = axs.flatten()
    for ax in axs:
        if not ax.lines and not ax.collections:
            ax.axis("off")
