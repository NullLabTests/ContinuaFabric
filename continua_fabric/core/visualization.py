from typing import Dict, List, Optional
import numpy as np


def plot_energy_traces(
    task_energies: Dict[str, List[float]],
    title: str = "PC Energy Convergence by Task",
    save_path: Optional[str] = None,
):
    """Plot per-task energy traces from ContinualPCEngine training.

    Args:
        task_energies: Dict mapping task_name -> list of epoch-averaged energies.
        title: Plot title.
        save_path: If given, save figure to this path (requires matplotlib).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(task_energies)))

    for (task_name, energies), color in zip(task_energies.items(), colors):
        epochs = np.arange(1, len(energies) + 1)
        ax.plot(epochs, energies, marker="o", label=task_name, color=color)
        ax.fill_between(epochs, energies, alpha=0.15, color=color)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Energy")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    plt.close()
