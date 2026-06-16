import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field


@dataclass
class CLMetrics:
    """Container for continual learning evaluation metrics.

    All metrics are computed from a matrix R of shape (T, T) where
    R[i][j] = performance (lower energy is better, higher accuracy is better)
    on task j *after* learning task i.

    Reference:
        Lopez-Paz & Ranzato (2017) "Gradient Episodic Memory for Continual
        Learning" — https://arxiv.org/abs/1706.08840
    """

    task_names: List[str]
    performance_matrix: np.ndarray  # shape (T, T), R[i][j] = perf on task j after task i
    higher_is_better: bool = True   # True for accuracy, False for energy

    @property
    def n_tasks(self) -> int:
        return len(self.task_names)

    def accuracy(self) -> float:
        """Average performance on all tasks after learning the last task."""
        return float(np.mean(self.performance_matrix[-1]))

    def backward_transfer(self) -> np.ndarray:
        """Per-task BWT: how much performance on task j changed after learning later tasks.

        BWT_j = R[T-1][j] - R[j][j]
        Negative BWT = forgetting, positive BWT = improvement.
        """
        bwt = np.zeros(self.n_tasks)
        for j in range(self.n_tasks):
            after_task_j = self.performance_matrix[j, j]
            at_end = self.performance_matrix[-1, j]
            bwt[j] = at_end - after_task_j
        return bwt

    def average_backward_transfer(self) -> float:
        """Average BWT across all tasks (excluding the last)."""
        bwt = self.backward_transfer()
        if self.n_tasks <= 1:
            return 0.0
        return float(np.mean(bwt[:-1]))

    def forward_transfer(self) -> np.ndarray:
        """Per-task FWT: how much learning earlier tasks helps with task j.

        FWT_j = R[j-1][j] - R_random[j]   (for j > 0)
        where R_random is performance before any training on task j.

        Positive FWT = positive transfer.
        """
        fwt = np.zeros(self.n_tasks)
        fwt[0] = 0.0  # no forward transfer for the first task
        for j in range(1, self.n_tasks):
            before_training = self.performance_matrix[0, j]
            after_prev = self.performance_matrix[j - 1, j]
            fwt[j] = after_prev - before_training
        return fwt

    def average_forward_transfer(self) -> float:
        """Average FWT across all tasks (excluding the first)."""
        fwt = self.forward_transfer()
        if self.n_tasks <= 1:
            return 0.0
        return float(np.mean(fwt[1:]))

    def forgetting(self) -> np.ndarray:
        """Per-task forgetting: drop in performance from best to final."""
        forget = np.zeros(self.n_tasks)
        for j in range(self.n_tasks):
            best = np.max(self.performance_matrix[:j + 1, j])
            final = self.performance_matrix[-1, j]
            forget[j] = best - final
        return forget

    def average_forgetting(self) -> float:
        return float(np.mean(self.forgetting()))

    def plasticity(self) -> float:
        """Plasticity: ability to learn new tasks, measured as average
        improvement from first to last encounter of each task."""
        improvements = []
        for j in range(self.n_tasks):
            relevant = self.performance_matrix[j:, j]
            if len(relevant) > 1:
                imp = relevant[-1] - relevant[0]
                improvements.append(imp)
        if not improvements:
            return 0.0
        return float(np.mean(improvements))

    def summary(self) -> Dict[str, float]:
        return {
            "n_tasks": self.n_tasks,
            "accuracy": self.accuracy(),
            "avg_backward_transfer": self.average_backward_transfer(),
            "avg_forward_transfer": self.average_forward_transfer(),
            "avg_forgetting": self.average_forgetting(),
            "plasticity": self.plasticity(),
        }


def count_parameters(params: any) -> int:
    """Count total trainable parameters in a PC graph params tree."""
    return sum(
        int(x.size) for x in jax.tree_util.tree_leaves(params)
    )


def compute_cl_metrics(
    eval_fn: Callable,
    params: any,
    task_loaders: List[any],
    structure: any,
    rng_key: jax.Array,
    higher_is_better: bool = True,
) -> CLMetrics:
    """Convenience: build a performance matrix and return CLMetrics.

    eval_fn(params, structure, task_loader, rng_key) should return a
    scalar performance value (higher = better for accuracy, lower = better for energy).
    """
    n_tasks = len(task_loaders)
    perf_matrix = np.zeros((n_tasks, n_tasks))

    for after_task in range(n_tasks):
        for tid in range(n_tasks):
            perf = eval_fn(params, structure, task_loaders[tid], rng_key)
            perf_matrix[after_task, tid] = perf

    task_names = [f"task_{i}" for i in range(n_tasks)]
    return CLMetrics(
        task_names=task_names,
        performance_matrix=perf_matrix,
        higher_is_better=higher_is_better,
    )
