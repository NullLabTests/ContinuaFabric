import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from tqdm.auto import tqdm


@dataclass
class ContinualBenchmarkResult:
    """Container for benchmark results."""
    task_order: List[str]
    accuracies: List[float]  # accuracy on each task after all tasks learned
    forgetting: List[float]  # per-task forgetting
    final_accuracy: float
    average_accuracy: float
    average_forgetting: float


class SplitMNISTBenchmark:
    """Split MNIST into 5 binary classification tasks: {0,1}, {2,3}, ...

    Each task is a 2-way classification problem.  Tasks are presented
    sequentially and the model must perform well on all 5 at the end.
    """

    def __init__(self, batch_size: int = 64, seed: int = 42):
        self.batch_size = batch_size
        self.seed = seed
        self._load_data()

    def _load_data(self) -> None:
        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError(
                "tensorflow-datasets is required. Install with: "
                "pip install tensorflow-datasets tensorflow"
            )

        ds = tfds.load("mnist", split=["train", "test"], as_supervised=True)
        train_ds, test_ds = ds

        def _process(ds, batch_size):
            x_list, y_list = [], []
            for img, label in tfds.as_numpy(ds):
                x_list.append(img.flatten().astype(np.float32) / 255.0)
                y_list.append(label)
            return np.array(x_list), np.array(y_list)

        self.train_x, self.train_y = _process(train_ds, self.batch_size)
        self.test_x, self.test_y = _process(test_ds, self.batch_size)

        self._create_splits()

    def _create_splits(self) -> None:
        self.task_splits = []
        for task_id in range(5):
            labels = [task_id * 2, task_id * 2 + 1]
            train_mask = np.isin(self.train_y, labels)
            test_mask = np.isin(self.test_y, labels)

            train_x_t = self.train_x[train_mask]
            train_y_t = self.train_y[train_mask]
            test_x_t = self.test_x[test_mask]
            test_y_t = self.test_y[test_mask]

            train_y_t = (train_y_t // 2).astype(np.int32)
            test_y_t = (test_y_t // 2).astype(np.int32)

            self.task_splits.append({
                "train_x": train_x_t,
                "train_y": train_y_t,
                "test_x": test_x_t,
                "test_y": test_y_t,
                "labels": labels,
            })

    def get_task(self, task_id: int) -> dict:
        """Get a data loader dict for a specific task."""
        task = self.task_splits[task_id]
        return task

    def get_task_loader(self, task_id: int, split: str = "train"):
        """Generator yielding batches for a specific task."""
        task = self.task_splits[task_id]
        x_key = f"{split}_x"
        y_key = f"{split}_y"
        x_data = task[x_key]
        y_data = task[y_key]
        n = len(x_data)

        for i in range(0, n, self.batch_size):
            batch_x = x_data[i:i + self.batch_size]
            batch_y = y_data[i:i + self.batch_size]

            if split == "train":
                # One-hot encode for output
                y_onehot = np.zeros((len(batch_y), 2), dtype=np.float32)
                y_onehot[np.arange(len(batch_y)), batch_y] = 1.0
                yield {"x": batch_x, "y": y_onehot}
            else:
                yield {"x": batch_x}, batch_y

    def evaluate_on_all_tasks(
        self,
        model_fn: Callable,
    ) -> List[float]:
        """Evaluate model on all 5 tasks and return per-task accuracies."""
        accuracies = []
        for task_id in range(5):
            task = self.task_splits[task_id]
            x_test = task["test_x"]
            y_test = task["test_y"]
            correct = 0
            total = 0

            for i in range(0, len(x_test), self.batch_size):
                batch_x = x_test[i:i + self.batch_size]
                batch_y = y_test[i:i + self.batch_size]

                preds = model_fn(batch_x)
                pred_labels = np.argmax(preds, axis=1)
                correct += np.sum(pred_labels == batch_y)
                total += len(batch_y)

            accuracies.append(correct / max(total, 1))

        return accuracies


class PermutedMNISTBenchmark:
    """Permuted MNIST: each task applies a different random pixel permutation.

    This tests the model's ability to handle task identity and avoid
    catastrophic forgetting across 10+ tasks with shared input statistics.
    """

    def __init__(self, n_tasks: int = 10, batch_size: int = 64, seed: int = 42):
        self.n_tasks = n_tasks
        self.batch_size = batch_size
        self.seed = seed
        self._load_data()
        self._generate_permutations()

    def _load_data(self) -> None:
        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError(
                "tensorflow-datasets is required. Install with: "
                "pip install tensorflow-datasets tensorflow"
            )

        ds = tfds.load("mnist", split=["train", "test"], as_supervised=True)
        train_ds, test_ds = ds

        def _process(ds):
            x_list, y_list = [], []
            for img, label in tfds.as_numpy(ds):
                x_list.append(img.flatten().astype(np.float32) / 255.0)
                y_list.append(label)
            return np.array(x_list), np.array(y_list)

        self.train_x, self.train_y = _process(train_ds)
        self.test_x, self.test_y = _process(test_ds)

    def _generate_permutations(self) -> None:
        rng = np.random.RandomState(self.seed)
        n_pixels = 28 * 28
        self.permutations = [
            rng.permutation(n_pixels).astype(np.int32)
            for _ in range(self.n_tasks)
        ]

    def _apply_permutation(self, x: np.ndarray, perm: np.ndarray) -> np.ndarray:
        return x[:, perm]

    def get_task_loader(self, task_id: int, split: str = "train"):
        perm = self.permutations[task_id]
        x_data = self.train_x if split == "train" else self.test_x
        y_data = self.train_y if split == "train" else self.test_y

        n = len(x_data)
        for i in range(0, n, self.batch_size):
            batch_x = x_data[i:i + self.batch_size]
            batch_y = y_data[i:i + self.batch_size]
            batch_x_perm = self._apply_permutation(batch_x, perm)

            y_onehot = np.zeros((len(batch_y), 10), dtype=np.float32)
            y_onehot[np.arange(len(batch_y)), batch_y.astype(int)] = 1.0

            yield {"x": batch_x_perm, "y": y_onehot}

    def evaluate_on_all_tasks(
        self,
        model_fn: Callable,
    ) -> List[float]:
        accuracies = []
        for task_id in range(self.n_tasks):
            perm = self.permutations[task_id]
            x_test = self._apply_permutation(self.test_x, perm)
            y_test = self.test_y
            correct = 0
            total = 0

            for i in range(0, len(x_test), self.batch_size):
                batch_x = x_test[i:i + self.batch_size]
                batch_y = y_test[i:i + self.batch_size]

                preds = model_fn(batch_x)
                pred_labels = np.argmax(preds, axis=1)
                correct += np.sum(pred_labels == batch_y)
                total += len(batch_y)

            accuracies.append(correct / max(total, 1))

        return accuracies
