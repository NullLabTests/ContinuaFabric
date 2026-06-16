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


class CIFAR100SuperclassBenchmark:
    """CIFAR-100 grouped into 20 superclass tasks (5 classes each).

    Each task is a 5-way classification problem.  Tasks are presented
    sequentially.  This is a challenging continual-learning benchmark
    with 20 tasks and fine-grained distinctions within each task.

    Superclasses follow the official CIFAR-100 label hierarchy:
    https://www.cs.toronto.edu/~kriz/cifar.html
    """

    # fmt: off
    SUPERCLASSES = [
        (0,    "aquatic_mammals"),       # beaver, dolphin, otter, seal, whale
        (1,    "fish"),                  # aquarium_fish, flatfish, ray, shark, trout
        (2,    "flowers"),               # orchids, poppies, roses, sunflowers, tulips
        (3,    "food_containers"),       # bottles, bowls, cans, cups, plates
        (4,    "fruit_and_vegetables"),  # apples, mushrooms, oranges, pears, sweet_peppers
        (5,    "household_electrical"),  # clock, computer_keyboard, lamp, telephone, television
        (6,    "household_furniture"),   # bed, chair, couch, table, wardrobe
        (7,    "insects"),              # bee, beetle, butterfly, caterpillar, cockroach
        (8,    "large_carnivores"),     # bear, leopard, lion, tiger, wolf
        (9,    "large_manmade_outdoor"),# bridge, castle, house, road, skyscraper
        (10,   "large_natural_scenes"), # cloud, forest, mountain, plain, sea
        (11,   "large_omnivores"),      # camel, cattle, chimpanzee, elephant, kangaroo
        (12,   "medium_mammals"),       # fox, porcupine, possum, raccoon, skunk
        (13,   "non_insect_invertebrates"), # crab, lobster, snail, spider, worm
        (14,   "people"),               # baby, boy, girl, man, woman
        (15,   "reptiles"),             # crocodile, dinosaur, lizard, snake, turtle
        (16,   "small_mammals"),        # hamster, mouse, rabbit, shrew, squirrel
        (17,   "trees"),                # maple, oak, palm, pine, willow
        (18,   "vehicles_1"),           # bicycle, bus, motorcycle, pickup_truck, train
        (19,   "vehicles_2"),           # lawn_mower, rocket, streetcar, tank, tractor
    ]
    # fmt: on
    N_SUPERCLASSES = 20
    N_FINE_PER_SUPERCLASS = 5

    def __init__(self, batch_size: int = 64, seed: int = 42):
        self.batch_size = batch_size
        self.seed = seed
        self._load_data()
        self._create_splits()

    def _load_data(self) -> None:
        try:
            import tensorflow_datasets as tfds
        except ImportError:
            raise ImportError(
                "tensorflow-datasets is required. Install with: "
                "pip install tensorflow-datasets tensorflow"
            )

        ds = tfds.load("cifar100", split=["train", "test"], as_supervised=True)
        train_ds, test_ds = ds

        def _process(ds):
            x_list, y_list = [], []
            for img, label in tfds.as_numpy(ds):
                x_list.append(img.flatten().astype(np.float32) / 255.0)
                y_list.append(label)
            return np.array(x_list), np.array(y_list)

        self.train_x, self.train_y = _process(train_ds)
        self.test_x, self.test_y = _process(test_ds)

    def _create_splits(self) -> None:
        """Create 20 task splits, each with 5 fine-grained classes."""
        self.task_splits = []
        for super_id, super_name in self.SUPERCLASSES:
            fine_start = super_id * self.N_FINE_PER_SUPERCLASS
            fine_labels = list(range(fine_start, fine_start + self.N_FINE_PER_SUPERCLASS))

            train_mask = np.isin(self.train_y, fine_labels)
            test_mask = np.isin(self.test_y, fine_labels)

            train_x_t = self.train_x[train_mask]
            train_y_t = self.train_y[train_mask]
            test_x_t = self.test_x[test_mask]
            test_y_t = self.test_y[test_mask]

            train_y_t = (train_y_t - fine_start).astype(np.int32)
            test_y_t = (test_y_t - fine_start).astype(np.int32)

            self.task_splits.append({
                "train_x": train_x_t,
                "train_y": train_y_t,
                "test_x": test_x_t,
                "test_y": test_y_t,
                "labels": fine_labels,
                "super_name": super_name,
            })

    def get_task_loader(self, task_id: int, split: str = "train"):
        """Generator yielding batches for a specific superclass task."""
        task = self.task_splits[task_id]
        x_key = f"{split}_x"
        y_key = f"{split}_y"
        x_data = task[x_key]
        y_data = task[y_key]
        n = len(x_data)

        for i in range(0, n, self.batch_size):
            batch_x = x_data[i:i + self.batch_size]
            batch_y = y_data[i:i + self.batch_size]

            y_onehot = np.zeros((len(batch_y), 5), dtype=np.float32)
            y_onehot[np.arange(len(batch_y)), batch_y] = 1.0

            yield {"x": batch_x, "y": y_onehot}

    def evaluate_on_all_tasks(
        self,
        model_fn: Callable,
        n_tasks: Optional[int] = None,
    ) -> List[float]:
        """Evaluate on all (or first n_tasks) superclass tasks."""
        n = n_tasks if n_tasks is not None else self.N_SUPERCLASSES
        accuracies = []
        for task_id in range(n):
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
