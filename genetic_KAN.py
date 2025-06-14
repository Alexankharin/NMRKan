import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange

from src.kharkan.modelKAN import KharKAN


def make_dataset_from_function(
    num_samples: int,
    func,
    min_x: float = -32,
    max_x: float = -5,
    min_y: float = -15,
    max_y: float = -0.1,
    ratio_threshold: float = 100,
):
    """
    Generate inputs (x0, x1, x0/x1, x1/x0) and labels via `func(x0, x1)`.

    Guarantees:
      - train_input:  torch.FloatTensor, shape (num_samples, 4)
      - train_label:  torch.FloatTensor, shape (num_samples, k)

    func(x0, x1) can return:
      - tuple/list of k 1D arrays, each of length batch_kept
      - a 1D array of length batch_kept   → treated as k=1
      - a 2D array shape (batch_kept, k)
    """
    collected = 0
    X0s, X1s, RXs, RYs = [], [], [], []
    Ys = []

    while collected < num_samples:
        # oversample so that after masking we get enough points
        batch_size = int((num_samples - collected) * 1.5) + 10

        # 1) random in [min_x, max_x), [min_y, max_y)
        x0 = np.random.uniform(min_x, max_x, size=batch_size)
        x1 = np.random.uniform(min_y, max_y, size=batch_size)

        # 2) filter out extreme ratios
        ratioX = x0 / x1
        ratioY = x1 / x0
        mask = (np.abs(ratioX) < ratio_threshold * 10) & (np.abs(ratioY) < ratio_threshold * 10)
        x0m, x1m = x0[mask], x1[mask]
        rxm, rym = ratioX[mask], ratioY[mask]

        # 3) call user function
        out = func(x0m, x1m)

        # 4) normalize `out` to a (batch_kept, k) array
        if isinstance(out, (tuple, list)):
            # tuple/list of k arrays length batch_kept
            Y = np.stack(out, axis=1)
        elif isinstance(out, np.ndarray):
            if out.ndim == 1:
                # single-output → (batch_kept, 1)
                Y = out.reshape(-1, 1)
            elif out.ndim == 2:
                # already (batch_kept, k)
                Y = out
            else:
                raise ValueError(f"func returned array with ndim={out.ndim}")
        else:
            raise ValueError("func must return tuple, list, or numpy.ndarray")

        # 5) collect
        X0s.append(x0m)
        X1s.append(x1m)
        RXs.append(rxm)
        RYs.append(rym)
        Ys.append(Y)

        collected += len(x0m)

    # 6) concatenate and slice to exactly num_samples
    x0_all = np.concatenate(X0s)[:num_samples]
    x1_all = np.concatenate(X1s)[:num_samples]
    rx_all = np.concatenate(RXs)[:num_samples]
    ry_all = np.concatenate(RYs)[:num_samples]
    y_all  = np.concatenate(Ys)[:num_samples]

    # 7) stack inputs and convert to torch
    X = np.stack([x0_all, x1_all, rx_all, ry_all], axis=1).astype(np.float32)
    Y = y_all.astype(np.float32)

    return {
        'train_input': torch.from_numpy(X),
        'train_label': torch.from_numpy(Y),
    }

def get_perturbation(x, y):
    """
    Example `func(x, y)`: returns three arrays (f0, f1, f2),
    each of shape (batch_kept,).
    """
    f0 = x / np.sqrt(2) + x * x / (8 * y)
    f1 = x / np.sqrt(2) - x * x / (8 * y)
    f2 = x * np.sqrt(2)
    return (f0, f1, f2)

class GAWeightPerturbation:
    """
    A simple genetic-algorithm wrapper that evolves a population of KAN models
    by perturbing their weights. Supports two mutation modes:
      1) Gaussian noise added to every weight tensor.
      2) Random replacement of a fraction p of weights with new random values.

    Usage:
      - Initialize with your hyperparameters and data tensors.
      - Call `best_state, best_fitness = ga.run()` to evolve and get the best model.
      - Optionally, fully retrain `best_state` for more epochs and extract symbolic formulas.
    """

    def __init__(
        self,
        shape,
        model_cls,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        device: torch.device = None,
        population_size: int = 10,
        num_parents: int = 4,
        offspring_per_parent: int = 2,
        generations: int = 5,
        train_steps: int = 10000,
        lr: float = 1e-4,
        l05_penalty: float = 0.1,
        sigma: float = 1e-3,
        mutation_type: str = 'gaussian',  # 'gaussian' or 'random_replace'
        random_replace_p: float = 0.05,    # fraction of weights to replace if using random_replace
    ):
        """
        :param shape: tuple for KharKAN constructor, e.g. (4,4,3)
        :param model_cls: class of your model, e.g. KharKAN
        :param inputs, labels: full training set (already on device)
        :param population_size: number of individuals per generation
        :param num_parents: how many top performers to select each gen
        :param offspring_per_parent: how many children each parent produces
        :param generations: how many GA iterations to run
        :param train_steps: training steps per individual per generation
        :param lr: learning rate for Adam during evaluation
        :param l05_penalty: weight of L0.5 regularization in loss
        :param sigma: standard deviation for Gaussian perturbations
        :param mutation_type: 'gaussian' or 'random_replace'
        :param random_replace_p: if mutation_type=='random_replace', this fraction of weights is reset
        """
        self.shape = shape
        self.Model = model_cls
        self.inputs = inputs
        self.labels = labels
        self.device = device or torch.device("cpu")

        # GA hyperparameters
        self.P = population_size
        self.K = num_parents
        self.M = offspring_per_parent
        self.G = generations
        self.train_steps = train_steps

        # Training hyperparameters
        self.lr = lr
        self.l05_penalty = l05_penalty

        # Mutation hyperparameters
        self.sigma = sigma
        self.mutation_type = mutation_type
        self.random_replace_p = random_replace_p

        self.criterion = nn.MSELoss(reduction="none")

        # Initialize population: list of model state_dicts with fresh random weights
        self.population = []
        for _ in range(self.P):
            model = self.Model(self.shape).to(self.device)
            self.population.append(model.state_dict())

    def _train_and_score(self, state_dict):
        """
        Instantiate a KAN, load its weights, train for self.train_steps,
        and return (updated_state_dict, fitness_score).
        Fitness is plain MSE on the training set after the small-budget training.
        """
        # 1) Build model and optimizer
        model = self.Model(self.shape).to(self.device)
        model.load_state_dict(state_dict)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # 2) Short training loop
        for _ in range(self.train_steps):
            optimizer.zero_grad()
            preds = model(self.inputs)
            mse_loss = self.criterion(preds, self.labels).mean()
            reg_loss = model.L05_loss() * self.l05_penalty
            loss = mse_loss + reg_loss
            loss.backward()
            optimizer.step()

        # 3) Compute final fitness (lower is better)
        with torch.no_grad():
            final_preds = model(self.inputs)
            fitness = nn.functional.mse_loss(final_preds, self.labels).item()

        # Return its new state_dict and fitness for selection
        return model.state_dict(), fitness

    def _select_parents(self, scored_population):
        """
        Select the top-K individuals by lowest fitness.
        scored_population: list of (state_dict, fitness)
        """
        scored_population.sort(key=lambda x: x[1])  # sort by fitness ascending
        parents = [sd for sd, _ in scored_population[: self.K]]
        return parents

    def _mutate(self, parent_state):
        """
        Produce one child state dict from the parent_state by applying one of:
          - Gaussian noise to every weight tensor (mutation_type='gaussian')
          - Random replacement of p% of entries in each tensor (mutation_type='random_replace')
        Non-weight buffers (e.g. running_mean) are cloned unchanged.
        """
        child_state = {}
        for key, tensor in parent_state.items():
            if isinstance(tensor, torch.Tensor) and tensor.dtype.is_floating_point:
                # Decide mutation mode
                if self.mutation_type == 'gaussian':
                    # Add small Gaussian noise to every element
                    noise = torch.randn_like(tensor) * self.sigma
                    child_state[key] = tensor + noise

                elif self.mutation_type == 'random_replace':
                    # Copy original tensor
                    t = tensor.clone()
                    # Create mask for entries to replace
                    mask = torch.rand_like(t) < self.random_replace_p
                    # Replace masked entries with new random values (Gaussian noise)
                    t[mask] = torch.randn_like(t[mask]) * self.sigma
                    child_state[key] = t

                else:
                    raise ValueError(f"Unknown mutation_type {self.mutation_type}")

            else:
                # Non-float tensors (e.g. int buffers) or non-tensors: just clone
                if isinstance(tensor, torch.Tensor):
                    child_state[key] = tensor.clone()
                else:
                    child_state[key] = copy.deepcopy(tensor)

        return child_state

    def run(self):
        """
        Execute the full GA:
          - For each generation:
              * Evaluate & score all individuals
              * Select top-K parents
              * Produce new population via mutation
          - After all generations, do one more evaluation to pick the absolute best
        Returns: (best_state_dict, best_fitness)
        """
        for gen in range(1, self.G + 1):
            print(f"\n=== Generation {gen}/{self.G} ===")
            scored = []
            # Evaluate every individual in the current population
            for idx, state in enumerate(self.population, start=1):
                new_state, fit = self._train_and_score(state)
                scored.append((new_state, fit))
                print(f"  Individual {idx}/{self.P} → MSE: {fit:.6f}")

            # Select parents
            parents = self._select_parents(scored)
            best_fit = min(f for _, f in scored)
            print(f"  → Best fitness this gen: {best_fit:.6f}")

            # Build next generation
            next_pop = []
            # 1) Keep parents unchanged
            next_pop.extend(parents)
            # 2) For each parent, create M offspring
            for parent in parents:
                for _ in range(self.M):
                    child = self._mutate(parent)
                    next_pop.append(child)

            # Trim or pad to maintain population size P
            self.population = next_pop[: self.P]
            while len(self.population) < self.P:
                # If somehow too few (unlikely), add a fresh random model
                m = self.Model(self.shape).to(self.device)
                self.population.append(m.state_dict())

        # Final evaluation: pick the absolute best
        final_scored = [self._train_and_score(s) for s in self.population]
        fits = [f for _, f in final_scored]
        best_idx = int(np.argmin(fits))
        best_state, best_fit = final_scored[best_idx]
        print(f"\n*** GA complete. Best individual #{best_idx} → MSE: {best_fit:.6f}")
        return best_state, best_fit


if __name__ == "__main__":
    # === Example usage ===
    import numpy as np
    from src.kharkan.NMR import get_frequences_ordered

    def get_perturbation(x, y):
        """Synthetic function: returns three outputs based on x,y."""
        f0 = x / 2**0.5 + x * x / 8 / y
        f1 = x / 2**0.5 - x * x / 8 / y
        f2 = x * 2**0.5
        return f0, f1, f2

    # Generate a toy dataset
    def make_dataset(n):
        x0 = np.random.uniform(-1, 1, n)
        x1 = np.random.uniform(-1, 1, n)
        ratioX = x0 / x1
        ratioY = x1 / x0
        labels = np.stack(get_perturbation(x0, x1), axis=1)
        inputs = np.stack([x0, x1, ratioX, ratioY], axis=1)
        return (
            torch.tensor(inputs, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.float32),
        )
    data = make_dataset_from_function(20000, get_perturbation, min_x=-32, max_x=-5, min_y=-15, max_y=-0.1, ratio_threshold=10)
    inputs, labels = data['train_input'], data['train_label']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, labels = inputs.to(device), labels.to(device)
    # Instantiate the GA with either 'gaussian' or 'random_replace'
    ga = GAWeightPerturbation(
        shape=(4, 4, 3),
        model_cls=KharKAN,
        inputs=inputs,
        labels=labels,
        device=device,
        population_size=16,
        num_parents=4,
        offspring_per_parent=4,
        generations=5,
        train_steps=10000,
        lr=1e-4,
        l05_penalty=0.1,
        sigma=1,  # standard deviation for Gaussian noise or gaussuan replacement
        mutation_type='random_replace',  # try 'gaussian' or 'random_replace'
        random_replace_p=0.1,            # 10% of weights reset randomly
    )

    # Run the GA and get the best model so far
    best_state, best_score = ga.run()

    # Finally: full retrain of the winner (optional)
    final_model = KharKAN((4, 4, 3)).to(device)
    final_model.load_state_dict(best_state)
    optimizer = optim.Adam(final_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    EPOCHS = 200000

    for epoch in trange(EPOCHS, desc="Full Retrain"):
        optimizer.zero_grad()
        preds = final_model(inputs)
        mse = criterion(preds, labels)
        reg = final_model.L05_loss() * 0.1
        (mse + reg).backward()
        optimizer.step()

    # Extract symbolic formulas
    exprs = final_model.symbolic_formula(round_digits=5)
    print(exprs)
