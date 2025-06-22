import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange, tqdm
from loguru import logger

from nmrkan.models import KharKAN


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
    """
    logger.info("Starting dataset generation for {} samples", num_samples)
    collected = 0
    X0s, X1s, RXs, RYs = [], [], [], []
    Ys = []

    pbar = tqdm(total=num_samples, desc="Generating data")
    while collected < num_samples:
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

        # 4) normalize output to (batch_kept, k)
        if isinstance(out, (tuple, list)):
            Y = np.stack(out, axis=1)
        elif isinstance(out, np.ndarray):
            if out.ndim == 1:
                Y = out.reshape(-1, 1)
            elif out.ndim == 2:
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
        pbar.update(len(x0m))

    pbar.close()
    logger.info("Concatenating and slicing to exactly {} samples", num_samples)

    # 6) concatenate and slice
    x0_all = np.concatenate(X0s)[:num_samples]
    x1_all = np.concatenate(X1s)[:num_samples]
    rx_all = np.concatenate(RXs)[:num_samples]
    ry_all = np.concatenate(RYs)[:num_samples]
    y_all  = np.concatenate(Ys)[:num_samples]

    # 7) stack and convert to torch
    X = np.stack([x0_all, x1_all, rx_all, ry_all], axis=1).astype(np.float32)
    Y = y_all.astype(np.float32)

    logger.info("Dataset generation complete")
    return {
        'train_input': torch.from_numpy(X),
        'train_label': torch.from_numpy(Y),
    }


def get_perturbation(x, y):
    """
    Example `func(x, y)`: returns three arrays (f0, f1, f2).
    """
    f0 = x / np.sqrt(2) + x * x / (8 * y)
    f1 = x / np.sqrt(2) - x * x / (8 * y)
    f2 = x * np.sqrt(2)
    return (f0, f1, f2)


class GAWeightPerturbation:
    """
    Genetic algorithm wrapper that evolves KAN models by perturbing weights.
    """

    def __init__(
        self,
        shape,
        model_cls,
        inputs,
        labels,
        device=None,
        population_size=8,
        num_parents=2,
        offspring_per_parent=4,
        generations=5,
        train_steps=10000,
        lr=1e-4,
        l05_penalty=0.1,
        sigma=1e-3,
        mutation_type="gaussian",
        random_replace_p=0.05,
    ):
        logger.info(
            "Initializing GA: P={}, K={}, M={}, G={} ",
            population_size,
            num_parents,
            offspring_per_parent,
            generations,
        )
        self.shape = shape
        self.Model = model_cls
        self.inputs = inputs
        self.labels = labels
        self.device = device or torch.device("cpu")
        self.P = population_size
        self.K = num_parents
        self.M = offspring_per_parent
        self.G = generations
        self.train_steps = train_steps
        self.lr = lr
        self.l05_penalty = l05_penalty
        self.sigma = sigma
        self.mutation_type = mutation_type
        self.random_replace_p = random_replace_p
        self.criterion = nn.MSELoss(reduction="none")

        # Initialize random population
        self.population = []
        for i in range(self.P):
            model = self.Model(self.shape).to(self.device)
            self.population.append(model.state_dict())
        logger.info("Population initialized with {} individuals", self.P)

    def _train_and_score(self, state_dict):
        model = self.Model(self.shape).to(self.device)
        model.load_state_dict(state_dict)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # short training with progress bar
        for _ in trange(self.train_steps, desc="Eval train steps", leave=False):
            optimizer.zero_grad()
            preds = model(self.inputs)
            mse_loss = self.criterion(preds, self.labels).mean()
            reg_loss = model.L05_loss() * self.l05_penalty
            (mse_loss + reg_loss).backward()
            optimizer.step()

        # compute fitness
        with torch.no_grad():
            final_preds = model(self.inputs)
            fitness = nn.functional.mse_loss(final_preds, self.labels).item()
        return model.state_dict(), fitness

    def _mutate(self, parent_state):
        child_state = {}
        for key, tensor in parent_state.items():
            if isinstance(tensor, torch.Tensor) and tensor.dtype.is_floating_point:
                if self.mutation_type in ("gaussian", "both"):
                    t = tensor + torch.randn_like(tensor) * self.sigma
                else:
                    t = tensor.clone()
                if self.mutation_type in ("random_replace", "both"):
                    mask = torch.rand_like(t) < self.random_replace_p
                    t[mask] = torch.randn_like(t[mask]) * self.sigma
                child_state[key] = t
            else:
                child_state[key] = (
                    tensor.clone()
                    if isinstance(tensor, torch.Tensor)
                    else copy.deepcopy(tensor)
                )
        return child_state

    def run(self):
        logger.info("Starting GA run for {} generations", self.G)

        # 1) initial scoring of the population
        scored = [self._train_and_score(s) for s in self.population]

        for gen in range(1, self.G + 1):
            logger.info("=== Generation {}/{} ===", gen, self.G)

            # 2) pick the K best parents
            scored.sort(key=lambda x: x[1])
            parents = scored[: self.K]

            # 3) generate M offspring from each parent (mutate + train)
            children = []
            for p_state, p_fit in parents:
                for _ in range(self.M):
                    mutant = self._mutate(p_state)
                    children.append(mutant)

            # 4) score all offspring
            scored_children = [self._train_and_score(c) for c in children]

            # 5) combine parents (with their existing fitness) and offspring,
            #    then take the best P individuals overall
            combined = parents + scored_children
            combined.sort(key=lambda x: x[1])
            next_gen = combined[: self.P]

            # 6) log the fitness of the top K so you can still see “parents”
            for i, (_, fit) in enumerate(next_gen[: self.K], 1):
                logger.info("Parent #{} best fitness: {:.6f}", i, fit)

            # 7) set up for the next iteration
            self.population = [state for state, _ in next_gen]
            scored = next_gen

        # done!
        best_state, best_fit = min(scored, key=lambda x: x[1])
        logger.success("GA complete. Best MSE: {:.6f}", best_fit)
        return best_state, best_fit


if __name__ == "__main__":
    import numpy as np
    from nmrkan.nmr import get_frequences_ordered

    def get_perturbation(x, y):
        f0 = x / 2**0.5 + x * x / 8 / y
        f1 = x / 2**0.5 - x * x / 8 / y
        f2 = x * 2**0.5
        return f0, f1, f2

    # Generate dataset
    data = make_dataset_from_function(
        20000,
        get_perturbation,
        min_x=-32,
        max_x=-5,
        min_y=-15,
        max_y=-0.1,
        ratio_threshold=10,
    )
    inputs, labels = data['train_input'], data['train_label']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs, labels = inputs.to(device), labels.to(device)

    # Initialize GA
    ga = GAWeightPerturbation(
        shape=(4, 4, 3),
        model_cls=KharKAN,
        inputs=inputs,
        labels=labels,
        device=device,
        population_size=8,
        num_parents=2,
        offspring_per_parent=4,
        generations=5,
        train_steps=20000,
        lr=1e-4,
        l05_penalty=0.1,
        sigma=1.0,
        mutation_type="both",
        random_replace_p=0.1,
    )

    # Run GA
    best_state, best_score = ga.run()

    # Full retrain of the best model
    logger.info("Starting full retrain for 20000 epochs")
    final_model = KharKAN((4, 4, 3)).to(device)
    final_model.load_state_dict(best_state)
    optimizer = optim.Adam(final_model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    EPOCHS = 20000

    for _ in trange(EPOCHS, desc="Full Retrain"):
        optimizer.zero_grad()
        preds = final_model(inputs)
        mse = criterion(preds, labels)
        reg = final_model.L05_loss() * 0.1
        (mse + reg).backward()
        optimizer.step()

    exprs = final_model.symbolic_formula(round_digits=5)
    logger.success("Symbolic expressions:\n{}", exprs)
