#!/usr/bin/env python3
import time
import gc
import argparse
import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import sympy as sp
import optuna
import tqdm

from nmrkan.models import KharKAN, _clean_expr
from nmrkan.nmr import get_frequences, get_frequences_ordered

# --- Set up logging -----------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Symbolic utilities ---------------------------------------------
def fix_powers(expr: sp.Expr) -> sp.Expr:
    """
    Convert float exponents that are integer values (e.g., 2.0) into actual integer exponents.
    This helps to clean up symbolic expressions for better readability.
    """
    if not isinstance(expr, sp.Expr):
        return expr
    return expr.replace(
        lambda e: isinstance(e, sp.Pow) and isinstance(e.exp, sp.Float) and float(e.exp).is_integer(),
        lambda e: sp.Pow(e.base, int(e.exp))
    )

def round_expr(expr: sp.Expr, digits: int = 5) -> sp.Expr:
    """
    Round all numerical atoms in the expression to the specified number of digits.
    """
    nums = {n: round(float(n), digits) for n in expr.atoms(sp.Number)}
    return expr.xreplace(nums)

# --- Data & target -------------------------------------------------
def f2(x: np.ndarray) -> np.ndarray:
    """
    Compute simulated NMR frequencies for a given input array x.
    """
    return np.array(get_frequences_ordered(x[:, 0], x[:, 1]))

def make_dataset(num_samples: int):
    """
    Generate a dataset for training the symbolic regressor.
    The dataset consists of random samples with constraints to avoid extreme values.
    Returns a dict with 'train_input' and 'train_label' as torch tensors.
    Ensures that the final dataset contains at least num_samples after masking.
    """
    samples_collected = 0
    xsez_list, ysez_list, ratioX_list, ratioY_list = [], [], [], []
    while samples_collected < num_samples:
        # Generate more than needed to increase chance of enough after masking
        batch_size = int((num_samples - samples_collected) * 1.5) + 10
        xsez = -np.abs(np.random.rand(batch_size) * 27 + 5)  # from -32 to -5
        ysez = -np.abs(np.random.rand(batch_size) * 14.9 + 0.1)  # from -15 to -0.1
        ratioX = xsez / ysez
        ratioY = ysez / xsez
        TH = 10
        mask = (
            (np.abs(ratioX) < TH)
            & (np.abs(ratioY) < TH)
            & (np.abs(xsez) < TH)
            & (np.abs(ysez) < TH)
        )
        xsez, ysez, ratioX, ratioY = xsez[mask], ysez[mask], ratioX[mask], ratioY[mask]
        xsez_list.append(xsez)
        ysez_list.append(ysez)
        ratioX_list.append(ratioX)
        ratioY_list.append(ratioY)
        samples_collected += len(xsez)
    # Concatenate and trim to exactly num_samples
    xsez = np.concatenate(xsez_list)[:num_samples]
    ysez = np.concatenate(ysez_list)[:num_samples]
    ratioX = np.concatenate(ratioX_list)[:num_samples]
    ratioY = np.concatenate(ratioY_list)[:num_samples]
    answers = f2(np.stack([xsez, ysez], axis=1))
    logger.info(f"Generated {len(xsez)} samples with shape {xsez.shape}")
    logger.info(
        f"Labels stats: min={answers.min()}, max={answers.max()}, mean={answers.mean()}"
    )
    return {
        'train_input': torch.tensor(
            np.stack([xsez, ysez, ratioX, ratioY], axis=1)
        ).float(),
        'train_label': torch.tensor(answers).float()
    }

# --- Training & extraction ----------------------------------------
def evaluate_architecture(
    shape,
    data,
    epochs: int,
    lr: float,
    device: str,
    lambda_l05: float,
    clip_norm: float,
    batch_size: int = 512,
    early_stop: float = 1e-3,
):
    """
    Train a KharKAN model with the given architecture and hyperparameters.
    Returns final MSE, total symbolic complexity, and symbolic expressions for each output.
    """
    logger.info(f"Evaluating shape={shape}, epochs={epochs}, lr={lr}")
    t_start = time.time()

    model = KharKAN(shape).to(device)
    if hasattr(torch, "compile"):
        model = torch.compile(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    inputs = data['train_input'].to(device)
    labels = data['train_label'].to(device)
    dataset = TensorDataset(inputs, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    loop_start = time.time()
    pbar = tqdm.tqdm(range(epochs), desc="Epochs")
    for i in pbar:
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            mse = criterion(preds, yb)
            l05_penalty = model.L05_loss()
            loss = mse + lambda_l05 * l05_penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
            optimizer.step()
        with torch.no_grad():
            full_preds = model(inputs)
            rel_err = torch.mean(
                torch.abs(full_preds - labels) / torch.abs(labels + 1e-8)
            )
        if i % 100 == 0 or i == epochs - 1:
            pbar.set_postfix(
                {
                    "mse": mse.item(),
                    "l05": (lambda_l05 * l05_penalty).item(),
                    "loss": loss.item(),
                    "rel_err": rel_err.item(),
                }
            )
        if rel_err.item() <= early_stop:
            pbar.update(epochs - i)
            break
    loop_duration = time.time() - loop_start
    logger.info(f"Training loop duration: {loop_duration:.2f}s")

    # Compute final MSE
    with torch.no_grad():
        final_mse = criterion(model(inputs), labels).item()

    # Symbolic extraction
    sym_start = time.time()
    formulas = model.symbolic_formula(clean=True, round_digits=5)
    sym_duration = time.time() - sym_start
    logger.info(f"Symbolic extraction duration: {sym_duration:.2f}s")

    exprs = {}
    complexities = {}
    x0, x1, x2, x3 = sp.symbols('x_0 x_1 x_2 x_3')
    for i, key in enumerate(['z_0', 'z_1', 'z_2']):
        raw = formulas.get(key, None)
        expr_key = None
        if raw is not None and isinstance(raw, sp.Expr):
            # Substitute ratio features for interpretability
            e = raw.subs({x2: x0/x1, x3: x1/x0}).expand()
            e = fix_powers(e)
            e = round_expr(e, 5)
            e = _clean_expr(e, eps=1e-5)
            expr_key = e
            complexities[key] = sp.count_ops(e)
        exprs[key] = expr_key
    # If any output is missing, return inf to penalize this architecture
    bad_any = any(exprs[k] is None for k in ['z_0','z_1','z_2'])
    if bad_any:
        # model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        return float('inf'), float('inf'), None, None, None

    # model.cpu()
    del model, optimizer, criterion, preds, inputs, labels
    gc.collect()
    torch.cuda.empty_cache()

    total_duration = time.time() - t_start
    logger.info(f"Total evaluate_architecture duration: {total_duration:.2f}s")

    return (
        final_mse,
        complexities['z_0'] + complexities['z_1'] + complexities['z_2'],
        exprs['z_0'],
        exprs['z_1'],
        exprs['z_2'],
    )

# --- Main & CLI ----------------------------------------------------
def main():
    """
    Main entrypoint for running symbolic regression architecture search.
    Uses Optuna for hyperparameter optimization.
    """
    parser = argparse.ArgumentParser(
        description="Search KharKAN architectures with Optuna"
    )
    parser.add_argument(
        "--num-samples", type=int, default=10000,
        help="Number of random samples to generate for dataset"
    )
    parser.add_argument(
        "--epochs", type=int, default=100000,
        help="Training epochs per trial"
    )
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument(
        "--lambda-l05",
        type=float,
        default=1e-3,
        help="L0.5 regularization weight (sum of absolute values of weights)",
    )
    parser.add_argument(
        "--clip-norm", type=float, default=1.0,
        help="Gradient clipping norm"
    )
    parser.add_argument(
        "--n-trials", type=int, default=10, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--study-name", type=str, default="kan_multiobj_per_layer",
        help="Optuna study name"
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="outputs/pareto_trials",
        help="Prefix for saved CSV/pickle",
    )
    parser.add_argument(
        "--device", choices=["cpu", "cuda"], default=None,
        help="Device to use (auto-detect if unset)"
    )
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument(
        "--early-stop",
        type=float,
        default=1e-3,
        help="Stop training when relative error drops below this value",
    )
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Generate training data
    dataset = make_dataset(args.num_samples)

    # Set up Optuna study for multi-objective optimization
    study = optuna.create_study(
        directions=['minimize','minimize'],
        study_name=args.study_name
    )

    def objective_wrapper(trial):
        """
        Objective function for Optuna trial. Samples architecture and trains model.
        Returns MSE and symbolic complexity.
        """
        trial_start = time.time()
        nl = trial.suggest_int('num_layers', 1, 2)
        units = [
            trial.suggest_int(f'neurons_layer_{i+1}', 2, 8, log=True)
            for i in range(nl)
        ]
        shape = (4, *units, 3)
        logger.info(f"Trial {trial.number}: layers={nl}, units={units}")

        mse, comp, e0, e1, e2 = evaluate_architecture(
            shape,
            dataset,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            lambda_l05=args.lambda_l05,
            clip_norm=args.clip_norm,
            batch_size=args.batch_size,
            early_stop=args.early_stop,
        )
        dur = time.time() - trial_start
        logger.info(f"Trial {trial.number} duration: {dur:.2f}s")

        trial.set_user_attr('units', units)
        trial.set_user_attr('total_neurons', sum(units))
        trial.set_user_attr('expr0', e0)
        trial.set_user_attr('expr1', e1)
        trial.set_user_attr('expr2', e2)
        gc.collect()
        torch.cuda.empty_cache()
        return mse, comp

    try:
        study.optimize(objective_wrapper, n_trials=args.n_trials)
    except KeyboardInterrupt:
        logger.warning("Optimization interrupted with Ctrl+C — saving partial results...")

    # Collect and save best results
    records = []
    for t in study.best_trials:
        records.append({
            'layers': len(t.user_attrs['units']),
            'units': t.user_attrs['units'],
            'total_neurons': t.user_attrs['total_neurons'],
            'mse': t.values[0],
            'complexity': t.values[1],
            'expr0': t.user_attrs['expr0'],
            'expr1': t.user_attrs['expr1'],
            'expr2': t.user_attrs['expr2'],
        })

    df = pd.DataFrame(records).sort_values(['mse','complexity']).reset_index(drop=True)
    pickle_path = f"{args.output_prefix}.pkl"
    csv_path    = f"{args.output_prefix}.csv"
    df.to_pickle(pickle_path)
    df.to_csv(csv_path, index=False)

    logger.info(f"Saved results to {pickle_path} and {csv_path}")
    print(df)

if __name__ == "__main__":
    main()
