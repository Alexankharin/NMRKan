#!/usr/bin/env python3
import argparse
import gc
import logging
import time
from itertools import product

import numpy as np
import pandas as pd
import torch
import sympy as sp
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from nmrkan.models import KharKAN, _clean_expr
from nmrkan.nmr import get_frequences_ordered

# --- Logging setup -----------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# --- Symbolic utilities ---------------------------------------------
def fix_powers(expr: sp.Expr) -> sp.Expr:
    if not isinstance(expr, sp.Expr):
        return expr
    return expr.replace(
        lambda e: isinstance(e, sp.Pow) and isinstance(e.exp, sp.Float) and float(e.exp).is_integer(),
        lambda e: sp.Pow(e.base, int(e.exp))
    )

def round_expr(expr: sp.Expr, digits: int = 5) -> sp.Expr:
    nums = {n: round(float(n), digits) for n in expr.atoms(sp.Number)}
    return expr.xreplace(nums)

# --- Data generation -----------------------------------------------
def f2(x: np.ndarray) -> np.ndarray:
    return np.array(get_frequences_ordered(x[:, 0], x[:, 1]))

def make_dataset(num_samples: int):
    samples = []
    while len(samples) < num_samples:
        batch = int((num_samples - len(samples)) * 1.5) + 10
        x0 = -np.abs(np.random.rand(batch) * 27 + 5)
        x1 = -np.abs(np.random.rand(batch) * 14.9 + 0.1)
        r0 = x0 / x1
        r1 = x1 / x0
        mask = (
            (np.abs(r0) < 10) & (np.abs(r1) < 10)
            & (np.abs(x0) < 10) & (np.abs(x1) < 10)
        )
        x0, x1, r0, r1 = x0[mask], x1[mask], r0[mask], r1[mask]
        for i in range(len(x0)):
            samples.append([x0[i], x1[i], r0[i], r1[i]])
    arr = np.array(samples[:num_samples])
    labels = f2(arr[:, :2])
    inputs = torch.tensor(arr, dtype=torch.float)
    targets = torch.tensor(labels, dtype=torch.float)
    logger.info(f"Generated {len(inputs)} samples")
    return inputs, targets

# --- Training -----------------------------------------------
def train_model(
    shape,
    data,
    epochs,
    lr,
    lambda_l05,
    clip_norm,
    progress_interval,
    batch_size=512,
    early_stop=1e-3,
):
    """Train ``KharKAN`` model using mini-batches.

    Returns the final model, a history :class:`~pandas.DataFrame`, and the final
    mean squared error.
    """

    model = KharKAN(shape)
    if hasattr(torch, "compile"):
        model = torch.compile(model)
    device = data["inputs"].device
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss(reduction="none")

    dataset = TensorDataset(data["inputs"], data["targets"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    history = {
        "epoch": [],
        "train_loss": [],
        "train_mse": [],
        "train_l05": [],
        "train_rel_err": [],
    }
    pbar = tqdm(total=epochs, desc=f"Training {shape}", ncols=0, leave=True)

    for epoch in range(1, epochs + 1):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            mse = (criterion(preds, yb) / yb.abs()).mean()
            l05 = model.L05_loss()
            loss = mse + lambda_l05 * l05
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        with torch.no_grad():
            full_preds = model(data["inputs"])
            rel_err = torch.mean(
                torch.abs(full_preds - data["targets"]) /
                torch.abs(data["targets"] + 1e-8)
            )
            mse_full = (criterion(full_preds, data["targets"]) / data["targets"].abs()).mean()

        if epoch % progress_interval == 0 or epoch == epochs:
            history["epoch"].append(epoch)
            history["train_loss"].append(loss.item())
            history["train_mse"].append(mse_full.item())
            history["train_l05"].append((lambda_l05 * l05).item())
            history["train_rel_err"].append(rel_err.item())
            pbar.set_postfix({
                "epoch": f"{epoch}/{epochs}",
                "mse": f"{mse_full.item():.6f}",
                "rel_err": f"{rel_err.item():.6f}",
            })
        if rel_err.item() <= early_stop:
            pbar.update(epochs - epoch + 1)
            break
        pbar.update(1)

    pbar.close()

    df_hist = pd.DataFrame(history)
    final_mse = df_hist["train_mse"].iloc[-1] if len(df_hist) else float("inf")
    return model, df_hist, final_mse

# --- Main & CLI ----------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Grid search without Lightning")
    parser.add_argument('--num-samples', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--lambda-l05', type=float, default=1e-3)
    parser.add_argument('--clip-norm', type=float, default=1.0)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    parser.add_argument('--progress-interval', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--early-stop', type=float, default=1e-3,
                        help='Stop training when relative error falls below this value')
    parser.add_argument('--output-prefix', type=str, default='outputs/grid_search')
    args = parser.parse_args()

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Prepare data
    inputs, targets = make_dataset(args.num_samples)
    inputs = inputs.to(device)
    targets = targets.to(device)
    data = {'inputs': inputs, 'targets': targets}

    # Grid search
    NUM_LAYERS = [1, 2]
    NEURONS = [1, 2, 4, 8]
    results = []

    for nl in NUM_LAYERS:
        for units in product(NEURONS, repeat=nl):
            shape = (4, *units, 3)
            logger.info(f"Starting training for shape={shape}")
            start_time = time.time()

            model, df_hist, final_mse = train_model(
                shape,
                data,
                args.epochs,
                args.lr,
                args.lambda_l05,
                args.clip_norm,
                args.progress_interval,
                batch_size=args.batch_size,
                early_stop=args.early_stop,
            )
            duration = time.time() - start_time
            eps_per_s = args.epochs / duration
            logger.info(f"Finished {shape} in {duration:.2f}s ({eps_per_s:.1f} eps/s)")

            # save history CSV
            hist_csv = f"{args.output_prefix}_l{nl}_u{'-'.join(map(str,units))}_history.csv"
            df_hist.to_csv(hist_csv, index=False)
            logger.info(f"Saved history to {hist_csv}")

            # Symbolic extraction
            formulas = model.symbolic_formula(clean=True, round_digits=5)
            exprs, complexities = {}, {}
            x0, x1, x2, x3 = sp.symbols('x_0 x_1 x_2 x_3')
            for key in ['z_0', 'z_1', 'z_2']:
                raw = formulas.get(key, None)
                expr = None
                if isinstance(raw, sp.Expr):
                    e = raw.subs({x2: x0/x1, x3: x1/x0}).expand()
                    e = fix_powers(e)
                    e = round_expr(e, 5)
                    e = _clean_expr(e, eps=1e-5)
                    expr = e
                    complexities[key] = sp.count_ops(e)
                exprs[key] = expr
            total_comp = sum(complexities.values()) if complexities else float('inf')

            results.append({
                'num_layers': nl,
                'units': list(units),
                'mse': final_mse,
                'complexity': total_comp,
                'expr0': str(exprs['z_0']),
                'expr1': str(exprs['z_1']),
                'expr2': str(exprs['z_2'])
            })

            # cleanup
            del model
            gc.collect()
            if device == 'cuda': torch.cuda.empty_cache()

    # save summary
    df_res = pd.DataFrame(results)
    summary_csv = f"{args.output_prefix}_summary.csv"
    df_res.to_csv(summary_csv, index=False)
    logger.info(f"Saved summary to {summary_csv}")

if __name__ == '__main__':
    main()
