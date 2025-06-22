import argparse
import time
import numpy as np
import torch
from tqdm import trange
import sympy as sp
from genetic_KAN import make_dataset_from_function, get_perturbation
from nmrkan.models import KharKAN, _clean_expr


def clean_up_formula(expr: dict[str, sp.Expr], name: str) -> sp.Expr | None:
    x0, x1, x2, x3 = sp.symbols("x_0 x_1 x_2 x_3")
    raw = expr.get(name, None)
    if isinstance(raw, sp.Expr):
        e = raw.subs({x2: x0/x1, x3: x1/x0}).expand()
        nums = {n: round(float(n), 5) for n in e.atoms(sp.Number)}
        e = e.xreplace(nums)
        e = _clean_expr(e, eps=1e-5)
        return e
    return None


def train(args: argparse.Namespace) -> dict:
    device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    data = make_dataset_from_function(
        args.num_samples,
        get_perturbation,
        min_x=-1,
        max_x=1,
        min_y=-1,
        max_y=1,
        ratio_threshold=30,
    )
    inputs = data["train_input"].to(device)
    labels = data["train_label"].to(device)

    model = KharKAN((4, 4, 3)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = torch.nn.MSELoss(reduction="none")

    start = time.time()
    pbar = trange(args.epochs, desc="Epochs")
    for i in pbar:
        opt.zero_grad()
        preds = model(inputs)
        mse = crit(preds, labels)
        mse_norm = mse / (torch.abs(labels) + 1e-8)
        l05_penalty = model.L05_loss()
        loss = mse.mean() + args.l05_penalty * l05_penalty + mse_norm.mean() * 100
        loss.backward()
        opt.step()
        if i % 100 == 0 or i == args.epochs - 1:
            rel_err = torch.mean(torch.abs(preds - labels) / (torch.abs(labels) + 1e-8))
            pbar.set_postfix(mse=mse.mean().item(), rel_err=rel_err.item())
    duration = time.time() - start

    expr = model.symbolic_formula(round_digits=5)
    z0 = clean_up_formula(expr, "z_0")
    result = {
        "duration": duration,
        "expr_z0": str(z0) if z0 is not None else None,
    }
    return result


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Simple KAN training for profiling")
    ap.add_argument("--num-samples", type=int, default=20000)
    ap.add_argument("--epochs", type=int, default=50000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--l05-penalty", type=float, default=0.1)
    ap.add_argument("--output-json", type=str, default=None)
    return ap.parse_args()


if __name__ == "__main__":
    args = parse()
    res = train(args)
    if args.output_json:
        import json
        with open(args.output_json, "w") as f:
            json.dump(res, f, indent=2)
    else:
        print(res)