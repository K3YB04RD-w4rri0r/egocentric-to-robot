"""Training loop.

python wm/train.py data/human_pushes.csv --out models/video.pt

"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from model import WorldModel, fit, save, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    X, Y, push = steps(pd.read_csv(args.data))
    print(f"{len(np.unique(push))} pushes, {len(X)} steps of 0.1 s")
    start = time.time()
    model = fit(WorldModel(), X, Y, push, train_steps=args.train_steps, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    save(model, args.out, data=args.data, pushes=int(len(np.unique(push))), seed=args.seed)
    print(f"trained in {time.time() - start:.0f} s, saved to {args.out}")


if __name__ == "__main__":
    main()
