# src/eval_cross_length.py

import os
import json
import argparse

from src.eval import get_model_from_run, eval_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--n_points", type=int, required=True)
    parser.add_argument("--num_eval", type=int, default=1280)
    args = parser.parse_args()

    model, conf = get_model_from_run(args.run_dir, step=-1)
    model = model.cuda().eval()

    n_dims = conf.model.n_dims
    batch_size = conf.training.batch_size
    task_name = conf.training.task
    data_name = conf.training.data

    metrics = eval_model(
        model=model,
        task_name=task_name,
        data_name=data_name,
        n_dims=n_dims,
        n_points=args.n_points,
        prompting_strategy="standard",
        num_eval_examples=args.num_eval,
        batch_size=batch_size,
    )

    save_path = os.path.join(args.run_dir, f"metrics_crosslen_{args.n_points}.json")
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved cross-length metrics to {save_path}")


if __name__ == "__main__":
    main()
