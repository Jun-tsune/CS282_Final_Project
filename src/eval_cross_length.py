import os
import json
import argparse

import torch

from src.eval import get_model_from_run, eval_model, build_evals


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Directory containing config.yaml and state.pt for a single run",
    )
    parser.add_argument(
        "--n_points",
        type=int,
        required=True,
        help="Context length to evaluate on (e.g., 2048, 4096)",
    )
    parser.add_argument(
        "--num_eval",
        type=int,
        default=256,
        help="Total number of evaluation examples (smaller is lighter)",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
        help="Maximum batch size to use during long-context eval",
    )
    args = parser.parse_args()

    model, conf = get_model_from_run(args.run_dir, step=-1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    n_dims = conf.model.n_dims
    task_name = conf.training.task
    data_name = conf.training.data

    eval_batch_size = min(args.max_batch_size, conf.training.batch_size)

    evaluation_kwargs = build_evals(conf)

    all_metrics = {}

    for eval_name, kwargs in evaluation_kwargs.items():
        local_kwargs = kwargs.copy()
        local_kwargs["task_name"] = task_name
        local_kwargs["data_name"] = data_name
        local_kwargs["n_dims"] = n_dims
        local_kwargs["n_points"] = args.n_points
        local_kwargs["num_eval_examples"] = args.num_eval
        local_kwargs["batch_size"] = eval_batch_size

        print(
            f"Running cross-length eval '{eval_name}' "
            f"with n_points={args.n_points}, batch_size={eval_batch_size}, "
            f"num_eval_examples={args.num_eval}"
        )

        metrics = eval_model(model=model, **local_kwargs)
        all_metrics[eval_name] = metrics

    save_path = os.path.join(args.run_dir, f"metrics_crosslen_{args.n_points}.json")
    with open(save_path, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"Saved cross-length metrics (all eval tasks) to {save_path}")


if __name__ == "__main__":
    main()
