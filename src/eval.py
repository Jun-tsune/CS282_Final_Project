import json
import os
import sys
import glob

from munch import Munch
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import yaml
import models.full_models as full_models
from src.utils.samplers import get_data_sampler, sample_transformation
from src.utils.utils_model import build_model
from tasks import get_task_sampler


def get_model_from_run(run_path, step=-1, only_conf=False):
    config_path = os.path.join(run_path, "config.yaml")
    with open(config_path) as fp:  # we don't Quinfig it to avoid inherits
        conf = Munch.fromDict(yaml.safe_load(fp))
    if only_conf:
        return None, conf

    model = full_models.build_model(conf.model)

    if step == -1:
        # Try to load state.pt first (latest checkpoint)
        state_path = os.path.join(run_path, "state.pt")
        if os.path.exists(state_path):
            state = torch.load(state_path)
            model.load_state_dict(state["model_state_dict"])
        else:
            # If state.pt doesn't exist, try to find the latest model_*.pt file
            model_files = glob.glob(os.path.join(run_path, "model_*.pt"))
            if model_files:
                # Sort by step number and get the latest one
                def get_step(fname):
                    basename = os.path.basename(fname)
                    try:
                        return int(basename.replace("model_", "").replace(".pt", ""))
                    except:
                        return -1
                model_files.sort(key=get_step, reverse=True)
                latest_model = model_files[0]
                print(f"Warning: state.pt not found, using latest checkpoint: {os.path.basename(latest_model)}")
                state_dict = torch.load(latest_model)
                model.load_state_dict(state_dict)
            else:
                raise FileNotFoundError(
                    f"No checkpoint files found in {run_path}. "
                    f"Training may not have completed yet. "
                    f"Expected files: state.pt or model_*.pt"
                )
    else:
        model_path = os.path.join(run_path, f"model_{step}.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Checkpoint file not found: {model_path}. "
                f"Available checkpoints may have different step numbers."
            )
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)

    return model, conf


# Functions for evaluation


def _get_model_device(model):
    """Infer the device a model is currently on."""
    if hasattr(model, "device"):
        dev = getattr(model, "device")
        if isinstance(dev, torch.device):
            return dev
        return torch.device(dev)

    try:
        param = next(model.parameters())
        return param.device
    except StopIteration:
        # Model without parameters defaults to CPU
        return torch.device("cpu")
    except AttributeError:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_batch(model, task_sampler, xs, xs_p=None):
    task = task_sampler()
    device = _get_model_device(model)

    if xs_p is None:
        ys = task.evaluate(xs)
        output = model(xs.to(device), ys.to(device))
        if isinstance(output, tuple):
            pred = output[0]
        else:
            pred = output.detach()
        metrics = task.get_metric()(pred.cpu(), ys)
    else:
        b_size, n_points, _ = xs.shape
        metrics = torch.zeros(b_size, n_points)
        for i in range(n_points):
            xs_comb = torch.cat((xs[:, :i, :], xs_p[:, i:, :]), dim=1)
            ys = task.evaluate(xs_comb)

            output = model(xs_comb.to(device), ys.to(device), inds=[i]).detach()
            if isinstance(output, tuple):
                pred = output[0]
            else:
                pred = output.detach()
            metrics[:, i] = task.get_metric()(pred.cpu(), ys)[:, i]

    return metrics


# Functions for generating different kinds of train/test data


def gen_standard(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)

    return xs, None


def gen_opposite_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = -xs_train_pre

    return xs_train_pre, xs_test_post


def gen_random_quadrants(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    pattern = torch.randn([b_size, 1, xs.shape[2]]).sign()

    xs_train_pre = xs.abs() * pattern
    xs_test_post = xs

    return xs_train_pre, xs_test_post


def gen_orthogonal_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    n_dim = xs.shape[2]
    n_points = min(n_points, n_dim)
    # raise ValueError("number of points should be at most the dimension.")
    xs_train_pre = xs
    xs_test_post = torch.zeros(xs.shape)
    for i in range(n_points):
        xs_test_post_i = xs[:, i : i + 1, :]
        xs_train_pre_i = xs[:, :i, :]
        _, _, Vt = torch.linalg.svd(xs_train_pre_i, full_matrices=False)
        xs_train_pre_i_projection = Vt.transpose(1, 2) @ Vt
        xs_test_post_i_orthogonalized = (
            xs_test_post_i - xs_test_post_i @ xs_train_pre_i_projection
        )
        xs_test_post_i_normalized = (
            xs_test_post_i_orthogonalized
            * xs_test_post_i.norm(dim=2).unsqueeze(2)
            / xs_test_post_i_orthogonalized.norm(dim=2).unsqueeze(2)
        )

        xs_test_post[:, i : i + 1, :] = xs_test_post_i_normalized

    return xs_train_pre, xs_test_post


def gen_overlapping_train_test(data_sampler, n_points, b_size):
    xs = data_sampler.sample_xs(n_points, b_size)
    xs_train_pre = xs
    xs_test_post = xs.clone()
    b_size = xs.shape[0]
    for i in range(1, n_points):
        xs_train_pre_i = xs[:, :i, :]
        perm = torch.stack([torch.randperm(i) for _ in range(b_size)]).unsqueeze(dim=1)
        ind_mat = (perm == 0) + 0.0
        xs_test_post[:, i : i + 1, :] = ind_mat @ xs_train_pre_i

    return xs_train_pre, xs_test_post


def aggregate_metrics(metrics, bootstrap_trials=1000):
    """
    Takes as input a tensor of shape (num_eval, n_points) and returns a dict with
    per-point mean, stddev, and bootstrap limits
    """
    results = {}
    results["mean"] = metrics.mean(dim=0)
    results["std"] = metrics.std(dim=0, unbiased=True)
    n = len(metrics)
    bootstrap_indices = torch.randint(n, size=(bootstrap_trials, n))
    bootstrap_means = metrics[bootstrap_indices].mean(dim=1).sort(dim=0)[0]
    results["bootstrap_low"] = bootstrap_means[int(0.05 * bootstrap_trials), :]
    results["bootstrap_high"] = bootstrap_means[int(0.95 * bootstrap_trials), :]

    return {k: v.tolist() for k, v in results.items()}


def eval_model(
    model,
    task_name,
    data_name,
    n_dims,
    n_points,
    prompting_strategy,
    num_eval_examples=1280,
    batch_size=64,
    data_sampler_kwargs={},
    task_sampler_kwargs={},
):
    """
    Evaluate a model on a task with a variety of strategies.
       Args:
       - task: which base task we are evaluating on. E.g., "linear_regression"
       - prompting_strategy: how to construct the prompt, e.g., "random_quadrants"
       - num_eval_examples: total number of examples to evaluate on
       - **sampler_kwargs: remaining arguments to pass directly to the sampler
    """

    assert num_eval_examples % batch_size == 0
    data_sampler = get_data_sampler(data_name, n_dims, **data_sampler_kwargs)
    task_sampler = get_task_sampler(
        task_name, n_dims, batch_size, **task_sampler_kwargs
    )

    all_metrics = []

    generating_func = globals()[f"gen_{prompting_strategy}"]
    for i in range(num_eval_examples // batch_size):
        xs, xs_p = generating_func(data_sampler, n_points, batch_size)

        metrics = eval_batch(model, task_sampler, xs, xs_p)
        all_metrics.append(metrics)

    metrics = torch.cat(all_metrics, dim=0)

    return aggregate_metrics(metrics)


def build_evals(conf, only_standard=True):
    """
    Build evaluation configurations.
    
    Args:
        conf: Configuration object
        only_standard: If True, only evaluate 'standard' strategy (default: True)
    """
    n_dims = conf.model.n_dims
    n_points = conf.training.curriculum.points.end
    batch_size = conf.training.batch_size

    task_name = conf.training.task
    data_name = conf.training.data

    base_kwargs = {
        "task_name": task_name,
        "n_dims": n_dims,
        "n_points": n_points,
        "batch_size": batch_size,
        "data_name": data_name,
        "prompting_strategy": "standard",
    }

    evaluation_kwargs = {}

    evaluation_kwargs["standard"] = {"prompting_strategy": "standard"}
    
    # If only_standard is True, return early with just the standard strategy
    if only_standard:
        for name, kwargs in evaluation_kwargs.items():
            # allow kwargs to override base_kwargs values
            evaluation_kwargs[name] = base_kwargs.copy()
            evaluation_kwargs[name].update(kwargs)
        return evaluation_kwargs
    
    # Below code only runs if only_standard is False
    if task_name != "linear_regression":
        if task_name in ["relu_2nn_regression"]:
            evaluation_kwargs["linear_regression"] = {"task_name": "linear_regression"}
        for name, kwargs in evaluation_kwargs.items():
            # allow kwargs to override base_kwargs values
            evaluation_kwargs[name] = base_kwargs.copy()
            evaluation_kwargs[name].update(kwargs)
        return evaluation_kwargs

    for strategy in [
        "random_quadrants",
        "orthogonal_train_test",
        "overlapping_train_test",
    ]:
        evaluation_kwargs[strategy] = {"prompting_strategy": strategy}

    for method in ["half_subspace", "skewed"]:
        if "subspace" in method:
            eigenvals = torch.zeros(n_dims)
            eigenvals[: n_dims // 2] = 1
        else:
            eigenvals = 1 / (torch.arange(n_dims) + 1)

        scale = sample_transformation(eigenvals, normalize=True)
        evaluation_kwargs[f"{method}"] = {
            "data_sampler_kwargs": {"scale": scale},
        }

    for dim in ["x", "y"]:
        for scale in [0.333, 0.5, 2, 3]:
            if dim == "x":
                eigenvals = scale * torch.ones(n_dims)
                t = sample_transformation(eigenvals)
                scaling_args = {"data_sampler_kwargs": {"scale": t}}
            else:
                eigenvals = scale * torch.ones(n_dims)
                scaling_args = {"task_sampler_kwargs": {"scale": scale}}

            evaluation_kwargs[f"scale-{dim}={scale}"] = scaling_args

    """
    evaluation_kwargs[f"noisyLR"] = {
        "task_sampler_kwargs": {"renormalize_ys": True, "noise_std": 1},
        "task_name": "noisy_linear_regression",
    }
    """

    for name, kwargs in evaluation_kwargs.items():
        # allow kwargs to override base_kwargs values
        evaluation_kwargs[name] = base_kwargs.copy()
        evaluation_kwargs[name].update(kwargs)

    return evaluation_kwargs


def compute_evals(all_models, evaluation_kwargs, save_path=None, recompute=False):
    try:
        with open(save_path) as fp:
            all_metrics = json.load(fp)
    except Exception:
        all_metrics = {}

    for eval_name, kwargs in tqdm(evaluation_kwargs.items()):
        metrics = {}
        if eval_name in all_metrics and not recompute:
            metrics = all_metrics[eval_name]
        for model in all_models:
            if model.name in metrics and not recompute:
                continue

            metrics[model.name] = eval_model(model, **kwargs)
        all_metrics[eval_name] = metrics

    if save_path is not None:
        with open(save_path, "w") as fp:
            json.dump(all_metrics, fp, indent=2)

    return all_metrics


def get_run_metrics(
    run_path, step=-1, cache=True, skip_model_load=False, skip_baselines=False, only_standard=True
):
    """
    Get evaluation metrics for a run.
    
    Args:
        run_path: Path to the run directory
        step: Checkpoint step to load (-1 for latest)
        cache: Whether to use cached metrics
        skip_model_load: If True, skip loading the model
        skip_baselines: If True, skip baseline models
        only_standard: If True, only evaluate 'standard' strategy (default: True)
    """
    if skip_model_load:
        _, conf = get_model_from_run(run_path, only_conf=True)
        all_models = []
    else:
        model, conf = get_model_from_run(run_path, step)
        model = model.cuda().eval()
        all_models = [model]
        if not skip_baselines:
            # Check if model has get_relevant_baselines method
            if hasattr(model, 'get_relevant_baselines'):
                all_models += model.get_relevant_baselines(conf.training.task)
            else:
                # If method doesn't exist, skip baselines (for custom models)
                pass
    evaluation_kwargs = build_evals(conf, only_standard=only_standard)

    if not cache:
        save_path = None
    elif step == -1:
        save_path = os.path.join(run_path, "metrics.json")
    else:
        save_path = os.path.join(run_path, f"metrics_{step}.json")

    recompute = False
    if save_path is not None and os.path.exists(save_path):
        checkpoint_created = os.path.getmtime(run_path)
        cache_created = os.path.getmtime(save_path)
        if checkpoint_created > cache_created:
            recompute = True

    all_metrics = compute_evals(all_models, evaluation_kwargs, save_path, recompute)
    return all_metrics


def conf_to_model_name(conf):
    if conf.model.family == "gpt2":
        return {
            (3, 2): "Transformer-xs",
            (6, 4): "Transformer-small",
            (12, 8): "Transformer",
        }[(conf.model.n_layer, conf.model.n_head)]
    else:
        return conf.wandb.name


def baseline_names(name):
    if "OLS" in name:
        return "Least Squares"
    if name == "averaging":
        return "Averaging"
    if "NN" in name:
        k = name.split("_")[1].split("=")[1]
        return f"{k}-Nearest Neighbors"
    if "lasso" in name:
        alpha = name.split("_")[1].split("=")[1]
        return f"Lasso (alpha={alpha})"
    if "gd" in name:
        return "2-layer NN, GD"
    if "decision_tree" in name:
        return "Greedy Tree Learning"
    if "xgboost" in name:
        return "XGBoost"
    return name


def read_run_dir(run_dir):
    all_runs = {}
    for task in os.listdir(run_dir):
        task_dir = os.path.join(run_dir, task)
        for run_id in os.listdir(task_dir):
            run_path = os.path.join(task_dir, run_id)
            _, conf = get_model_from_run(run_path, only_conf=True)
            params = {}
            params["run_id"] = run_id
            params["task"] = task
            params["model"] = conf_to_model_name(conf)
            params["kwargs"] = "_".join(
                f"{k}={v}" for k, v in conf.training.task_kwargs.items()
            )
            num_tasks = (
                conf.training.num_tasks if "num_tasks" in conf.training else None
            )
            params["num_tasks"] = num_tasks if num_tasks is not None else -1
            num_examples = (
                conf.training.num_training_examples
                if "num_training_examples" in conf.training
                else None
            )
            params["num_examples"] = num_examples if num_examples is not None else -1
            params["n_dims"] = conf.model.n_dims
            params["n_layer"] = conf.model.n_layer
            params["n_head"] = conf.model.n_head
            params["run_name"] = conf.wandb.name

            for k, v in params.items():
                if k not in all_runs:
                    all_runs[k] = []
                all_runs[k].append(v)

    df = pd.DataFrame(all_runs).sort_values("run_name")
    assert len(df) == len(df.run_name.unique())
    return df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/eval.py <run_dir or run_id> [--all-strategies]")
        print("  --all-strategies: Evaluate all strategies (default: only 'standard')")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    
    # Check if --all-strategies flag is provided
    only_standard = True  # Default: only evaluate standard strategy
    if len(sys.argv) > 2:
        if "--all-strategies" in sys.argv:
            only_standard = False
        elif "--only-standard" in sys.argv:
            only_standard = True
    
    # Also check environment variable
    if os.environ.get("EVAL_ALL_STRATEGIES", "").lower() in ("1", "true", "yes"):
        only_standard = False
    elif os.environ.get("EVAL_ONLY_STANDARD", "").lower() in ("1", "true", "yes"):
        only_standard = True
    
    if only_standard:
        print("Evaluation mode: Only 'standard' strategy (use --all-strategies to evaluate all)")
    else:
        print("Evaluation mode: All strategies")
    
    # Normalize path (handle relative paths)
    if not os.path.isabs(run_dir):
        run_dir = os.path.abspath(run_dir)
    
    # Check if the input path is a single run directory (contains config.yaml)
    config_path = os.path.join(run_dir, "config.yaml")
    
    # Check if it's a single run directory
    if os.path.isfile(config_path):
        # Single run directory - evaluate directly
        print(f"Evaluating run: {run_dir}")
        metrics = get_run_metrics(run_dir, only_standard=only_standard)
    elif os.path.isdir(run_dir):
        # Directory containing multiple task directories or run directories
        # First check if it contains run directories directly (with config.yaml)
        has_run_dirs = False
        for item in os.listdir(run_dir):
            item_path = os.path.join(run_dir, item)
            if os.path.isdir(item_path):
                item_config = os.path.join(item_path, "config.yaml")
                if os.path.isfile(item_config):
                    # This is a run directory
                    has_run_dirs = True
                    print(f"Evaluating run: {item_path}")
                    metrics = get_run_metrics(item_path, only_standard=only_standard)
        
        if not has_run_dirs:
            # Directory containing multiple task directories
            for task in os.listdir(run_dir):
                task_dir = os.path.join(run_dir, task)
                # Skip if not a directory
                if not os.path.isdir(task_dir):
                    continue
                print(f"Evaluating task {task}")
                for run_id in tqdm(os.listdir(task_dir)):
                    run_path = os.path.join(run_dir, task, run_id)
                    # Skip if not a directory
                    if not os.path.isdir(run_path):
                        continue
                    metrics = get_run_metrics(run_path, only_standard=only_standard)
    else:
        print(f"Error: {run_dir} is not a valid directory or file")
        sys.exit(1)
