import itertools
from typing import Any, Dict
from omegaconf import DictConfig, OmegaConf
from pathlib import Path

import torch
from tqdm import tqdm

from eval import (
    _evaluate_loader,
    _safe_reset,
    _summarize,
    evaluate_from_config,
    pretty_print_results,
)
from utils import build_model, load_cfg, load_full_dataset, locate, parse_args

BASE_CONFIG = Path("configs/tent.yaml")

PARAM_GRID = {
    "tta.args.optim_lr": [0.001, 0.0001],
    "tta.args.optim_steps": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}


def apply_overrides(cfg: DictConfig, overrides: Dict[str, Any]):
    for key, value in overrides.items():
        OmegaConf.update(cfg, key, value, merge=False)


def evaluate_from_config_val(
    cfg_or_path, corruptions=None, print_results=True, baseline=None
) -> Dict[str, Any]:
    # load cfg
    cfg = load_cfg(cfg_or_path)

    # load full_dataset
    dataset_cls, dataset_args = load_full_dataset(cfg.dataset, {"kind": "exploratory"})
    full_dataset = dataset_cls(**dataset_args)

    # load base model
    model = build_model(cfg)

    # load tta
    tta_cfg = cfg.get("tta")
    tta_args = parse_args(tta_cfg.get("args", {}))
    tta_method_cls = locate(cfg.tta.class_path)
    tta_method = tta_method_cls(model, **tta_args)

    eval_cfg = cfg.get("evaluation", {})
    device = torch.device(
        eval_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    )
    tta_method.to(device)
    model.to(device)

    if corruptions is None:
        corruptions = full_dataset.get_available_corruptions()
    else:
        assert len(set(corruptions)) & len(
            set(full_dataset.get_available_corruptions())
        ) == len(
            set(corruptions)
        ), f"Invalid corruptions specified: {set(corruptions) - set(full_dataset.get_available_corruptions())}"

    results = {}
    for corruption in tqdm(
        corruptions,
        desc=f"Evaluating Corruptions for {baseline or 'method'}",
        leave=False,
    ):
        scenario_dataset = full_dataset.filter_by_corruption(corruption)
        loader = torch.utils.data.DataLoader(
            scenario_dataset, batch_size=eval_cfg.get("batch_size", 128), shuffle=False
        )

        _safe_reset(tta_method)
        scenario_results = _evaluate_loader(
            tta_method,
            loader,
            device,
            eval_cfg.get("max_batches"),
            corruption=corruption,
        )
        results[corruption] = scenario_results

    aggregate = _summarize(results)

    all_res = {"per_scenario": results, "aggregate": aggregate}
    if print_results:
        pretty_print_results(all_res)
    return all_res


if __name__ == "__main__":
    cfg: DictConfig = load_cfg(BASE_CONFIG)

    keys = list(PARAM_GRID.keys())
    combos = list(itertools.product(*(PARAM_GRID[k] for k in keys))) if keys else [()]

    best_score = float("-inf")
    best_cfg: DictConfig | None = None

    print(
        f"Starting grid search over {len(combos)} configurations on exploratory dataset...\n"
    )

    for index, combo in enumerate(combos):
        overrides = dict(zip(keys, combo)) if keys else {}
        current_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        apply_overrides(current_cfg, overrides)

        result = evaluate_from_config_val(current_cfg, print_results=False)
        score = float(result["aggregate"].get("mean_accuracy", 0.0))
        print(f"conf{index}: mean_accuracy={score:.4f}")

        if score > best_score:
            best_score = score
            best_cfg = current_cfg

    print("\nBest Configuration:")
    print(OmegaConf.to_yaml(best_cfg))
    print(f"Best Mean Accuracy: {best_score:.4f}")

    result = evaluate_from_config(best_cfg, print_results=False)
    score = float(result["aggregate"].get("mean_accuracy", 0.0))
    print(f"\nBest Configuration Mean Accuracy on public test bench: {score:.4f}")
