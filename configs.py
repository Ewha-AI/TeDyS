# configs.py
from __future__ import annotations

import argparse
from typing import Dict, Optional


def parse_aliases(s: str) -> Dict[str, str]:
    d: Dict[str, str] = {}
    if not s:
        return d
    for pair in s.split(","):
        if "=" in pair:
            k, v = pair.split("=", 1)
            d[k.strip()] = v.strip()
    return d


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Train classifier on pre-extracted longitudinal features")

    g = ap.add_argument_group("data")
    g.add_argument("--data_csv", type=str, required=None)
    g.add_argument("--fold", type=int, default=0, help="0..4; -1 = run all folds inside main.py")
    g.add_argument("--outdir", type=str, required=True)
    g.add_argument("--ct_path_template", type=str, default=None)
    g.add_argument("--path_index_csv", type=str, default=None)
    g.add_argument("--chemo_aliases", type=str, default="")
    g.add_argument("--exclude_ccrt", action="store_true")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--dataset", type=str, default="oasis", choices=["oasis"])
    g.add_argument("--label_col", type=str, default=None)

    g = ap.add_argument_group("train")
    g.add_argument("--epochs", type=int, default=50)
    g.add_argument("--batch_size", type=int, default=8)
    g.add_argument("--batch_size_eval", type=int, default=8)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--weight_decay", type=float, default=1e-5)
    g.add_argument("--num_workers", type=int, default=8)
    g.add_argument("--amp", action="store_true")
    g.add_argument("--ddp", action="store_true")
    g.add_argument("--local_rank", type=int, default=-1)
    g.add_argument("--evaluate_only", action="store_true")
    g.add_argument("--resume", type=str, default=None)

    g = ap.add_argument_group("viz")
    g.add_argument("--saliency", type=str, default="off", choices=["off", "grad", "ig", "smoothgrad"])
    g.add_argument("--saliency_reps", type=int, default=2)

    g = ap.add_argument_group("multiscale")
    g.add_argument("--use_multiscale_feature", action="store_true")
    g.add_argument("--feature_root", type=str, default=None)
    g.add_argument("--ms_suffix", type=str, default="_nnunet_ms.npz")
    g.add_argument("--ms_scales", type=str, default="res5")
    g.add_argument("--ms_pool", type=str, default="avg", choices=["avg", "max", "avgmax"])

    g = ap.add_argument_group("model")
    g.add_argument("--num_classes", type=int, default=4)
    g.add_argument("--ar_no_tx", action="store_true")

    g = ap.add_argument_group("qall")
    g.add_argument("--qall_d_model", type=int, default=256)
    g.add_argument("--qall_depth", type=int, default=2)
    g.add_argument("--qall_heads", type=int, default=4)
    g.add_argument("--qall_mlp_ratio", type=float, default=4.0)
    g.add_argument("--qall_dropout", type=float, default=0.0)
    g.add_argument("--qall_num_query_tokens", type=int, default=6)

    g.add_argument("--qall_use_time_embed", action="store_true")
    g.add_argument("--qall_dt_hidden", type=int, default=128)

    g.add_argument("--qall_ta_heads", type=int, default=4)
    g.add_argument("--qall_ta_pool_tokens", type=int, default=1)

    g.add_argument("--qall_tx_proj", type=int, default=128)

    g.add_argument("--std_ot_eps", type=float, default=0.1)
    g.add_argument("--std_ot_iters", type=int, default=20)

    g.add_argument("--ltd_ode_width", type=int, default=256)
    g.add_argument("--ltd_ode_steps", type=int, default=4)

    g.add_argument("--qall_use_slot", action="store_true")
    g.add_argument("--qall_slot_num", type=int, default=6)
    g.add_argument("--qall_slot_iters", type=int, default=3)
    g.add_argument("--qall_slot_heads", type=int, default=4)

    g = ap.add_argument_group("qall_ar_query")
    g.add_argument("--qall_use_ar_query", action="store_true")
    g.add_argument("--qall_ar_dt_mode", type=str, default="add", choices=["none", "add", "film"])

    return ap


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = build_parser()
    return ap.parse_args(argv)
