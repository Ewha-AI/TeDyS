# engine.py
from __future__ import annotations

import os
import json
import inspect
from typing import Dict, Any, Optional

import torch
from torch.amp import autocast as amp_autocast
from tqdm.auto import tqdm

from loss_metrics import compute_metrics


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _recursive_to_device(v: Any, device: torch.device):
    if isinstance(v, torch.Tensor):
        return v.to(device, non_blocking=True)
    if isinstance(v, dict):
        return {k: _recursive_to_device(vv, device) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        out = [_recursive_to_device(x, device) for x in v]
        return type(v)(out)
    return v


def _recursive_cast_to_dtype(v: Any, dtype: torch.dtype):
    if isinstance(v, torch.Tensor):
        if v.is_floating_point() and v.dtype != dtype:
            return v.to(dtype=dtype)
        return v
    if isinstance(v, dict):
        return {k: _recursive_cast_to_dtype(vv, dtype) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        out = [_recursive_cast_to_dtype(x, dtype) for x in v]
        return type(v)(out)
    return v


def _call_model_signature_aware(model, batch: Dict[str, Any]) -> torch.Tensor:
    m = _unwrap_model(model)

    try:
        p = next(m.parameters())
        model_dtype = p.dtype
    except StopIteration:
        model_dtype = torch.float32

    try:
        sig = inspect.signature(m.forward)
        param_names = set(sig.parameters.keys())
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    except (ValueError, TypeError):
        param_names = set()
        has_var_kw = True

    alias = {
        "x_blood": "b_seq",
        "t_len_blood": "b_len",
        "t_blen": "b_len",
        "xq": "x_query",
        "xp": "x_pixel",
    }

    norm = dict(batch)
    for k_src, k_dst in alias.items():
        if k_src in norm and k_dst not in norm:
            norm[k_dst] = norm[k_src]

    def _pick_main_x(d):
        for key in ("x", "x6", "stats6", "x_query"):
            if key in d and isinstance(d[key], torch.Tensor):
                return d[key]
        return None

    if has_var_kw or param_names:
        kwargs = {}
        for k, v in norm.items():
            if k == "meta":
                continue
            if has_var_kw or (k in param_names):
                kwargs[k] = v

        if ("x_query" in kwargs) and ("x_pixel" in kwargs):
            if "x" in kwargs and ("x" not in param_names) and (not has_var_kw):
                kwargs.pop("x", None)

        kwargs = {k: _recursive_cast_to_dtype(v, model_dtype) for k, v in kwargs.items()}

        try:
            return m(**kwargs)
        except TypeError:
            x_main = _pick_main_x(norm)
            if x_main is None:
                raise
            x_main = _recursive_cast_to_dtype(x_main, model_dtype)
            try:
                return m(x_main, **{k: v for k, v in kwargs.items() if k != "x"})
            except TypeError:
                return m(x_main)

    x_main = _pick_main_x(norm)
    if x_main is None:
        raise TypeError("Cannot find a suitable main input tensor (x/x6/stats6/x_query) in batch.")
    x_main = _recursive_cast_to_dtype(x_main, model_dtype)
    return m(x_main)


def _batch_size(batch: Dict[str, Any]) -> int:
    if "x" in batch and isinstance(batch["x"], torch.Tensor):
        return int(batch["x"].size(0))
    if "x_query" in batch and isinstance(batch["x_query"], torch.Tensor):
        return int(batch["x_query"].size(0))
    if "y" in batch and isinstance(batch["y"], torch.Tensor):
        return int(batch["y"].size(0))
    raise KeyError("Cannot infer batch size (no x/x_query/y tensor found).")


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {k: _recursive_to_device(v, device) for k, v in batch.items()}


def _gather_logits_targets(logits_list, targets_list):
    return torch.cat(logits_list, dim=0), torch.cat(targets_list, dim=0)
    
def _get_moe_aux(model) -> Optional[torch.Tensor]:
    m = _unwrap_model(model)
    for k in ("last_moe_aux", "moe_aux", "aux_loss", "load_balance_loss"):
        v = getattr(m, k, None)
        if isinstance(v, torch.Tensor):
            return v
    return None

def train_one_epoch(
    model,
    optimizer,
    loss_fn,
    loader,
    device,
    epoch: int,
    *,
    scaler=None,
    scheduler=None,
    log_every: int = 20,
    amp: bool = True,
    sampler=None,
    print_fn=print,
    progress: bool = False,
    args=None,
) -> Dict[str, float]:
    model.train()
    if sampler is not None and hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    loss_meter, n_seen = 0.0, 0
    logits_accum, targets_accum = [], []

    iters = tqdm(loader, total=len(loader), desc=f"Train e{epoch}", dynamic_ncols=True, disable=not progress)
    for it, batch in enumerate(iters):
        batch = _to_device(batch, device)
        y = batch["y"]

        optimizer.zero_grad(set_to_none=True)

        if scaler is None or (not amp):
            logits = _call_model_signature_aware(model, batch)
        else:
            with amp_autocast(device.type, dtype=torch.float16):
                logits = _call_model_signature_aware(model, batch)
            if not torch.isfinite(logits).all():
                logits = _call_model_signature_aware(model, batch)

        if logits.ndim == 3:
            logits = logits.mean(1)

        y = y.to(logits.device)
        if y.ndim > 1:
            if y.shape == logits.shape or y.size(-1) == logits.size(-1):
                y = y.argmax(dim=-1)
            else:
                y = y.view(y.size(0))
        y = y.long()

        main_loss = loss_fn(logits, y)

        aux = _get_moe_aux(model)
        moe_aux_weight = float(getattr(args, "moe_aux_weight", 0.0)) if args is not None else 0.0
        if aux is not None and moe_aux_weight > 0:
            loss = main_loss + moe_aux_weight * aux.to(main_loss.device)
        else:
            loss = main_loss

        if not torch.isfinite(loss):
            raise RuntimeError("non-finite loss detected")

        if scaler is None or (not amp):
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        bs = _batch_size(batch)
        loss_meter += float(loss.item()) * bs
        n_seen += bs

        logits_accum.append(logits.detach().float().cpu())
        targets_accum.append(y.detach().cpu())

        if progress:
            iters.set_postfix(loss=f"{loss.item():.4f}")
        elif (it + 1) % log_every == 0:
            print_fn(f"[train] e{epoch} it {it+1}/{len(loader)} loss={loss.item():.4f}")

        if scheduler is not None and hasattr(scheduler, "step_iter"):
            scheduler.step_iter()

    epoch_loss = loss_meter / max(1, n_seen)
    logits_all, targets_all = _gather_logits_targets(logits_accum, targets_accum)
    metrics = compute_metrics(logits_all, targets_all, average="macro")
    return dict(loss=epoch_loss, **metrics)


@torch.no_grad()
def validate_epoch(
    model,
    loss_fn,
    loader,
    device,
    epoch: int,
    *,
    print_fn=print,
    progress: bool = False,
) -> Dict[str, float]:
    model.eval()

    loss_meter, n_seen = 0.0, 0
    logits_accum, targets_accum = [], []

    iters = tqdm(loader, total=len(loader), desc=f"Valid e{epoch}", dynamic_ncols=True, disable=not progress)
    for batch in iters:
        batch = _to_device(batch, device)
        y = batch["y"]

        logits = _call_model_signature_aware(model, batch)
        if logits.ndim == 3:
            logits = logits.mean(1)

        if y.ndim > 1:
            if y.size(-1) > 1:
                y = torch.argmax(y, dim=1)
            else:
                y = y.view(-1)
        y = y.long().to(logits.device)

        loss = loss_fn(logits, y)

        bs = _batch_size(batch)
        loss_meter += float(loss.item()) * bs
        n_seen += bs

        logits_accum.append(logits.detach().float().cpu())
        targets_accum.append(y.detach().cpu())

        if progress:
            iters.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss = loss_meter / max(1, n_seen)
    logits_all, targets_all = _gather_logits_targets(logits_accum, targets_accum)
    metrics = compute_metrics(logits_all, targets_all, average="macro")

    out = dict(loss=epoch_loss, **metrics)
    print_fn(f"[valid] e{epoch} loss={out['loss']:.4f} acc={out.get('accuracy', float('nan')):.4f} f1={out.get('f1', float('nan')):.4f}")
    return out


def save_json_log(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
