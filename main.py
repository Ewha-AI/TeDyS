# main.py
from __future__ import annotations

import os, sys, random
import inspect
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs import parse_args, parse_aliases
from loss_metrics import make_ce_loss
from engine import train_one_epoch, validate_epoch

from dataset.oasis2 import OASISLongitudinalMultiScaleFeatureDataset
from models.model_ar_qallv2 import build_ar_qall_tokenizer_model

try:
    GradScaler = torch.amp.GradScaler
except AttributeError:
    from torch.cuda.amp import GradScaler


def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _pad_time_first_dim(x: torch.Tensor, TT: int) -> torch.Tensor:
    T = x.shape[0]
    if T == TT:
        return x
    if T < TT:
        repeat_shape = [TT - T] + [1] * (x.ndim - 1)
        last = x[-1:].repeat(*repeat_shape)
        return torch.cat([x, last], dim=0)
    return x[:TT]


def _filter_kwargs(fn_or_cls: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        sig = inspect.signature(fn_or_cls)
        allowed = set(sig.parameters.keys())
    except Exception:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in allowed}


def collate_feature(batch):
    t_lens = [it["x"].shape[0] for it in batch]
    max_T = max(t_lens)

    Xs, XTXs, XTIMEs, Ys, Metas = [], [], [], [], []

    for it in batch:
        Xs.append(_pad_time_first_dim(it["x"], max_T))
        XTXs.append(_pad_time_first_dim(it["x_tx"], max_T))
        XTIMEs.append(_pad_time_first_dim(it["x_time"], max_T))
        Ys.append(torch.tensor(it["y"], dtype=torch.long))
        Metas.append(it["meta"])

    return dict(
        x=torch.stack(Xs, 0),
        x_tx=torch.stack(XTXs, 0),
        x_time=torch.stack(XTIMEs, 0),
        t_len=torch.tensor(t_lens, dtype=torch.long),
        y=torch.stack(Ys, 0),
        meta=Metas,
    )


def build_loaders(args):
    MsDs = OASISLongitudinalMultiScaleFeatureDataset

    label_col = str(getattr(args, "label_col", "")).strip() or "ttpd_class_6_12_18"

    common = dict(
        csv_path=args.data_csv,
        mode=None,
        fold=args.fold,
        label_col=label_col,
        ct_path_template=getattr(args, "ct_path_template", None),
        path_index_csv=getattr(args, "path_index_csv", None),
        feature_root=getattr(args, "feature_root", None),
        src_root=getattr(args, "src_root", None),
        feature_suffix=getattr(args, "feature_suffix", "_feature"),
        chemo_vocab_path=os.path.join(args.outdir, "chemo_vocab.json"),
        chemo_aliases=parse_aliases(getattr(args, "chemo_aliases", "")),
        include_ccrt=not getattr(args, "exclude_ccrt", False),
        time_divisor=365.0,
    )

    if not getattr(args, "use_multiscale_feature", False):
        raise RuntimeError("use_multiscale_feature=False path is not supported in this clean public version.")

    ms_scales_str = getattr(args, "ms_scales", "res4,res5")
    ms_scales = tuple([s.strip() for s in ms_scales_str.split(",") if s.strip()])

    def _norm_keep_key(v):
        if v is None:
            return None
        if isinstance(v, str):
            s = v.strip()
            if s == "" or s.lower() in ("none", "null", "off", "false"):
                return None
            return s
        return str(v)

    ms_keep_key = _norm_keep_key(getattr(args, "ms_keep_key", None))

    base_ms = dict(
        csv_path=args.data_csv,
        fold=args.fold,
        label_col=common["label_col"],
        ct_path_template=common["ct_path_template"],
        path_index_csv=common["path_index_csv"],
        feature_root=common["feature_root"],
        ms_suffix=getattr(args, "ms_suffix", ".npz"),
        ms_scales=ms_scales,
        ms_pool=getattr(args, "ms_pool", "avg"),
        ms_keep_spatial=bool(getattr(args, "ms_keep_spatial", False)),
        ms_keep_key=ms_keep_key,
        chemo_vocab_path=common["chemo_vocab_path"],
        chemo_aliases=common["chemo_aliases"],
        include_ccrt=common["include_ccrt"],
        time_divisor=common["time_divisor"],
    )

    ds_train = MsDs(**_filter_kwargs(MsDs, {**base_ms, "mode": "train"}))
    ds_val   = MsDs(**_filter_kwargs(MsDs, {**base_ms, "mode": "val"}))

    train_loader = DataLoader(
        ds_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_feature,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=args.batch_size_eval,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_feature,
    )
    return train_loader, val_loader


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = build_loaders(args)

    b0 = next(iter(train_loader))
    print("x:", b0["x"].shape)

    if ("x_time" not in b0) or (not isinstance(b0["x_time"], torch.Tensor)) or (b0["x_time"].ndim != 3) or (b0["x_time"].shape[-1] != 2):
        raise RuntimeError(
            "This QALL design requires x_time=(B,T,2) where [:,:,1] is delta-to-prev.\n"
            f"Got x_time type={type(b0.get('x_time'))} shape={getattr(b0.get('x_time'), 'shape', None)}"
        )

    x = b0.get("x", None)
    if not isinstance(x, torch.Tensor):
        raise RuntimeError(f"Expected x to be a torch.Tensor. Got {type(x)}")
    if x.ndim == 6:
        raise RuntimeError(
            "QALLv2 does not support spatial-kept x=(B,T,C,D,H,W) directly.\n"
            "Set --ms_keep_spatial False (recommended for QALLv2), so x becomes (B,T,F).\n"
            "If you truly need spatial, implement x_pyr=(B,T,S,F) and feed it to the model."
        )
    if x.ndim != 3:
        raise RuntimeError(f"Expected x to be (B,T,F). Got shape={tuple(x.shape)}")

    in_dim = int(b0["x"].shape[-1])
    Vdim = int(b0["x_tx"].shape[-1]) if ("x_tx" in b0 and b0["x_tx"] is not None) else 0

    use_time_embed = bool(getattr(args, "qall_use_time_embed", False))
    dt_hidden = int(getattr(args, "qall_dt_hidden", 128))

    # STD/LTD only
    std_ot_eps = float(getattr(args, "std_ot_eps", 0.1))
    std_ot_iters = int(getattr(args, "std_ot_iters", 20))
    ltd_ode_width = int(getattr(args, "ltd_ode_width", 256))
    ltd_ode_steps = int(getattr(args, "ltd_ode_steps", 4))

    use_ar_query = bool(getattr(args, "qall_use_ar_query", False))
    ar_dt_mode = str(getattr(args, "qall_ar_dt_mode", "add"))

    ta_heads = int(getattr(args, "qall_ta_heads", 4))
    ta_pool_tokens = int(getattr(args, "qall_ta_pool_tokens", 1))

    use_slot_kv = bool(getattr(args, "qall_use_slot", False))

    model_kwargs = dict(
        num_classes=int(args.num_classes),
        in_dim=in_dim,
        d_model=int(getattr(args, "qall_d_model", 256)),
        depth=int(getattr(args, "qall_depth", 2)),
        n_heads=int(getattr(args, "qall_heads", 4)),
        mlp_ratio=float(getattr(args, "qall_mlp_ratio", 4.0)),
        dropout=float(getattr(args, "qall_dropout", 0.0)),
        num_query_tokens=int(getattr(args, "qall_num_query_tokens", 6)),

        use_pos_time_embed=use_time_embed,
        dt_hidden=dt_hidden,

        # STD/LTD only
        std_ot_eps=std_ot_eps,
        std_ot_iters=std_ot_iters,
        ltd_ode_width=ltd_ode_width,
        ltd_ode_steps=ltd_ode_steps,

        # tx
        use_tx=not bool(getattr(args, "ar_no_tx", False)),
        tx_dim=int(Vdim),
        tx_proj=int(getattr(args, "qall_tx_proj", 128)),

        # AR
        use_ar_query=use_ar_query,
        ar_dt_mode=ar_dt_mode,

        # TA (non-AR)
        ta_heads=ta_heads,
        ta_pool_tokens=ta_pool_tokens,

        # slot
        use_slot_kv=use_slot_kv,
        slot_num=int(getattr(args, "qall_slot_num", 6)),
        slot_iters=int(getattr(args, "qall_slot_iters", 3)),
        slot_heads=int(getattr(args, "qall_slot_heads", 4)),  # FIX: correct arg name
    )

    sig = inspect.signature(build_ar_qall_tokenizer_model)
    allowed = set(sig.parameters.keys())
    model_kwargs = {k: v for k, v in model_kwargs.items() if k in allowed}

    model = build_ar_qall_tokenizer_model(**model_kwargs).to(device)

    print("[info] model = QALLv2 (STD=OT, LTD=ODE)")
    print(f"[info] in_dim(F)={in_dim}, tx_dim(V)={Vdim}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    use_amp = bool(getattr(args, "amp", False) and device.type == "cuda")
    scaler = GradScaler(enabled=use_amp) if device.type == "cuda" else None
    loss_fn = make_ce_loss(class_weights=None)

    ckpt_dir = os.path.join(args.outdir, f"fold{args.fold}")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_last = os.path.join(ckpt_dir, "last.pth")

    for epoch in range(1, int(args.epochs) + 1):
        train_one_epoch(
            model, optimizer, loss_fn, train_loader, device, epoch,
            scaler=scaler, scheduler=None, amp=use_amp,
            sampler=None,
            print_fn=print, progress=True,
            args=args,
        )

        validate_epoch(
            model, loss_fn, val_loader, device, epoch,
            print_fn=print, progress=True,
        )

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "scaler": (scaler.state_dict() if scaler is not None else None),
        }, ckpt_last)

    print("[done] training finished")


if __name__ == "__main__":
    main()
