# dataset/oasis2.py
from __future__ import annotations

import os, re, json, glob, warnings, math
from typing import Any, Dict, List, Optional, Tuple, Sequence, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def _as_1d_float(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 0:
        x = x.reshape(1)
    elif x.ndim > 1:
        x = x.reshape(-1)
    x = x.astype(np.float32)
    x = np.nan_to_num(x, copy=False)
    return x

def _one_hot(idx: int, K: int) -> np.ndarray:
    y = np.zeros((K,), dtype=np.float32)
    if 0 <= idx < K:
        y[idx] = 1.0
    return y

def _normalize_mask_cls(arr: np.ndarray, K: int) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 0:
        return _one_hot(int(a.item()), K)
    a = a.reshape(-1)
    if a.size == 1:
        return _one_hot(int(a[0]), K)
    if a.size == K:
        v = a.astype(np.float32)
        s = float(v.sum())
        if s > 0:
            v = v / s
        return np.nan_to_num(v, copy=False).astype(np.float32)
    v = a.astype(np.float32)
    if v.size < K:
        v = np.pad(v, (0, K - v.size))
    else:
        v = v[:K]
    s = float(v.sum())
    if s > 0:
        v = v / s
    return np.nan_to_num(v, copy=False).astype(np.float32)

_TOKEN_SPLIT_BAR = re.compile(r"\s*\|\s*")
_TOKEN_SPLIT_AND = re.compile(r"\s*&\s*")
_TOKEN_HASHNUM   = re.compile(r"\s#\d+\s*$", flags=re.IGNORECASE)

def _clean_token(tok: str) -> Optional[str]:
    if tok is None:
        return None
    t = str(tok).strip()
    if not t or t.lower() == "nan":
        return None
    if t.lower() == "pre":
        return None
    t = _TOKEN_HASHNUM.sub("", t).strip()
    return t or None

def _explode_tx_string(tx_field: str) -> List[List[str]]:
    out: List[List[str]] = []
    if not isinstance(tx_field, str) or not tx_field.strip():
        return out
    per_tp = _TOKEN_SPLIT_BAR.split(tx_field.strip())
    for cell in per_tp:
        if not cell:
            out.append([])
            continue
        toks = []
        for p in _TOKEN_SPLIT_AND.split(cell):
            ct = _clean_token(p)
            if ct:
                toks.append(ct)
        out.append(toks)
    return out

def _apply_aliases(tokens: List[str], chemo_aliases: Dict[str, str]) -> List[str]:
    return [chemo_aliases.get(t, t) for t in tokens]

def _build_or_load_vocab(
    df_all: pd.DataFrame,
    chemo_vocab_path: Optional[str],
    chemo_aliases: Optional[Dict[str, str]] = None
) -> List[str]:
    if chemo_vocab_path and os.path.exists(chemo_vocab_path):
        with open(chemo_vocab_path, "r", encoding="utf-8") as f:
            return list(json.load(f))
    vocab = set()
    for s in df_all["included_ct_tx"].fillna("").tolist():
        for toks in _explode_tx_string(s):
            for t in _apply_aliases(toks, chemo_aliases or {}):
                vocab.add(t)
    vocab = sorted(vocab)
    if chemo_vocab_path:
        os.makedirs(os.path.dirname(chemo_vocab_path), exist_ok=True)
        with open(chemo_vocab_path, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False, indent=2)
    return vocab

def _load_path_index(path_index_csv: str) -> Dict[Tuple[str, str], Dict[str, str]]:
    idx: Dict[Tuple[str, str], Dict[str, str]] = {}
    df = pd.read_csv(path_index_csv)
    need = {"patient_id", "st", "ct_path"}
    if not need.issubset(df.columns):
        raise ValueError(f"path_index_csv must contain {need}, got {list(df.columns)}")
    for _, r in df.iterrows():
        idx[(str(r["patient_id"]), str(r["st"]))] = {"ct_path": str(r["ct_path"])}
    return idx

def _pick_candidate(cands: List[str], date_hint: Optional[str]) -> Optional[str]:
    if not cands:
        return None
    if date_hint:
        d8 = re.sub(r"[^0-9]", "", date_hint)[:8]
        hits = [p for p in cands if d8 in os.path.basename(p)]
        if hits:
            cands = hits
    return sorted(cands)[-1]

def _format_or_glob_path(template: str, patient_id: str, st: str, date_str: Optional[str]) -> str:
    base = template.format(patient_id=patient_id, st=st, date=(date_str or ""))
    if os.path.exists(base):
        return base
    d, b = os.path.dirname(base), os.path.basename(base)
    if b.endswith(".nii.gz"):
        root, ext = b[:-7], ".nii.gz"
    else:
        root, ext = os.path.splitext(b)
    cands = glob.glob(os.path.join(d, f"{root}*{ext}"))
    return _pick_candidate(cands, date_str) or base

def _is_unused_template(tpl: Optional[str]) -> bool:
    if tpl is None:
        return False
    t = str(tpl).strip().lower()
    return (t == "unused") or t.startswith("unused/") or t.startswith("unused\\")

def _resolve_ct_paths(
    patient_id: str,
    st_list: List[str],
    ct_path_template: Optional[str],
    path_index: Optional[Dict[Tuple[str, str], Dict[str, str]]] = None,
    date_list: Optional[List[str]] = None
) -> List[str]:
    ct_paths = []
    for i, st in enumerate(st_list):
        if path_index is not None:
            rec = path_index.get((patient_id, st))
            if rec is None:
                raise FileNotFoundError(f"(pid={patient_id}, st={st}) not in path_index")
            ct_paths.append(rec["ct_path"])
        else:
            if ct_path_template is None:
                raise ValueError("ct_path_template or path_index_csv is required")
            date_str = (date_list[i] if (date_list and i < len(date_list)) else None)
            p = _format_or_glob_path(ct_path_template, patient_id, st, date_str)
            if (not _is_unused_template(ct_path_template)) and (not os.path.exists(p)):
                raise FileNotFoundError(f"CT not found: pid={patient_id}, st={st}, expect='{p}'")
            ct_paths.append(p)
    return ct_paths

def _parse_date(s: str) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(s).normalize()
    except Exception:
        return None

def _dates_to_time_feats(date_list: Optional[List[str]], T: int, divisor: float = 365.0) -> np.ndarray:
    dates: List[Optional[pd.Timestamp]] = []
    if date_list:
        for d in date_list[:T]:
            dates.append(_parse_date(d))
    while len(dates) < T:
        dates.append(None)

    if dates and dates[0] is not None:
        t0 = dates[0]
        days = []
        for i in range(T):
            di = dates[i]
            if di is None:
                gap = (days[i-1] - (days[i-2] if i-2 >= 0 else 0)) if i > 0 else 0
                days.append(days[i-1] + (gap if gap > 0 else 30) if i > 0 else 0)
            else:
                days.append(int((di - t0).days))
    else:
        days = [i * 30 for i in range(T)]

    deltas = [0] + [max(0, days[i] - days[i-1]) for i in range(1, T)]
    div = float(divisor) if divisor and divisor > 0 else 1.0
    return np.stack([np.array(days) / div, np.array(deltas) / div], axis=1).astype(np.float32)

def _strip_nii_gz(name: str) -> str:
    return name[:-7] if name.endswith(".nii.gz") else os.path.splitext(name)[0]

def _ct_to_ms_feature_path(
    ct_path: str,
    feature_root: Optional[str],
    ms_suffix: str = "_nnunet_ms.npz",
) -> str:
    base = os.path.basename(ct_path)
    stem = _strip_nii_gz(base)
    if stem.endswith("_0000"):
        stem = stem[:-5]
    feat_name = stem + ms_suffix
    if feature_root:
        return os.path.join(feature_root, feat_name)
    return os.path.join(os.path.dirname(ct_path), feat_name)

def _load_ms_npz(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"multiscale feature file not found: {path}")
    out: Dict[str, np.ndarray] = {}
    with np.load(path) as f:
        for k in f.files:
            out[k] = np.asarray(f[k])
    return out

def _pool_compat(feat: np.ndarray, mode: str = "avg") -> np.ndarray:
    feat = np.asarray(feat)
    if feat.ndim == 4:
        f = feat.astype(np.float32)
        if mode == "avg":
            v = f.mean(axis=(1,2,3))
        elif mode == "max":
            v = f.max(axis=(1,2,3))
        elif mode == "avgmax":
            v = np.concatenate([f.mean(axis=(1,2,3)), f.max(axis=(1,2,3))], axis=0)
        else:
            raise ValueError(f"Unknown pool mode: {mode}")
        return np.nan_to_num(v, copy=False).astype(np.float32)
    if feat.ndim == 1:
        v = feat.astype(np.float32)
        return np.nan_to_num(v, copy=False).astype(np.float32)
    raise ValueError(f"Expected (C,D,H,W) or (C,), got shape={feat.shape}")

def _select_and_vectorize_ms(
    ms: Dict[str, np.ndarray],
    scales: Sequence[str],
    pool: str = "avg",
) -> np.ndarray:
    vecs: List[np.ndarray] = []
    for s in scales:
        if s not in ms:
            raise KeyError(f"Key '{s}' not found in ms npz. available={sorted(ms.keys())}")
        v = _pool_compat(ms[s], mode=pool)
        vecs.append(v)
    return np.concatenate(vecs, axis=0).astype(np.float32)

def _ms_scale_to_tokens_gridpool(
    feat_any: np.ndarray,
    grid: Tuple[int, int, int] = (2, 2, 2),
    pool: Literal["avg", "max"] = "avg",
) -> torch.Tensor:
    feat_any = np.asarray(feat_any)
    gd, gh, gw = (int(grid[0]), int(grid[1]), int(grid[2]))
    N = gd * gh * gw

    if feat_any.ndim == 4:
        x = torch.from_numpy(np.nan_to_num(feat_any.astype(np.float32), copy=False)).float()
        x = x.unsqueeze(0)
        if pool == "avg":
            y = F.adaptive_avg_pool3d(x, output_size=(gd, gh, gw))
        elif pool == "max":
            y = F.adaptive_max_pool3d(x, output_size=(gd, gh, gw))
        else:
            raise ValueError(f"gridpool pool must be avg|max, got {pool}")
        y = y.squeeze(0).permute(1, 2, 3, 0).contiguous()
        tok = y.view(N, y.shape[-1])
        return tok

    if feat_any.ndim == 1:
        v = torch.from_numpy(np.nan_to_num(feat_any.astype(np.float32), copy=False)).float()
        return v.unsqueeze(0).repeat(N, 1)

    raise ValueError(f"Expected (C,D,H,W) or (C,), got {feat_any.shape}")

def _ms_scale_to_tokens_flatten(feat_any: np.ndarray) -> torch.Tensor:
    feat_any = np.asarray(feat_any)
    if feat_any.ndim == 4:
        x = torch.from_numpy(np.nan_to_num(feat_any.astype(np.float32), copy=False)).float()
        C, D, H, W = x.shape
        return x.permute(1, 2, 3, 0).contiguous().view(D * H * W, C)
    if feat_any.ndim == 1:
        v = torch.from_numpy(np.nan_to_num(feat_any.astype(np.float32), copy=False)).float()
        return v.unsqueeze(0)
    raise ValueError(f"Expected (C,D,H,W) or (C,), got {feat_any.shape}")

def _ct_to_cls6_token_path(
    ct_path: str,
    cls6_cache_root: str,
    cls6_suffix: str = "_cls6_tok.npz",
) -> str:
    base = os.path.basename(ct_path)
    stem = _strip_nii_gz(base)
    if stem.endswith("_0000"):
        stem = stem[:-5]
    cand1 = os.path.join(cls6_cache_root, stem + cls6_suffix)
    stem2 = re.sub(r"_[0-9]{8}$", "", stem)
    cand2 = os.path.join(cls6_cache_root, stem2 + cls6_suffix)
    if os.path.exists(cand1):
        return cand1
    if os.path.exists(cand2):
        return cand2
    return cand1

def _as_float(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr).astype(np.float32)
    return np.nan_to_num(a, copy=False)

def _ensure_KD(arr: np.ndarray, K: int) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2 and a.shape[0] == K:
        return _as_float(a)
    if a.ndim == 1 and a.size == K:
        return _as_float(a[:, None])
    v = a.reshape(-1)
    if v.size % K == 0:
        D = v.size // K
        return _as_float(v.reshape(K, D))
    D = int(math.ceil(v.size / K))
    tgt = K * D
    if v.size < tgt:
        v = np.pad(v, (0, tgt - v.size))
    else:
        v = v[:tgt]
    return _as_float(v.reshape(K, D))

def _ensure_K(arr: np.ndarray, K: int) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 0:
        return _normalize_mask_cls(a, K)
    v = a.reshape(-1)
    if v.size == K:
        vv = v.astype(np.float32)
        s = float(vv.sum())
        if s > 0:
            vv = vv / s
        return np.nan_to_num(vv, copy=False)
    return _normalize_mask_cls(v, K)


class OASISLongitudinalMultiScaleFeatureDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        mode: str,
        fold: Optional[int],
        label_col: str = "prog_event",

        ct_path_template: Optional[str] = None,
        path_index_csv: Optional[str] = None,

        feature_root: Optional[str] = None,
        ms_suffix: str = "_nnunet_ms.npz",
        ms_scales: Sequence[str] = ("res5",),
        ms_pool: str = "avg",

        ms_pyr_enable: bool = True,
        ms_pyr_scales: Optional[Sequence[str]] = None,
        ms_pyr_mode: Literal["gridpool", "flatten"] = "gridpool",
        ms_pyr_grid: Tuple[int, int, int] = (2, 2, 2),
        ms_pyr_pool: Literal["avg", "max"] = "avg",

        use_extra_tokens: bool = False,
        x6_key: str = "x6",
        stats6_key: str = "stats6",
        maskcls_key: str = "mask_cls",
        maskcls_num_classes: int = 6,
        make_dstats: bool = False,

        cls6_cache_root: Optional[str] = None,
        cls6_suffix: str = "_cls6_tok.npz",
        cls6_required: bool = False,

        token_keep_shape: bool = True,
        token_num_classes: int = 6,

        chemo_vocab_path: Optional[str] = None,
        chemo_aliases: Optional[Dict[str, str]] = None,
        include_ccrt: bool = True,

        time_divisor: float = 365.0,
        direct_feat_naming: bool = False,  # OASIS usually works with ct->stem mapping already
    ) -> None:
        super().__init__()
        assert mode in ("train", "val", "test")
        self.mode = mode
        self.fold = fold
        self.label_col = label_col

        self.ct_path_template = ct_path_template
        self.path_index = _load_path_index(path_index_csv) if path_index_csv else None

        self.feature_root = feature_root
        self.ms_suffix = ms_suffix
        self.ms_scales = tuple(ms_scales)
        self.ms_pool = ms_pool

        self.ms_pyr_enable = bool(ms_pyr_enable)
        self.ms_pyr_scales = tuple(ms_pyr_scales) if ms_pyr_scales is not None else tuple(ms_scales)
        self.ms_pyr_mode = ms_pyr_mode
        self.ms_pyr_grid = (int(ms_pyr_grid[0]), int(ms_pyr_grid[1]), int(ms_pyr_grid[2]))
        self.ms_pyr_pool = ms_pyr_pool

        self.use_extra_tokens = bool(use_extra_tokens)
        self.x6_key = str(x6_key)
        self.stats6_key = str(stats6_key)
        self.maskcls_key = str(maskcls_key)
        self.maskcls_num_classes = int(maskcls_num_classes)
        self.make_dstats = bool(make_dstats)

        self.cls6_cache_root = cls6_cache_root
        self.cls6_suffix = str(cls6_suffix)
        self.cls6_required = bool(cls6_required)

        self.token_keep_shape = bool(token_keep_shape)
        self.token_num_classes = int(token_num_classes)

        self.time_divisor = float(time_divisor)
        self.chemo_aliases = chemo_aliases or {}
        self.include_ccrt = include_ccrt
        self.direct_feat_naming = bool(direct_feat_naming)

        df_all = pd.read_csv(csv_path)

        if "included_ct_tx" not in df_all.columns:
            df_all["included_ct_tx"] = ""

        need_cols = {"patient_id", "included_ct_st_indices", "included_ct_tx", label_col}
        if not need_cols.issubset(df_all.columns):
            raise ValueError(f"CSV must contain {need_cols}, got {list(df_all.columns)}")

        self.chemo_vocab = _build_or_load_vocab(df_all, chemo_vocab_path, self.chemo_aliases)
        if not self.include_ccrt:
            self.chemo_vocab = [t for t in self.chemo_vocab if t != "CCRT"]
        self._tok2idx = {t: i for i, t in enumerate(self.chemo_vocab)}

        if "fold" in df_all.columns and self.fold is not None:
            if self.mode == "train":
                df = df_all[df_all["fold"] != self.fold].copy()
            else:
                df = df_all[df_all["fold"] == self.fold].copy()
        else:
            df = df_all.copy()

        self.samples: List[Dict[str, Any]] = []
        bad = 0

        for _, row in df.iterrows():
            pid = str(row["patient_id"]).strip()

            st_raw = str(row["included_ct_st_indices"]) if not pd.isna(row["included_ct_st_indices"]) else ""
            st_list = [s.strip() for s in st_raw.split(",") if s.strip()]
            if len(st_list) == 0:
                bad += 1
                continue

            date_list = None
            if "included_ct_dates" in df.columns and isinstance(row.get("included_ct_dates", None), str):
                date_list = [s.strip() for s in str(row["included_ct_dates"]).split(",") if s.strip()]

            tx_per_tp = _explode_tx_string(row.get("included_ct_tx", ""))
            if len(tx_per_tp) < len(st_list):
                tx_per_tp += [[] for _ in range(len(st_list) - len(tx_per_tp))]
            elif len(tx_per_tp) > len(st_list):
                tx_per_tp = tx_per_tp[:len(st_list)]
            tx_per_tp = [_apply_aliases(toks, self.chemo_aliases) for toks in tx_per_tp]
            if not self.include_ccrt:
                tx_per_tp = [[t for t in toks if t != "CCRT"] for toks in tx_per_tp]

            y = row.get(self.label_col, None)
            if pd.isna(y):
                bad += 1
                continue
            y = int(y)

            try:
                ct_paths = _resolve_ct_paths(pid, st_list, self.ct_path_template, self.path_index, date_list)
            except Exception as e:
                warnings.warn(f"[oasis_multiscale] resolve CT path failed: pid={pid} err={e}")
                bad += 1
                continue

            if self.direct_feat_naming and self.feature_root:
                feat_paths = [os.path.join(self.feature_root, f"{pid}_{st}{self.ms_suffix}") for st in st_list]
            else:
                feat_paths = [
                    _ct_to_ms_feature_path(p, feature_root=self.feature_root, ms_suffix=self.ms_suffix)
                    for p in ct_paths
                ]

            self.samples.append(dict(
                patient_id=pid,
                st_list=st_list,
                date_list=date_list,
                ct_paths=ct_paths,
                feat_paths=feat_paths,
                tx_tokens_per_tp=tx_per_tp,
                y=y,
            ))

        if bad > 0:
            warnings.warn(f"[oasis_multiscale] skipped {bad} rows (path/label issues)")

    def __len__(self) -> int:
        return len(self.samples)

    def _vectorize_tx(self, tx_per_tp: List[List[str]]) -> np.ndarray:
        T, V = len(tx_per_tp), len(self.chemo_vocab)
        mat = np.zeros((T, V), dtype=np.float32)
        for t, toks in enumerate(tx_per_tp):
            for tok in toks:
                idx = self._tok2idx.get(tok, None)
                if idx is not None:
                    mat[t, idx] = 1.0
        return mat

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]

        feats_vec: List[np.ndarray] = []
        pyr_per_scale: Dict[str, List[torch.Tensor]] = {k: [] for k in self.ms_pyr_scales}

        x6_list: List[np.ndarray] = []
        stats6_list: List[np.ndarray] = []
        maskcls_list: List[np.ndarray] = []

        K = int(self.token_num_classes)

        for p, ct_path in zip(s["feat_paths"], s["ct_paths"]):
            ms = _load_ms_npz(p)

            v = _select_and_vectorize_ms(ms, scales=self.ms_scales, pool=self.ms_pool)
            feats_vec.append(v)

            if self.ms_pyr_enable:
                for sc in self.ms_pyr_scales:
                    if sc not in ms:
                        raise KeyError(f"Key '{sc}' not found in ms npz. available={sorted(ms.keys())}")
                    feat_any = ms[sc]
                    if self.ms_pyr_mode == "gridpool":
                        tok = _ms_scale_to_tokens_gridpool(
                            feat_any, grid=self.ms_pyr_grid, pool=self.ms_pyr_pool
                        )
                    elif self.ms_pyr_mode == "flatten":
                        tok = _ms_scale_to_tokens_flatten(feat_any)
                    else:
                        raise ValueError(f"ms_pyr_mode must be gridpool|flatten, got {self.ms_pyr_mode}")
                    pyr_per_scale[sc].append(tok)

            if self.use_extra_tokens:
                if self.cls6_cache_root is None:
                    raise ValueError("use_extra_tokens=True but cls6_cache_root is None.")

                tok_path = _ct_to_cls6_token_path(
                    ct_path=ct_path,
                    cls6_cache_root=self.cls6_cache_root,
                    cls6_suffix=self.cls6_suffix,
                )

                if not os.path.exists(tok_path):
                    if self.cls6_required:
                        raise FileNotFoundError(f"cls6 token file not found: {tok_path}")
                    x6_list.append(np.zeros((K, 0), dtype=np.float32))
                    stats6_list.append(np.zeros((K, 0), dtype=np.float32))
                    maskcls_list.append(np.zeros((K,), dtype=np.float32))
                    continue

                with np.load(tok_path) as f:
                    if self.x6_key in f.files:
                        raw = f[self.x6_key]
                        x6_list.append(_ensure_KD(raw, K) if self.token_keep_shape else _as_float(raw).reshape(-1))
                    else:
                        x6_list.append(np.zeros((K, 0), dtype=np.float32) if self.token_keep_shape else np.zeros((0,), dtype=np.float32))

                    if self.stats6_key in f.files:
                        raw = f[self.stats6_key]
                        stats6_list.append(_ensure_KD(raw, K) if self.token_keep_shape else _as_float(raw).reshape(-1))
                    else:
                        stats6_list.append(np.zeros((K, 0), dtype=np.float32) if self.token_keep_shape else np.zeros((0,), dtype=np.float32))

                    if self.maskcls_key in f.files:
                        mraw = f[self.maskcls_key]
                        maskcls_list.append(_ensure_K(mraw, K))
                    else:
                        maskcls_list.append(np.zeros((K,), dtype=np.float32))

        Fdim = max(f.shape[0] for f in feats_vec)
        feats_vec = [(f if f.shape[0] == Fdim else np.pad(f, (0, Fdim - f.shape[0]))) for f in feats_vec]
        x = torch.from_numpy(np.stack(feats_vec, axis=0)).float()

        x_tx = torch.from_numpy(self._vectorize_tx(s["tx_tokens_per_tp"])).float()
        x_time = torch.from_numpy(
            _dates_to_time_feats(s.get("date_list", []), T=len(feats_vec), divisor=self.time_divisor)
        ).float()

        out: Dict[str, Any] = dict(x=x, x_tx=x_tx, x_time=x_time, y=int(s["y"]))

        if self.ms_pyr_enable:
            ms_pyr: Dict[str, torch.Tensor] = {}
            for sc, toks_list in pyr_per_scale.items():
                ms_pyr[sc] = torch.stack(toks_list, dim=0).float()
            out["ms_pyr"] = ms_pyr
            out["ms_pyr_meta"] = dict(
                scales=list(self.ms_pyr_scales),
                mode=self.ms_pyr_mode,
                grid=list(self.ms_pyr_grid),
                pool=self.ms_pyr_pool,
            )

        if self.use_extra_tokens:
            T = x.shape[0]
            if self.token_keep_shape:
                def _stack_pad_KD(vs: List[np.ndarray]) -> torch.Tensor:
                    D = max((v.shape[1] for v in vs), default=0)
                    vs2 = []
                    for v in vs:
                        if v.shape[1] == D:
                            vs2.append(v)
                        else:
                            vs2.append(np.pad(v, ((0,0),(0, D - v.shape[1]))))
                    return torch.from_numpy(np.stack(vs2, axis=0)).float()

                x6 = _stack_pad_KD(x6_list)
                stats6 = _stack_pad_KD(stats6_list)
                mask_cls = torch.from_numpy(np.stack(maskcls_list, axis=0)).float()

                out["x6"] = x6
                out["stats6"] = stats6
                out["mask_cls"] = mask_cls

                if self.make_dstats and stats6.numel() > 0:
                    dstats = torch.zeros_like(stats6)
                    if T >= 2:
                        dstats[1:] = stats6[1:] - stats6[:-1]
                    out["dstats6"] = dstats
            else:
                def _stack_pad_1d(vs: List[np.ndarray]) -> torch.Tensor:
                    D = max((v.shape[0] for v in vs), default=0)
                    vs2 = [(v if v.shape[0] == D else np.pad(v, (0, D - v.shape[0]))) for v in vs]
                    return torch.from_numpy(np.stack(vs2, axis=0)).float()

                out["x6"] = _stack_pad_1d([np.asarray(v).reshape(-1) for v in x6_list])
                out["stats6"] = _stack_pad_1d([np.asarray(v).reshape(-1) for v in stats6_list])
                out["mask_cls"] = _stack_pad_1d([np.asarray(v).reshape(-1) for v in maskcls_list])

                if self.make_dstats and out["stats6"].numel() > 0:
                    dstats = torch.zeros_like(out["stats6"])
                    if out["stats6"].shape[0] >= 2:
                        dstats[1:] = out["stats6"][1:] - out["stats6"][:-1]
                    out["dstats6"] = dstats

        meta = dict(
            patient_id=s["patient_id"],
            st_list=s["st_list"],
            date_list=s["date_list"] or [],
            ct_paths=s["ct_paths"],
            feature_paths=s["feat_paths"],
            ms_scales=list(self.ms_scales),
            ms_pool=self.ms_pool,
            ms_pyr_enable=self.ms_pyr_enable,
            ms_pyr_scales=list(self.ms_pyr_scales),
            ms_pyr_mode=self.ms_pyr_mode,
            ms_pyr_grid=list(self.ms_pyr_grid),
            ms_pyr_pool=self.ms_pyr_pool,
            use_extra_tokens=self.use_extra_tokens,
            token_keep_shape=self.token_keep_shape,
            token_num_classes=self.token_num_classes,
            x6_key=self.x6_key,
            stats6_key=self.stats6_key,
            maskcls_key=self.maskcls_key,
            make_dstats=self.make_dstats,
            cls6_cache_root=self.cls6_cache_root,
            cls6_suffix=self.cls6_suffix,
            cls6_required=self.cls6_required,
            direct_feat_naming=self.direct_feat_naming,
        )
        out["meta"] = meta
        return out
