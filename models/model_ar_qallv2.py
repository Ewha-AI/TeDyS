# models/model_ar_qallv2.py
from __future__ import annotations

from typing import Optional, Literal, Any, Tuple

import torch
import torch.nn as nn


def _safe_len_mask(t_len: Optional[torch.Tensor], B: int, T: int, device) -> torch.Tensor:
    """
    return: (B,T) bool, True=valid timestep
    """
    if t_len is None:
        return torch.ones((B, T), device=device, dtype=torch.bool)
    if not isinstance(t_len, torch.Tensor):
        t_len = torch.tensor(t_len, device=device)
    t_len = t_len.to(device)
    idx = torch.arange(T, device=device)[None, :].expand(B, T)
    return idx < t_len[:, None]


# ============================================================
# Time interval embedding
# ============================================================
class TimeDeltaEmbed(nn.Module):
    """Embed scalar delta time -> d_model."""
    def __init__(self, d_model: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, d_model),
        )

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        if dt.dim() == 1:
            dt = dt[:, None]
        dt = torch.log1p(dt.clamp_min(0))
        return self.net(dt)  # (B, d_model)


# =============================================================================
# Slot Attention (optional KV compressor)
# =============================================================================
class SlotAttention(nn.Module):
    """
    inputs: (B, N, D)
    return: slots (B, S, D)
    """
    def __init__(
        self,
        d_model: int,
        num_slots: int = 6,
        iters: int = 3,
        n_heads: int = 4,
        dropout: float = 0.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_slots = int(num_slots)
        self.iters = int(iters)
        self.eps = float(eps)

        self.slot_mu = nn.Parameter(torch.zeros(1, 1, d_model))
        self.slot_logsigma = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.slot_mu, std=0.02)
        nn.init.trunc_normal_(self.slot_logsigma, std=0.02)

        nhead_eff = max(1, min(n_heads, max(1, d_model // 32)))
        self.n_heads = nhead_eff
        self.scale = (d_model // nhead_eff) ** -0.5

        self.norm_in = nn.LayerNorm(d_model)
        self.norm_slots = nn.LayerNorm(d_model)

        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)

        self.gru = nn.GRUCell(d_model, d_model)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,           # (B,N) True valid
        slot_init: Optional[torch.Tensor] = None,      # (B,S,D) optional (AR init)
        slot_init_bias: Optional[torch.Tensor] = None  # (B,1,D) or (B,S,D)
    ) -> torch.Tensor:
        B, N, D = x.shape
        x = self.norm_in(x)

        mu = self.slot_mu.expand(B, self.num_slots, -1)
        sigma = self.slot_logsigma.exp().expand(B, self.num_slots, -1)

        if slot_init is not None:
            slots = slot_init
        elif self.training:
            slots = mu + sigma * torch.randn_like(mu)
        else:
            # Deterministic eval: use the expected slot init (no RNG draw),
            # so val/test metrics don't depend on global RNG state.
            slots = mu

        if slot_init_bias is not None:
            sb = slot_init_bias
            if sb.ndim == 3 and sb.shape[1] == 1:
                sb = sb.expand(B, self.num_slots, -1)
            if sb.ndim == 3 and sb.shape[1] == self.num_slots:
                slots = slots + sb

        k = self.to_k(x)
        v = self.to_v(x)

        H = self.n_heads
        Dh = D // H
        k = k.view(B, N, H, Dh)
        v = v.view(B, N, H, Dh)

        for _ in range(self.iters):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.to_q(slots_norm).view(B, self.num_slots, H, Dh)

            attn_logits = torch.einsum("bshd,bnhd->bhsn", q, k) * self.scale
            if mask is not None:
                neg_inf = torch.finfo(attn_logits.dtype).min
                attn_logits = attn_logits.masked_fill((~mask)[:, None, None, :], neg_inf)

            attn = attn_logits.softmax(dim=-1)
            attn = attn + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bhsn,bnhd->bshd", attn, v).reshape(B, self.num_slots, D)

            slots = self.gru(
                updates.reshape(B * self.num_slots, D),
                slots_prev.reshape(B * self.num_slots, D),
            ).view(B, self.num_slots, D)
            slots = slots + self.mlp(slots)

        return slots


# =============================================================================
# Sinkhorn OT (STD only)
# =============================================================================
class SinkhornOTAlign(nn.Module):
    """
    Batched Sinkhorn OT alignment with barycentric projection (log-domain).
    Xt, Xprev: (B, N, d)  (N must match)
    returns: Xprev aligned to Xt: (B, N, d)
    """
    def __init__(
        self,
        eps: float = 0.1,
        iters: int = 20,
        cost: Literal["l2"] = "l2",
        clamp_logK: float = 50.0,
        tiny: float = 1e-12,
        detach_duals: bool = False,
    ):
        super().__init__()
        self.eps = float(eps)
        self.iters = int(iters)
        self.cost = cost
        self.clamp_logK = float(clamp_logK)
        self.tiny = float(tiny)
        self.detach_duals = bool(detach_duals)

    @staticmethod
    def _pairwise_cost_l2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        a2 = (A * A).sum(dim=-1, keepdim=True)         # (B,N,1)
        b2 = (B * B).sum(dim=-1).unsqueeze(1)          # (B,1,N)
        ab = A @ B.transpose(1, 2)                     # (B,N,N)
        return (a2 + b2 - 2.0 * ab).clamp_min(0)

    def forward(self, Xt: torch.Tensor, Xprev: torch.Tensor) -> torch.Tensor:
        if Xt.shape[:2] != Xprev.shape[:2]:
            raise ValueError(f"SinkhornOTAlign expects same (B,N,*) shapes. Xt={Xt.shape}, Xprev={Xprev.shape}")

        orig_dtype = Xt.dtype
        Xt32 = Xt.float()
        Xp32 = Xprev.float()

        B, N, _ = Xt32.shape

        if self.cost != "l2":
            raise ValueError(f"SinkhornOTAlign(cost={self.cost}) not supported; use cost='l2'")
        C = self._pairwise_cost_l2(Xp32, Xt32)  # (B,N,N)

        eps = max(self.eps, 1e-6)
        logK = (-C / eps).clamp(min=-self.clamp_logK, max=0.0)  # (B,N,N)

        log_a = -torch.log(torch.tensor(float(N), device=logK.device, dtype=logK.dtype))
        log_b = log_a

        f = torch.zeros((B, N), device=logK.device, dtype=logK.dtype)
        g = torch.zeros((B, N), device=logK.device, dtype=logK.dtype)

        for _ in range(self.iters):
            f_new = log_a - torch.logsumexp(logK + g.unsqueeze(1), dim=-1)  # (B,N)
            f = f_new.detach() if self.detach_duals else f_new

            g_new = log_b - torch.logsumexp(logK + f.unsqueeze(2), dim=1)   # (B,N)
            g = g_new.detach() if self.detach_duals else g_new

        logP = logK + f.unsqueeze(2) + g.unsqueeze(1)  # (B,N,N)
        P = torch.exp(logP).clamp_min(self.tiny)        # (B,N,N)

        col_mass = P.sum(dim=1, keepdim=False).unsqueeze(-1).clamp_min(self.tiny)  # (B,N,1)
        Xprev_to_cur = torch.bmm(P.transpose(1, 2), Xp32) / col_mass               # (B,N,d)

        return Xprev_to_cur.to(dtype=orig_dtype)


# =============================================================================
# Global Neural ODE (LTD only)
# =============================================================================
class NeuralODEFunc(nn.Module):
    def __init__(self, d_model: int, t_hidden: int = 128, width: int = 256, dropout: float = 0.0):
        super().__init__()
        self.t_emb = TimeDeltaEmbed(d_model, hidden=t_hidden)
        self.net = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(width, d_model),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        te = self.t_emb(t)
        return self.net(torch.cat([z, te], dim=-1))


class GlobalNeuralODE(nn.Module):
    def __init__(self, d_model: int, t_hidden: int = 128, width: int = 256, n_steps: int = 4, dropout: float = 0.0):
        super().__init__()
        self.f = NeuralODEFunc(d_model, t_hidden=t_hidden, width=width, dropout=dropout)
        self.n_steps = int(n_steps)

    def forward(self, z0: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        B, _ = z0.shape
        tau = tau.clamp_min(0).to(device=z0.device, dtype=z0.dtype)
        n = max(1, self.n_steps)

        z = z0
        t = torch.zeros((B,), device=z0.device, dtype=z0.dtype)
        dt = (tau / float(n))

        for _ in range(n):
            k1 = self.f(z, t)
            k2 = self.f(z + 0.5 * dt[:, None] * k1, t + 0.5 * dt)
            k3 = self.f(z + 0.5 * dt[:, None] * k2, t + 0.5 * dt)
            k4 = self.f(z + dt[:, None] * k3, t + dt)
            z = z + (dt[:, None] / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t + dt
        return z


# =============================================================================
# Compare & Fuse: (Xt, STD, LTD) -> Ct
#   STD = OT aligned adjacent residual
#   LTD = baseline transported by ODE then compared
# =============================================================================
class CompareFuseTokens(nn.Module):
    def __init__(
        self,
        d_model: int,
        dt_hidden: int = 128,
        dropout: float = 0.0,
        # STD (OT)
        std_ot_eps: float = 0.1,
        std_ot_iters: int = 20,
        # LTD (ODE)
        ltd_ode_width: int = 256,
        ltd_ode_steps: int = 4,
    ):
        super().__init__()

        self.dt_emb = TimeDeltaEmbed(d_model, hidden=dt_hidden)
        self.tau_emb = TimeDeltaEmbed(d_model, hidden=dt_hidden)

        gate_in = 5 * d_model  # pXt, pSTD, pLTD, dtE, tauE
        self.gate = nn.Sequential(
            nn.Linear(gate_in, 2 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, 2),
        )
        self.proj = nn.Sequential(
            nn.Linear(2 * d_model, 2 * d_model),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )

        self.ot = SinkhornOTAlign(eps=std_ot_eps, iters=std_ot_iters)
        self.ode = GlobalNeuralODE(d_model, t_hidden=dt_hidden, width=ltd_ode_width, n_steps=ltd_ode_steps, dropout=dropout)
        self.global_token_shift = nn.Linear(d_model, d_model)

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)

    def _std_delta(self, Xt: torch.Tensor, Xprev: torch.Tensor) -> torch.Tensor:
        Xprev_aligned = self.ot(Xt, Xprev)
        return Xt - Xprev_aligned

    def _ltd_delta(self, Xt: torch.Tensor, Xfirst: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        z0 = self._pool(Xfirst)
        z_tau = self.ode(z0, tau)
        dz = (z_tau - z0)
        shift = self.global_token_shift(dz).unsqueeze(1)
        Xpred = Xfirst + shift
        return Xt - Xpred

    def forward(
        self,
        Xt: torch.Tensor,
        Xprev: torch.Tensor,
        Xfirst: torch.Tensor,
        dt: torch.Tensor,
        tau: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dSTD = self._std_delta(Xt, Xprev)
        dLTD = self._ltd_delta(Xt, Xfirst, tau)

        pXt = self._pool(Xt)
        pSTD = self._pool(dSTD)
        pLTD = self._pool(dLTD)

        dtE = self.dt_emb(dt)
        tauE = self.tau_emb(tau)

        w = torch.softmax(self.gate(torch.cat([pXt, pSTD, pLTD, dtE, tauE], dim=-1)), dim=-1)
        wSTD = w[:, 0].view(-1, 1, 1)
        wLTD = w[:, 1].view(-1, 1, 1)

        dMix = wSTD * dSTD + wLTD * dLTD
        Ct = self.proj(torch.cat([Xt, dMix], dim=-1))
        return Ct, w


# =============================================================================
# Compare -> Progress tokens
# =============================================================================
class CompareToProgressTokens(nn.Module):
    """
    Returns:
      ptok: (B,P,d) progression tokens
      w   : (B,2)
    """
    def __init__(
        self,
        d_model: int,
        num_prog_tokens: int = 4,
        dt_hidden: int = 128,
        dropout: float = 0.0,
        # STD (OT)
        std_ot_eps: float = 0.1,
        std_ot_iters: int = 20,
        # LTD (ODE)
        ltd_ode_width: int = 256,
        ltd_ode_steps: int = 4,
    ):
        super().__init__()
        self.P = int(num_prog_tokens)

        self.cmp = CompareFuseTokens(
            d_model=d_model,
            dt_hidden=dt_hidden,
            dropout=dropout,
            std_ot_eps=std_ot_eps,
            std_ot_iters=std_ot_iters,
            ltd_ode_width=ltd_ode_width,
            ltd_ode_steps=ltd_ode_steps,
        )

        self.prog_readout = nn.Parameter(torch.randn(1, self.P, d_model) * 0.02)
        self.prog_attn = nn.MultiheadAttention(d_model, num_heads=4, dropout=dropout, batch_first=True)
        self.prog_ln = nn.LayerNorm(d_model)

    def forward(
        self,
        Xt: torch.Tensor,
        Xprev: torch.Tensor,
        Xfirst: torch.Tensor,
        dt: torch.Tensor,
        tau: torch.Tensor,
        Xt_kpm: Optional[torch.Tensor] = None,  # (B,N) True=PAD
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        Ct, w = self.cmp(Xt=Xt, Xprev=Xprev, Xfirst=Xfirst, dt=dt, tau=tau)

        B = Xt.size(0)
        q = self.prog_readout.expand(B, -1, -1)  # (B,P,d)
        out, _ = self.prog_attn(query=q, key=Ct, value=Ct, key_padding_mask=Xt_kpm, need_weights=False)
        ptok = self.prog_ln(q + out)
        return ptok, w


# =============================================================================
# Q-only update block: Cross-Attn (Q <- mem) + FFN
# =============================================================================
class CrossAttnQBlock(nn.Module):
    """
    Updates q using mem as key/value.
      q   : (B, Kq, d)
      mem : (B, Nm, d)
      mem_kpm: (B, Nm) True=PAD
    """
    def __init__(self, d_model: int, nhead: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_m = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, q: torch.Tensor, mem: torch.Tensor, mem_kpm: Optional[torch.Tensor] = None) -> torch.Tensor:
        qn = self.ln_q(q)
        mn = self.ln_m(mem)
        out, _ = self.mha(query=qn, key=mn, value=mn, key_padding_mask=mem_kpm, need_weights=False)
        q = q + out

        qn = self.ln2(q)
        q = q + self.ffn(qn)
        return q


# =============================================================================
# Temporal Attention Pool (used only when non-AR)
# =============================================================================
class TemporalAttentionPool(nn.Module):
    """
    bank: (B, L, d) where L = T*Kq (or any length)
    returns: (B, d) pooled representation
    """
    def __init__(self, d_model: int, nhead: int = 4, dropout: float = 0.0, num_pool_tokens: int = 1):
        super().__init__()
        self.num_pool_tokens = int(num_pool_tokens)
        self.pool_q = nn.Parameter(torch.randn(1, self.num_pool_tokens, d_model) * 0.02)
        self.mha = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, bank: torch.Tensor, bank_kpm: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = bank.size(0)
        q = self.pool_q.expand(B, -1, -1)  # (B,Qp,d)
        out, _ = self.mha(query=q.to(bank.dtype), key=bank, value=bank, key_padding_mask=bank_kpm, need_weights=False)
        y = self.ln(q.to(bank.dtype) + out)
        return y.mean(dim=1).to(q.dtype)  # (B,d)


# =============================================================================
# Backbone: learned KQ queries per timepoint; KV = spatial(+slot optional) + progression
#   + optional AR query carry with dt update
# =============================================================================
class AR_Qall_Backbone(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int = 256,
        num_query_tokens: int = 6,
        depth: int = 1,
        nhead: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        use_pos_time_embed: bool = True,
        dt_hidden: int = 128,
        # progression tokens
        num_prog_tokens: int = 4,
        # STD (OT only)
        std_ot_eps: float = 0.1,
        std_ot_iters: int = 20,
        # LTD (ODE only)
        ltd_ode_width: int = 256,
        ltd_ode_steps: int = 4,
        # slot
        use_slot_kv: bool = False,
        slot_num: int = 6,
        slot_iters: int = 3,
        slot_heads: int = 4,
        # AR
        use_ar_query: bool = False,
        ar_dt_mode: Literal["none", "add", "film"] = "add",
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.num_query_tokens = int(num_query_tokens)
        self.use_pos_time_embed = bool(use_pos_time_embed)

        self.use_slot_kv = bool(use_slot_kv)
        self.use_ar_query = bool(use_ar_query)
        self.ar_dt_mode: Literal["none", "add", "film"] = ar_dt_mode

        self.x_proj = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, d_model),
        )

        self.q_base = nn.Parameter(torch.randn(self.num_query_tokens, d_model) * 0.02)
        self.dt_pos_emb = TimeDeltaEmbed(d_model, hidden=int(dt_hidden)) if self.use_pos_time_embed else None

        self.slot = (
            SlotAttention(
                d_model=d_model,
                num_slots=int(slot_num),
                iters=int(slot_iters),
                n_heads=int(slot_heads),
                dropout=float(dropout),
            )
            if self.use_slot_kv
            else None
        )

        self.compare_to_prog = CompareToProgressTokens(
            d_model=d_model,
            num_prog_tokens=int(num_prog_tokens),
            dt_hidden=int(dt_hidden),
            dropout=float(dropout),
            std_ot_eps=float(std_ot_eps),
            std_ot_iters=int(std_ot_iters),
            ltd_ode_width=int(ltd_ode_width),
            ltd_ode_steps=int(ltd_ode_steps),
        )

        # AR dt update
        self.ar_dt_emb = TimeDeltaEmbed(d_model, hidden=int(dt_hidden))
        self.ar_film = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.SiLU(),
            nn.Linear(2 * d_model, 2 * d_model),
        )

        self.blocks = nn.ModuleList(
            [CrossAttnQBlock(d_model, nhead=nhead, mlp_ratio=mlp_ratio, dropout=dropout) for _ in range(int(depth))]
        )
        self.out_norm = nn.LayerNorm(d_model)

    def _dt_from_x_time(self, x_time_t: Optional[torch.Tensor], B: int, device) -> torch.Tensor:
        if x_time_t is None:
            return torch.zeros((B,), device=device)
        dt = x_time_t[:, 1]
        if dt.shape[0] != B:
            dt = dt.expand(B)
        return dt

    def _abs_from_x_time(self, x_time_t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if x_time_t is None:
            return None
        return x_time_t[:, 0]

    def _apply_dt_pos(self, tok: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if (not self.use_pos_time_embed) or (self.dt_pos_emb is None):
            return tok
        dtE = self.dt_pos_emb(dt).unsqueeze(1)
        return tok + dtE

    def _ar_dt_update(self, Q: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        if not self.use_ar_query:
            return Q
        if self.ar_dt_mode == "none":
            return Q
        dtE = self.ar_dt_emb(dt).unsqueeze(1).to(Q.dtype)
        if self.ar_dt_mode == "add":
            return Q + dtE
        if self.ar_dt_mode == "film":
            gb = self.ar_film(dtE)
            gamma, beta = gb.chunk(2, dim=-1)
            return (1.0 + gamma.to(Q.dtype)) * Q + beta.to(Q.dtype)
        return Q

    def _maybe_slot(self, Xt: torch.Tensor, vt: torch.Tensor) -> torch.Tensor:
        if not self.use_slot_kv:
            return Xt
        assert self.slot is not None
        B, N, _ = Xt.shape
        tok_valid = vt[:, None].expand(B, N)
        slots = self.slot(Xt, mask=tok_valid)
        return slots * vt[:, None, None].to(slots.dtype)

    def forward(
        self,
        x: torch.Tensor,                         # (B,T,F)
        t_len: Optional[torch.Tensor] = None,
        x_time: Optional[torch.Tensor] = None,    # (B,T,2) [abs, delta]
        x_pyr: Optional[torch.Tensor] = None,     # (B,T,S,F)
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        returns q_seq: (B,T,Kq,d)
        """
        B, T = x.shape[0], x.shape[1]
        device = x.device
        valid_mask = _safe_len_mask(t_len, B, T, device=device)

        if x_pyr is not None:
            if x_pyr.dim() != 4:
                raise ValueError(f"x_pyr must be (B,T,S,F), got {tuple(x_pyr.shape)}")
            if x_pyr.size(0) != B or x_pyr.size(1) != T:
                raise ValueError(f"x_pyr B/T mismatch. x_pyr={tuple(x_pyr.shape)} vs x={(B,T)}")
            S = x_pyr.size(2)
            x_tok = self.x_proj(x_pyr.reshape(B * T * S, -1)).reshape(B, T, S, self.d_model)  # (B,T,S,d)
        else:
            x_tok = self.x_proj(x).unsqueeze(2)  # (B,T,1,d)

        abs0: Optional[torch.Tensor] = None
        tau_acc = torch.zeros((B,), device=device, dtype=x.dtype)

        Xfirst: Optional[torch.Tensor] = None
        Xprev: Optional[torch.Tensor] = None
        Q_prev: Optional[torch.Tensor] = None

        q_seq: list[torch.Tensor] = []

        for t in range(T):
            vt = valid_mask[:, t]  # (B,)
            x_time_t = x_time[:, t] if (x_time is not None) else None

            dt = self._dt_from_x_time(x_time_t, B=B, device=device).to(device=device)
            abs_t = self._abs_from_x_time(x_time_t)
            if abs_t is not None:
                abs_t = abs_t.to(device=device)

            if abs0 is None:
                abs0 = abs_t.detach() if (abs_t is not None) else torch.zeros((B,), device=device, dtype=dt.dtype)

            if abs_t is not None:
                tau = (abs_t - abs0).clamp_min(0)
            else:
                tau_acc = torch.where(vt, tau_acc + dt.to(tau_acc.dtype), tau_acc)
                tau = tau_acc

            Xt = x_tok[:, t]  # (B,N,d)
            Xt = self._apply_dt_pos(Xt, dt)

            Xt_used = self._maybe_slot(Xt, vt=vt)  # (B,N',d)

            if Xfirst is None:
                Xfirst = Xt_used
            if Xprev is None:
                Xprev = Xt_used

            Nsp = Xt_used.size(1)
            Xt_kpm = (~vt).unsqueeze(1).expand(B, Nsp)  # True=PAD

            Pt, _w = self.compare_to_prog(Xt_used, Xprev, Xfirst, dt=dt, tau=tau, Xt_kpm=Xt_kpm)

            kv = torch.cat([Xt_used, Pt], dim=1)  # (B,Nsp+P,d)
            kv_kpm = (~vt).unsqueeze(1).expand(B, kv.size(1))

            if (not self.use_ar_query) or (Q_prev is None):
                Q = self.q_base.unsqueeze(0).expand(B, -1, -1).contiguous()
            else:
                Q = Q_prev

            if self.use_ar_query and (Q_prev is not None):
                Q = self._ar_dt_update(Q, dt)

            for blk in self.blocks:
                Q = blk(Q, kv, mem_kpm=kv_kpm)
            Q = self.out_norm(Q)

            Q = Q * vt[:, None, None].to(Q.dtype)
            q_seq.append(Q)

            Xprev = torch.where(vt[:, None, None], Xt_used, Xprev)

            if self.use_ar_query:
                if Q_prev is None:
                    Q_prev = Q
                else:
                    Q_prev = torch.where(vt[:, None, None], Q, Q_prev)

        return torch.stack(q_seq, dim=1)  # (B,T,Kq,d)


# =============================================================================
# Classifier wrapper
#   - AR: last valid timestep rep
#   - non-AR: TemporalAttentionPool over all Q tokens
# =============================================================================
class AR_Qall_Classifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        in_dim: int,
        d_model: int = 256,
        depth: int = 1,
        n_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        num_query_tokens: int = 6,
        # TA pooling (non-AR only)
        ta_heads: int = 4,
        ta_pool_tokens: int = 1,
        use_pos_time_embed: bool = True,
        dt_hidden: int = 128,
        # progression
        num_prog_tokens: int = 4,
        # STD (OT only)
        std_ot_eps: float = 0.1,
        std_ot_iters: int = 20,
        # LTD (ODE only)
        ltd_ode_width: int = 256,
        ltd_ode_steps: int = 4,
        # slot
        use_slot_kv: bool = False,
        slot_num: int = 6,
        slot_iters: int = 3,
        slot_heads: int = 4,
        # tx
        use_tx: bool = True,
        tx_dim: int = 0,
        tx_proj: int = 128,
        # AR
        use_ar_query: bool = False,
        ar_dt_mode: Literal["none", "add", "film"] = "add",
    ):
        super().__init__()

        self.backbone = AR_Qall_Backbone(
            in_dim=in_dim,
            d_model=d_model,
            num_query_tokens=num_query_tokens,
            depth=depth,
            nhead=n_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
            use_pos_time_embed=use_pos_time_embed,
            dt_hidden=dt_hidden,
            num_prog_tokens=num_prog_tokens,
            std_ot_eps=std_ot_eps,
            std_ot_iters=std_ot_iters,
            ltd_ode_width=ltd_ode_width,
            ltd_ode_steps=ltd_ode_steps,
            use_slot_kv=use_slot_kv,
            slot_num=slot_num,
            slot_iters=slot_iters,
            slot_heads=slot_heads,
            use_ar_query=use_ar_query,
            ar_dt_mode=ar_dt_mode,
        )

        self.ta_pool = TemporalAttentionPool(
            d_model=d_model,
            nhead=int(ta_heads),
            dropout=float(dropout),
            num_pool_tokens=int(ta_pool_tokens),
        )

        self.use_tx = bool(use_tx and tx_dim > 0)
        self.tx_proj = nn.Linear(tx_dim, tx_proj) if self.use_tx else None

        out_in = d_model + (tx_proj if self.use_tx else 0)
        self.head = nn.Sequential(
            nn.LayerNorm(out_in),
            nn.Linear(out_in, out_in),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_in, num_classes),
        )

    def _make_bank_mask(self, t_len: Optional[torch.Tensor], B: int, T: int, Kq: int, device) -> torch.Tensor:
        valid_mask = _safe_len_mask(t_len, B, T, device=device)  # (B,T) True=valid
        pad = ~valid_mask
        return pad.unsqueeze(-1).expand(B, T, Kq).reshape(B, T * Kq)  # (B,T*Kq) True=PAD

    def forward(
        self,
        x: torch.Tensor,
        t_len: Optional[torch.Tensor] = None,
        x_tx: Optional[torch.Tensor] = None,
        x_time: Optional[torch.Tensor] = None,
        x_pyr: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ):
        q_seq = self.backbone(
            x=x,
            t_len=t_len,
            x_time=x_time,
            x_pyr=x_pyr,
        )  # (B,T,Kq,d)

        B, T, Kq, d = q_seq.shape
        bank = q_seq.reshape(B, T * Kq, d)
        bank_kpm = self._make_bank_mask(t_len, B=B, T=T, Kq=Kq, device=bank.device)

        if getattr(self.backbone, "use_ar_query", False):
            if t_len is None:
                rep = q_seq[:, -1].mean(dim=1)
            else:
                idx = (t_len.clamp(min=1) - 1).to(q_seq.device)
                rep = q_seq[torch.arange(B, device=q_seq.device), idx].mean(dim=1)
        else:
            rep = self.ta_pool(bank, bank_kpm=bank_kpm)

        if self.use_tx and x_tx is not None:
            if t_len is None:
                tx_last = x_tx[:, -1]
            else:
                idx = (t_len.clamp(min=1) - 1).view(B, 1, 1).expand(B, 1, x_tx.size(-1)).to(x_tx.device)
                tx_last = x_tx.gather(1, idx).squeeze(1)
            rep = torch.cat([rep, self.tx_proj(tx_last)], dim=-1)

        logits = self.head(rep)
        return logits


def build_ar_qall_tokenizer_model(
    num_classes: int,
    in_dim: int,
    d_model: int = 256,
    depth: int = 1,
    n_heads: int = 4,
    mlp_ratio: float = 4.0,
    dropout: float = 0.0,
    num_query_tokens: int = 6,
    # TA pooling (non-AR only)
    ta_heads: int = 4,
    ta_pool_tokens: int = 1,
    use_pos_time_embed: bool = True,
    dt_hidden: int = 128,
    # progression
    num_prog_tokens: int = 4,
    # STD (OT only)
    std_ot_eps: float = 0.1,
    std_ot_iters: int = 20,
    # LTD (ODE only)
    ltd_ode_width: int = 256,
    ltd_ode_steps: int = 4,
    # slot
    use_slot_kv: bool = False,
    slot_num: int = 6,
    slot_iters: int = 3,
    slot_heads: int = 4,
    # tx
    use_tx: bool = True,
    tx_dim: int = 0,
    tx_proj: int = 128,
    # AR
    use_ar_query: bool = False,
    ar_dt_mode: Literal["none", "add", "film"] = "add",
) -> nn.Module:
    return AR_Qall_Classifier(
        num_classes=num_classes,
        in_dim=in_dim,
        d_model=d_model,
        depth=depth,
        n_heads=n_heads,
        mlp_ratio=mlp_ratio,
        dropout=dropout,
        num_query_tokens=num_query_tokens,
        ta_heads=ta_heads,
        ta_pool_tokens=ta_pool_tokens,
        use_pos_time_embed=use_pos_time_embed,
        dt_hidden=dt_hidden,
        num_prog_tokens=num_prog_tokens,
        std_ot_eps=std_ot_eps,
        std_ot_iters=std_ot_iters,
        ltd_ode_width=ltd_ode_width,
        ltd_ode_steps=ltd_ode_steps,
        use_slot_kv=use_slot_kv,
        slot_num=slot_num,
        slot_iters=slot_iters,
        slot_heads=slot_heads,
        use_tx=use_tx,
        tx_dim=tx_dim,
        tx_proj=tx_proj,
        use_ar_query=use_ar_query,
        ar_dt_mode=ar_dt_mode,
    )
