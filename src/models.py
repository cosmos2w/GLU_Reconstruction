


from __future__ import annotations
from typing import Tuple, Callable, Optional, List
from torch.utils.checkpoint import checkpoint
from collections import deque

import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.checkpoint as cp
import torch.distributions as dist
import os, math
import numpy as np

try:
    from torchdiffeq import odeint  
except Exception:
    def odeint(*args, **kwargs):
        raise ImportError("torchdiffeq is required only for DelayEmbedNeuralODE, which is not part of the default public release path.")

from einops import repeat
try:
    from fairscale.nn import checkpoint_wrapper  
except Exception:
    def checkpoint_wrapper(module):
        return module

def apply_rope(x: torch.Tensor,
               pos: torch.Tensor,
               base: float = 10000.0) -> torch.Tensor:
    """
    Rotary Position Embedding.
    x   : (B, L, D)            - token features, D must be even
    pos : (B, L, 1) or (B, L)  - scalar position of every token
    """
    B, L, D = x.shape
    assert D % 2 == 0, "RoPE needs an even hidden size"
    half_D = D // 2

    # frequencies 1 / base^{2k/d}
    inv_freq = 1.0 / (base ** (torch.arange(0, half_D, device=x.device).float() / half_D))  # (D/2,)

    # broadcast positions → (B, L, D/2)
    theta = pos.float() * inv_freq            # (B, L, D/2)
    sin, cos = torch.sin(theta), torch.cos(theta)

    # split the embedding in two blocks [0:half_D | half_D:]
    x1, x2 = x[..., :half_D], x[..., half_D:]  # each (B, L, D/2)

    # apply planar rotation
    rot_x1 = x1 * cos - x2 * sin
    rot_x2 = x2 * cos + x1 * sin
    return torch.cat([rot_x1, rot_x2], dim=-1)

class MLP(nn.Module):
    def __init__(self, dims, activation='relu', use_bias=True, final_activation=None):
        """
        dims (list of int): List such that dims = [n_input, hidden1, ..., n_output].
        activation (str):   Activation for the hidden layers ("relu", "tanh", etc.).
        use_bias (bool):    Whether to use bias in each linear layer.
        final_activation (str or None): Final activation function (if any) for the last layer.
        """
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = []
        act_fn = self.get_activation(activation)
        final_act_fn = self.get_activation(final_activation) if final_activation is not None else None
        
        for i in range(len(dims)-1):
            self.layers.append(nn.Linear(dims[i], dims[i+1], bias=use_bias))
            if i < len(dims)-2:
                self.activations.append(act_fn)
            else:
                self.activations.append(final_act_fn)
                
    def get_activation(self, act_str):
        if act_str is None:
            return None
        if act_str.lower() == 'relu':
            return nn.ReLU()
        elif act_str.lower() == 'tanh':
            return nn.Tanh()
        elif act_str.lower() == 'sigmoid':
            return nn.Sigmoid()
        else:
            raise ValueError(f"Unsupported activation: {act_str}")

    def forward(self, x):
        for layer, act in zip(self.layers, self.activations):
            x = layer(x)
            if act is not None:
                x = act(x)
        return x

#   Gated Linear Unit with GELU
#   Use one part of the signal to dynamically control (gate) another part.
class GEGLU(nn.Module):
    def __init__(self, d_model, mult=4):
        super().__init__()
        self.proj = nn.Linear(d_model, mult * d_model * 2)  # 2× for gate
        self.out  = nn.Linear(mult * d_model, d_model)
        self.gelu = nn.GELU()

    def forward(self, x):
        a, b = self.proj(x).chunk(2, dim=-1)   # each (…,4d)
        return self.out(a * self.gelu(b))

class FourierEmbedding(nn.Module):
    """
    phi(x) = [sin(2pi B x) , cos(2pi B x)], B ∈ R^{d * M}
    """
    def __init__(self, in_dim: int = 2, num_frequencies: int = 64,
                 learnable: bool = True, sigma: float = 10.0):
        super().__init__()
        B = torch.randn(in_dim, num_frequencies) * sigma
        self.B = nn.Parameter(B, requires_grad=learnable)

    def forward(self, xy: torch.Tensor) -> torch.Tensor:
        proj = 2.0 * math.pi * xy @ self.B          # (..., M)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

class SpatiallyAwareCLSFusion(nn.Module):
    """
    Lightweight CLS fusion that adapts to query location.
    Minimal overhead: just 2 small MLPs for gating.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Simple gating mechanism
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid()
        )
        # Project CLS and local features
        self.cls_proj = nn.Linear(d_model, d_model)
        self.local_proj = nn.Linear(d_model, d_model)

    def forward(self, local_features, cls_features):
        """
        Args:
            local_features: [B*T, P, d_model] - from sensor aggregation
            cls_features: [B*T, 1, d_model] - global CLS token
        Returns:
            fused: [B*T, P, d_model]
        """
        # Expand CLS to all query points
        cls_expanded = cls_features.expand(-1, local_features.size(1), -1)
        # Compute adaptive gate based on local features
        # (locations that need global info will have different gates)
        gate = self.gate_net(local_features)  # [B*T, P, 1]
        # Weighted fusion
        cls_proj = self.cls_proj(cls_expanded)
        local_proj = self.local_proj(local_features)
        fused = gate * local_proj + (1 - gate) * cls_proj
        return fused

class CLSConditionedCoordEncoder(nn.Module):
    """
    Makes coordinate embeddings aware of global CLS context.
    Negligible cost: just element-wise modulation.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.cls_to_modulation = nn.Linear(d_model, d_model)
    def forward(self, coord_emb, cls_token):
        """
        Args:
            coord_emb: [B*T, P, d_model] - base coordinate embeddings
            cls_token: [B*T, 1, d_model] - global CLS
        Returns:
            modulated_emb: [B*T, P, d_model]
        """
        # CLS generates global modulation
        modulation = self.cls_to_modulation(cls_token)  # [B*T, 1, d_model]
        # Element-wise modulation (like FiLM)
        # This makes coords domain-aware
        modulated = coord_emb * (1.0 + 0.1 * torch.tanh(modulation))
        return modulated

# ------------------------------------------------------------------
class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float = 0.0,
        layers: int = 1,               # number of repeated applications
        use_layernorm: bool = False,   # stability when layers > 1
        residual: bool = False,        # q + attn_out
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.layers = layers
        self.residual = residual
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm = nn.LayerNorm(dim) if use_layernorm else nn.Identity()

    def forward(self, q, k, v):
        """
        q: (B, Tq, C), k: (B, Tk, C), v: (B, Tk, C)
        returns: (B, Tq, C)
        """
        x = q
        for _ in range(self.layers):
            out, _ = self.attn(x, k, v, need_weights=False)
            if self.residual:
                out = x + self.dropout(out)
            out = self.norm(out)
            x = out
        return x

def latent_block(dim, num_heads, dropout):
    return nn.TransformerEncoderLayer(dim,
                                      num_heads,
                                      dropout=dropout,
                                      batch_first=True,
                                      activation="gelu")

# --------------------------------------------------
#  Small re-usable building blocks
# --------------------------------------------------
def build_pos_value_proj(N_channels: int,
                         num_freqs: int,
                         All_dim: int) -> nn.ModuleDict:
    return nn.ModuleDict({
        "pos_embed": FourierEmbedding(2, num_freqs, learnable=True),
        "pos_linear": nn.Linear(2 * num_freqs, All_dim),
        "pos_norm": nn.RMSNorm(All_dim),
        "val_linear": nn.Linear(N_channels, All_dim),
        "val_norm": nn.RMSNorm(All_dim),
    })

def build_latent_bank(All_dim: int,
                      latent_tokens: int,
                      with_cls: bool = True) -> nn.ParameterDict:
    p = {}
    p["latent_param"] = nn.Parameter(torch.randn(1, latent_tokens, All_dim) * 0.02)
    if with_cls:
        p["cls_token"]  = nn.Parameter(torch.zeros(1, 1, All_dim))
    return nn.ParameterDict(p)

def build_transformer_stack(All_dim: int,
                            num_heads: int,
                            num_layers: int,
                            dropout: float = 0.) -> nn.TransformerEncoder:
    return nn.TransformerEncoder(
        latent_block(All_dim, num_heads, dropout),
        num_layers=num_layers
    )

# --------------------------------------------------
#  Stand-alone Domain Adaptive Encoder Module
# --------------------------------------------------
class DomainAdaptiveEncoder(nn.Module):

    def __init__(self,
                 All_dim       : int,
                 num_heads     : int,
                 latent_layers : int,
                 N_channels    : int,
                 num_freqs     : int = 64,
                 latent_tokens : int = 8,
                 pooling       : str = "none",      # currently ignored
                 *,
                 retain_cls    : bool = False,
                 retain_lat    : bool = False,
                 channel_to_encode: Optional[List[int]] = None,
                 ):
        super().__init__()
        assert pooling in ("mean", "cls", "none")

        # --- Store channel selection and determine input size ---
        self.channel_to_encode = channel_to_encode
        if self.channel_to_encode is None:
            # If no specific channels are given, use all N_channels
            num_selected_channels = N_channels
        else:
            # If a list of channels is given, the number of channels is its length
            # Add validation to ensure channel indices are valid
            if max(self.channel_to_encode) >= N_channels or min(self.channel_to_encode) < 0:
                raise ValueError(f"channel_to_encode contains indices out of range for N_channels={N_channels}")
            num_selected_channels = len(self.channel_to_encode)
            print(f"Encoder will use {num_selected_channels} specific channels: {self.channel_to_encode}")

        # 1. Positional + value projection sub-modules ---------------------------
        self.embed = build_pos_value_proj(num_selected_channels, num_freqs, All_dim)

        # 2. Latent / CLS parameters --------------------------------------------
        self.latents = build_latent_bank(All_dim, latent_tokens, with_cls = retain_cls)

        # 3. Attention blocks ----------------------------------------------------
        self.cross_attn            = CrossAttention(All_dim, num_heads, dropout = 0.0,
                                                    layers = latent_layers
                                                    )
        self.cross_norm            = nn.RMSNorm(All_dim)
        self.cross_latent_mixer    = build_transformer_stack(All_dim, num_heads,
                                                             latent_layers)

        self.token_to_latent       = CrossAttention(All_dim, num_heads, dropout = 0.0,
                                                    layers = latent_layers
                                                    )
        self.token_to_latent_norm  = nn.RMSNorm(All_dim)
        self.token_to_latent_mixer = build_transformer_stack(All_dim, num_heads,
                                                             latent_layers)

        # -----------------------------------------------------------------------
        self.retain_cls = retain_cls
        print(f'self.retain_cls is {self.retain_cls} ! ')
        self.retain_lat = retain_lat
        print(f'self.retain_lat is {self.retain_lat} ! ')

    def forward(self,
                coords_tuv   : torch.Tensor,                     # (B,T,N, 2+C)
                U            : torch.Tensor,                     # (unused here)
                original_phi : Optional[torch.Tensor] = None     # (B,S)
                ):

        B, T, N_inp, _ = coords_tuv.shape

        # ------------------------------------------------------------------ 
        # Split coordinates and sensor values
        xy   = coords_tuv[..., 0:2]                                             # (B,T,N,2)
        # --- Select the correct value channels ---
        if self.channel_to_encode is None:
            C_in = self.embed['val_linear'].in_features
            vals = coords_tuv[..., 2:2 + C_in]                                  # (B,T,N,C)
        else:
            # Select specific channels using advanced indexing.
            # The value channels start at index 2 of the last dimension.
            value_channels_in_tensor = [c + 2 for c in self.channel_to_encode]
            vals = coords_tuv[..., value_channels_in_tensor]                   # (B,T,N, len(channel_to_encode))

        # Positional & value embeddings
        pos_tok = self.embed['pos_linear'](self.embed['pos_embed'](xy))
        pos_tok = self.embed['pos_norm'](pos_tok)

        val_tok = self.embed['val_linear'](vals)
        val_tok = self.embed['val_norm'](val_tok)

        tok      = pos_tok + val_tok                                           # (B,T,N,D)
        tok_flat = tok.reshape(B * T, N_inp, -1)                               # (B*T,N,D)

        # ------------------------------------------------------------------ 
        lat = self.latents['latent_param'].expand(B * T, -1, -1)               # (B*T,L,D)
        # Prepare latent + CLS tokens
        if self.retain_cls:
            cls = self.latents['cls_token'].expand(B * T, 1, -1)                   # (B*T,1,D)
            lat = torch.cat([cls, lat], dim=1)                                     # (B*T,1+L,D)

        # Cross-attention: sensor tokens -> latent tokens
        lat = self.cross_norm(lat)
        lat = lat + self.cross_attn(lat, tok_flat, tok_flat)
        lat = lat + self.cross_latent_mixer(lat)
        # Keep a copy of the final latents 
        final_latents = lat

        # ------------------------------------------------------------------ 
        # Feed refined latent information back into sensor tokens
        tok_flat_up  = self.token_to_latent_norm(tok_flat)
        tok_flat_up  = tok_flat_up + self.token_to_latent(tok_flat_up, lat, lat)
        tok_flat_out = tok_flat_up + self.token_to_latent_mixer(tok_flat_up)

        # ------------------------------------------------------------------ 
        # retain_latents has higher prioriy
        if self.retain_lat:
            # The global tokens go first, then the local tokens
            tok_flat_out = torch.cat([final_latents, tok_flat_out], dim=1)     # (B*T, L+N, D)
        # Optionally keep CLS as part of the output token set
        elif self.retain_cls:
            cls_upd   = lat[:, :1, :]                                          # (B*T,1,D)
            tok_flat_out = torch.cat([cls_upd, tok_flat_out], dim=1)           # (B*T,1+N,D)

        # ------------------------------------------------------------------ 
        # Reshape back to (B,T,⋯)
        S        = tok_flat_out.size(1) # This is 1+N, L+N or N
        lat_vec  = tok_flat_out.view(B, T, S, -1)                              # (B,T,S,D)

        mask     = torch.ones(B, T, S, dtype=torch.bool, device=lat_vec.device)
        coords   = coords_tuv[:, 0, :, :2]                                     # (B,N,2)
        merged_phi = (original_phi
                      if original_phi is not None
                      else torch.ones(B, S, device=lat_vec.device))

        return lat_vec, mask, coords, merged_phi

# ================================================================
# Baseline - Full-softmax temporal decoder 
# ================================================================
class MultiheadSoftmaxAttention(nn.Module):
    """
    Causal multi-head soft-max attention that always calls
    torch.nn.functional.scaled_dot_product_attention (SDPA).

    · forward(x) : B × T × D  →  B × T × D
    · step(x_t)  : B ×   D    →  B ×   D         (recurrent KV cache)
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.hdim    = d_model // n_heads
        self.dropout_p = dropout

        # projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # caches for autoregressive generation
        self.K: torch.Tensor | None = None   # (B,h,T,d)
        self.V: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _split(self, x: torch.Tensor):
        """(B,T,D) -> (B,h,T,d)"""
        B, T, _ = x.shape
        return (
            x.view(B, T, self.n_heads, self.hdim)
            .transpose(1, 2)                       # B h T d
        )

    def _merge(self, x: torch.Tensor):
        """(B,h,T,d) -> (B,T,D)"""
        B, h, T, d = x.shape
        return (
            x.transpose(1, 2)
            .contiguous()
            .view(B, T, h * d)
        )

    # ------------------------------------------------------------------
    # full sequence (teacher forcing)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self._split(self.W_q(x))                # B h T d
        k = self._split(self.W_k(x))
        v = self._split(self.W_v(x))

        # SDPA automatically picks Flash / Triton / math
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=True,
        )                                           # B h T d

        return self.out(self._merge(y))             # B T D

    # ------------------------------------------------------------------
    # incremental step
    # ------------------------------------------------------------------
    def reset_state(self, batch_size: int, device=None):
        self.K = None
        self.V = None
        self._bs = batch_size
        self._device = device

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        x_t : (B, D)
        """
        B, _ = x_t.shape
        if (self.K is None) or (B != self._bs):
            self.reset_state(B, x_t.device)

        q = self.W_q(x_t).view(B, self.n_heads, 1, self.hdim)      # B h 1 d
        k = self.W_k(x_t).view(B, self.n_heads, 1, self.hdim)
        v = self.W_v(x_t).view(B, self.n_heads, 1, self.hdim)

        # append to cache along seq-length dimension
        self.K = k if self.K is None else torch.cat((self.K, k), dim=2)
        self.V = v if self.V is None else torch.cat((self.V, v), dim=2)

        y = F.scaled_dot_product_attention(
            q, self.K, self.V,
            attn_mask=None,
            dropout_p=0.0,            # no dropout during generation
            is_causal=True,
        )                             # B h 1 d
        y = y.squeeze(2)              # B h d
        y = y.transpose(1, 2).contiguous().reshape(B, self.d_model)
        return self.out(y)

# Relative posotional encoding
class TemporalDecoderSoftmax(nn.Module):

    def __init__(
        self,
        d_model      : int,
        n_layers     : int = 4,
        n_heads      : int = 4,
        max_len      : int = 4096,
        dt           : float = 0.02,

        n_window     : int = 16,   # Fixed window size for history

        learnable_dt : bool  = False,
        dropout      : float = 0.0,
        rope_base     : float = 1000.0,
        checkpoint_every_layer: bool = True,
    ):
        super().__init__()
        self.d_model  = d_model
        self.n_layers = n_layers
        self.n_heads  = n_heads
        self.rope_base = rope_base
        self.n_window  = n_window
        self.layers = nn.ModuleList(
            [MultiheadSoftmaxAttention(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.pre_norms  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.post_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
            ) for _ in range(n_layers)
        ])

        # Heun integrator projection head
        self.head = nn.Linear(d_model, d_model)

        # learnable absolute positional embedding
        self.pos_emb = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

        # dt handling ---------------------------------------------------
        self.register_buffer("dt_const", torch.tensor(dt))
        if learnable_dt:
            self.dt_scale = nn.Parameter(torch.zeros(()))
        else:
            self.dt_scale = None

        self.use_ckpt = checkpoint_every_layer

        print('Initialized TemporalDecoderSoftmax FIXED for relative posotional encoding !')

    # FIXED for relative posotional encoding
    def apply_rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        # x: (B,T,D), pos: (1,T,1) float
        B, T, D = x.shape
        d = D // 2

        x_dtype = x.dtype
        x = x.float()
        pos = pos.float()

        freq = self.rope_base ** (-torch.arange(d, device=x.device, dtype=torch.float32) / d)  # (d,)
        angle = pos * freq.view(1, 1, d)  # (1,T,d)
        s, c = angle.sin(), angle.cos()

        x1, x2 = x[..., :d], x[..., d:]
        xr = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
        return xr.to(x_dtype)

    # FIXED for relative posotional encoding
    def _add_pos(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        # IGNORE offset: always relative within the provided sequence
        if x.dim() == 3:
            B, T, D = x.shape
            pos = torch.arange(0, T, device=x.device, dtype=torch.float32).view(1, T, 1)
            return self.apply_rope(x, pos)
        else:
            # single token -> treat as length-1 sequence at position 0
            pos = torch.zeros(1, 1, 1, device=x.device, dtype=torch.float32)  # (1,1,1)
            return self.apply_rope(x.unsqueeze(1), pos).squeeze(1)

    # FIXED for relative posotional encoding
    def _encode_window(self, x_win: torch.Tensor) -> torch.Tensor:
        """
        x_win: (B, T_win, D)
        returns: (B, T_win, D)
        """
        x = self._add_pos(x_win, 0)  # relative positions 0..T_win-1

        for idx in range(len(self.layers)):
            # always non-incremental here
            x = self._block_ckpt(idx, x) if self.training else self._block(idx, x, incremental=False)
        return x

    def _effective_dt(self):
        if self.dt_scale is None:
            return self.dt_const
        return self.dt_const * F.softplus(self.dt_scale)

    # ------------------------------------------------------------------
    #  Single transformer block (with optional checkpointing)
    # ------------------------------------------------------------------
    def _block(self, idx: int, x: torch.Tensor, *, incremental: bool) -> torch.Tensor:
        ln_pre, attn, ln_post, ffn = \
            self.pre_norms[idx], self.layers[idx], self.post_norms[idx], self.ffns[idx]

        residual = x
        x = ln_pre(x)
        x = attn.step(x) if incremental else attn(x)
        x = ln_post(residual + x)
        x = x + ffn(x)
        return x

    def _block_ckpt(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        if not (self.use_ckpt and self.training):
            return self._block(idx, x, incremental=False)

        def fn(y):
            return self._block(idx, y, incremental=False)
        return cp.checkpoint(fn, x, use_reentrant=False)

    # ------------------------------------------------------------------
    #  Parallel forward (teacher forcing over full sequence)
    # ------------------------------------------------------------------
    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        x_seq : (B, T, D)
        """
        x_seq = self._add_pos(x_seq, 0)

        for idx in range(len(self.layers)):
            x_seq = self._block_ckpt(idx, x_seq)

        k1 = self.head(x_seq)
        dt = self._effective_dt()
        k2 = self.head(self._forward_no_ckpt(x_seq + dt * k1))
        return x_seq + 0.5 * dt * (k1 + k2)

    # ------------------------------------------------------------------
    #  Autoregressive rollout with gradients (training)
    # ------------------------------------------------------------------
    # FIXED for relative posotional encoding
    def rollout_with_grad(
        self,
        obs_window: torch.Tensor,   # (B, T_obs, D)
        N_fore: int,
        *,
        truncate_k: int | None = 64,
        teacher_force_seq: torch.Tensor | None = None,
        teacher_force_prob: float = 0.0,
    ) -> torch.Tensor:

        assert self.training, "Call only in training mode"
        B, T_obs, D = obs_window.shape
        dev = obs_window.device

        # Sliding history of raw latent states (NO caches)
        W = self.n_window
        history = deque(maxlen=W)
        for t in range(T_obs):
            history.append(obs_window[:, t])  # (B,D)

        outputs = [obs_window]  # keep first T_obs

        steps_left = N_fore - T_obs
        if steps_left <= 0:
            return obs_window[:, :N_fore]

        dt_eff = self._effective_dt()
        latent_cur = obs_window[:, -1]  # (B,D)

        for step in range(steps_left):
            # Build window tensor: (B, T_win, D)
            x_win = torch.stack(list(history), dim=1)

            # Encode window, take last token representation
            y_win = self._encode_window(x_win)
            y_last = y_win[:, -1]  # (B,D)

            k1 = self.head(y_last)               # (B,D)
            latent_next = latent_cur + dt_eff * k1

            # Teacher forcing (optional)
            if (
                teacher_force_seq is not None
                and step < teacher_force_seq.size(1)
                and torch.rand((), device=dev) < teacher_force_prob
            ):
                latent_next = teacher_force_seq[:, step]

            # Append
            history.append(latent_next)
            outputs.append(latent_next.unsqueeze(1))
            latent_cur = latent_next

            # Truncated BPTT: detach history states
            if truncate_k and ((step + 1) % truncate_k == 0):
                history = deque([h.detach() for h in history], maxlen=W)
                latent_cur = history[-1]

        return torch.cat(outputs, dim=1)  # (B, N_fore, D)

    # ------------------------------------------------------------------
    #  Greedy generation (no grad) – evaluation / inference
    # ------------------------------------------------------------------
    # FIXED for relative posotional encoding
    @torch.no_grad()
    def generate(self, obs_window: torch.Tensor, N_fore: int) -> torch.Tensor:
        B, T_obs, D = obs_window.shape
        W = self.n_window
        dt = self._effective_dt()

        history = deque(maxlen=W)
        for t in range(T_obs):
            history.append(obs_window[:, t])

        outputs = [obs_window]
        steps_left = N_fore - T_obs
        if steps_left <= 0:
            return obs_window[:, :N_fore]

        latent_cur = obs_window[:, -1]
        for _ in range(steps_left):
            x_win = torch.stack(list(history), dim=1)
            y_win = self._encode_window(x_win)
            y_last = y_win[:, -1]
            k1 = self.head(y_last)
            latent_cur = latent_cur + dt * k1

            history.append(latent_cur)
            outputs.append(latent_cur.unsqueeze(1))

        return torch.cat(outputs, dim=1)

    # ------------------------------------------------------------------
    #  shared utilities
    # ------------------------------------------------------------------
    def _forward_no_ckpt(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.layers)):
            x = self._block(idx, x, incremental=False)
        return x

    def _step_layers(self, x_t: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.layers)):
            x_t = self._block(idx, x_t, incremental=True)
        return x_t

# ================================================================
# Uncertainty-aware Leader-Follower-Dynamics temporal decoder 
# ================================================================
class ud_MultiheadSoftmaxAttention(nn.Module):
    def __init__(self, d_model, num_heads, gamma=1.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # Learnable weights for penalties (minimal addition)
        self.w_sigma = nn.Parameter(torch.tensor(1.0))
        self.w_phi = nn.Parameter(torch.tensor(1.0))
        self.gamma = gamma

        # Initialize instance vars for step method
        self.K = None
        self.V = None

    def forward(self, query, key, value, phi=None, logvar=None):
        B, T, _ = query.shape
        assert query.shape == key.shape == value.shape, "Query, key, value shapes must match"
        q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # New: Uncertainty-aware modulation (penalize high logvar/low phi)
        if logvar is not None:
            assert logvar.shape[:2] == (B, T), f"Logvar shape mismatch: expected (B, T, *), got {logvar.shape}"
            logvar_broadcast = logvar.unsqueeze(1).expand(-1, self.num_heads, T, -1).mean(dim=-1, keepdim=True)  # Avg per time-step, expanded for heads
            scores -= self.w_sigma * logvar_broadcast.unsqueeze(2)  # Broadcast to match scores shape
        if phi is not None:
            assert phi.shape[:2] == (B, T), f"Phi shape mismatch: expected (B, T, *), got {phi.shape}"
            phi_broadcast = phi.unsqueeze(1).expand(-1, self.num_heads, T, -1).mean(dim=-1, keepdim=True)
            scores -= self.w_phi * (1 - phi_broadcast.unsqueeze(2))
            # Additional phi modulation from my proposal
            phi_denom = torch.clamp(phi_broadcast.unsqueeze(2) ** self.gamma, min=1e-6)  # Clamp for stability
            scores = scores / phi_denom
        
        attn = scores.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out)

    def reset_state(self, B, device):
        self.K = self.V = None

    def step(self, x_t, phi=None, logvar=None):
        # x_t : (B,1,D)
        assert x_t.dim() == 3 and x_t.shape[1] == 1, f"Expected (B,1,D), got {x_t.shape}"
        q = self.q_proj(x_t).view(x_t.shape[0], 1, self.num_heads, self.head_dim).transpose(1, 2)  # Multi-head reshape
        k = self.k_proj(x_t).view(x_t.shape[0], 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_t).view(x_t.shape[0], 1, self.num_heads, self.head_dim).transpose(1, 2)
        k_flat = k.transpose(1, 2).squeeze(1)  # Flatten for cat
        v_flat = v.transpose(1, 2).squeeze(1)
        self.K = k_flat if self.K is None else torch.cat([self.K, k_flat], 1)
        self.V = v_flat if self.V is None else torch.cat([self.V, v_flat], 1)

        scores = (q @ self.K.view(q.shape[0], self.num_heads, -1, self.head_dim).transpose(-2,-1)) / math.sqrt(self.head_dim)
        if logvar is not None:
            scores -= self.w_sigma * logvar.mean(dim=-1, keepdim=True).unsqueeze(1)  # Expanded for heads
        if phi is not None:
            scores -= self.w_phi * (1 - phi.mean(dim=-1, keepdim=True).unsqueeze(1))
            phi_denom = torch.clamp(phi.mean(dim=-1, keepdim=True).unsqueeze(1) ** self.gamma, min=1e-6)
            scores = scores / phi_denom

        attn = scores.softmax(-1)
        out  = attn @ self.V.view(attn.shape[0], self.num_heads, -1, self.head_dim)
        out = out.transpose(1, 2).contiguous().view(x_t.shape[0], 1, self.d_model)
        return self.out_proj(out)

class CAU_MultiheadCrossAttention(nn.Module):
    """
    Causal multi-head cross-attention with optional importance weights.

    query:      [B, T_q, D] or [B, 1, N_q, D] or [B, D]
    key_value:  [B, T_kv, D] or [B, T_kv, N_s, D] or [B, N_s, D] or [B, D]

    Importance weights (imp_weights):
      - If key_value provides a sensor axis of length N_s, pass imp_weights as [B, N_s].
      - We append these per-step to an internal importance cache aligned with K/V tokens.
      - Applied as additive mask: log(clamp(imp, eps, 1.0)) added to attention logits.

    Caches:
      - self.K, self.V: [B, h, S, d], where S is the accumulated source length.
      - self.imp_cache: [B, S] for per-source importance, optional.
    """
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.hdim = d_model // n_heads
        self.dropout_p = dropout

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

        # K/V caches (already split into heads) and aligned importance cache
        self.K: torch.Tensor | None = None  # [B, h, S, d]
        self.V: torch.Tensor | None = None  # [B, h, S, d]
        self.imp_cache: torch.Tensor | None = None  # [B, S]
        self._bs: int | None = None
        self._device = None

    def _split(self, x: torch.Tensor):
        # x: [B, T, D] -> [B, h, T, d]
        B, T, _ = x.shape
        return x.view(B, T, self.n_heads, self.hdim).transpose(1, 2)

    def _merge(self, x: torch.Tensor):
        # x: [B, h, T, d] -> [B, T, D]
        B, h, T, d = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, h * d)

    def reset_state(self, batch_size: int, device=None):
        self.K = None
        self.V = None
        self.imp_cache = None
        self._bs = batch_size
        self._device = device

    @staticmethod
    def _to_B_T_D(x: torch.Tensor):
        """
        Normalize x to [B, T, D] and return a callable to restore original shape.
        Supported:
          - [B, D] -> [B, 1, D]
          - [B, T, D] -> [B, T, D]
          - [B, T, N, D] -> [B, T*N, D]
        """
        if x.dim() == 2:
            # [B, D]
            B, D = x.shape
            xf = x.unsqueeze(1)  # [B, 1, D]
            def restore(y):  # y: [B, 1, D]
                return y.squeeze(1)  # [B, D]
            return xf, restore

        if x.dim() == 3:
            # [B, T, D]
            B, T, D = x.shape
            def restore(y):  # [B, T, D]
                return y
            return x, restore

        if x.dim() == 4:
            # [B, T, N, D] -> flatten to [B, T*N, D]
            B, T, N, D = x.shape
            xf = x.reshape(B, T * N, D)
            def restore(y):  # [B, T*N, D] -> [B, T, N, D]
                return y.view(B, T, N, D)
            return xf, restore

        raise ValueError(f"Unsupported tensor rank: {x.shape}")

    def _append_imp(self, imp_new: torch.Tensor | None, Tk: int, *, B: int, device, dtype):
        """
        Append per-source importance for the newly appended KV tokens (length Tk).
        imp_new: [B, Tk] or None -> appends ones if None.
        """
        if imp_new is None:
            imp_new = torch.ones(B, Tk, device=device, dtype=dtype)
        if self.imp_cache is None:
            self.imp_cache = imp_new
        else:
            self.imp_cache = torch.cat([self.imp_cache, imp_new], dim=1)  # [B, S+Tk]

    def forward(self, query: torch.Tensor, key_value: torch.Tensor, imp_weights: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parallel attention (no caching).
        query:     [B, T_q, D] or [B, T_q, N_q, D] or [B, D]
        key_value: [B, T_kv, D] or [B, T_kv, N_s, D] or [B, N_s, D] or [B, D]
        imp_weights:
          - If key_value provides N_s, pass [B, N_s]; repeated over time if T_kv > 1.
        """
        q, restore_q = self._to_B_T_D(query)      # [B, T_q, D]
        kv, _ = self._to_B_T_D(key_value)         # [B, T_kv', D] (T_kv' may be T_kv or T_kv*N_s)

        B, T_q, _ = q.shape
        B2, T_kv_flat, D = kv.shape
        assert B == B2

        qh = self._split(self.W_q(q))             # [B, h, T_q, d]
        kh = self._split(self.W_k(kv))            # [B, h, T_kv_flat, d]
        vh = self._split(self.W_v(kv))            # [B, h, T_kv_flat, d]

        # Build importance mask aligned with source tokens (T_kv_flat)
        attn_mask = None
        if imp_weights is not None:
            if key_value.dim() == 4:  # [B, T_kv, N_s, D] flattened -> [B, T_kv*N_s, D]
                Bk, T_kv, N_s, _ = key_value.shape
                assert Bk == B
                assert imp_weights.shape == (B, N_s), f"imp_weights must be [B, N_s]=[{B}, {N_s}], got {imp_weights.shape}"
                imp_full = imp_weights.unsqueeze(1).expand(B, T_kv, N_s).reshape(B, T_kv * N_s)  # [B, T_kv*N_s]
            elif key_value.dim() == 3:
                # If user provided [B, T_kv] we can use it; otherwise ignore
                if imp_weights.shape == (B, key_value.shape[1]):
                    imp_full = imp_weights  # [B, T_kv]
                else:
                    imp_full = None
            else:
                imp_full = None

            if imp_full is not None:
                log_imp = torch.log(torch.clamp(imp_full, min=1e-6)).to(qh.dtype)  # [B, S]
                attn_mask = log_imp.view(B, 1, 1, -1)  # [B, 1, 1, S], broadcast over heads and T_q

        y = F.scaled_dot_product_attention(
            qh, kh, vh,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            # is_causal=True  # causal along the flattened time-major source axis

            # In this architecture, causality is handled by the autoregressive loop structure. 
            # The attention mechanism itself should process the entire context in parallel.
            is_causal=False,
        )  # [B, h, T_q, d]

        out = self.out(self._merge(y))  # [B, T_q, D]
        return restore_q(out)

    def step(self, query_t: torch.Tensor, key_value_t: torch.Tensor, imp_weights_t: torch.Tensor | None = None, append: bool = True) -> torch.Tensor:
        """
        Incremental attention with caching.

        query_t:
          - [B, D] or [B, 1, D] or [B, 1, N_q, D] or [B, N_q, D]  (returns matching shape)
        key_value_t:
          - [B, D] or [B, 1, D] or [B, N_s, D] or [B, 1, N_s, D]

        imp_weights_t:
          - If key_value_t provides N_s (i.e., has a sensor axis), pass [B, N_s].
            These are appended to an internal importance cache aligned with K/V tokens.
          - If kv has no sensor axis, importance is ignored (treated as ones).

        append: If False, compute with temporary KV/imp (no permanent append) for efficiency in Heun midpoint.
        """
        q, restore_q = self._to_B_T_D(query_t)    # [B, T_q, D]
        kv, _ = self._to_B_T_D(key_value_t)       # [B, T_kv', D] where T_kv' could be N_s (for sensors-at-step) or 1 (for CLS-at-step)

        B, T_q, _ = q.shape
        B2, Tk, D = kv.shape
        assert B == B2

        # Project and split
        qh = self._split(self.W_q(q))             # [B, h, T_q, d]
        k_new = self._split(self.W_k(kv))         # [B, h, Tk, d]
        v_new = self._split(self.W_v(kv))         # [B, h, Tk, d]

        # Importance for this step: detect if kv provided a sensor axis
        imp_step = None
        if imp_weights_t is not None:
            # We consider kv had a sensor axis if imp_weights_t.dim() == 2 and shapes match
            if imp_weights_t.dim() == 2 and imp_weights_t.shape[0] == B and imp_weights_t.shape[1] == Tk:
                imp_step = imp_weights_t.to(q.dtype)
        # If None, treat as ones (will be handled below)

        if append:
            # Append to caches
            if self.K is None:
                self.K = k_new
                self.V = v_new
            else:
                self.K = torch.cat([self.K, k_new], dim=2)  # concat along source length S
                self.V = torch.cat([self.V, v_new], dim=2)
            self._append_imp(imp_step, Tk=Tk, B=B, device=q.device, dtype=q.dtype)
            K_use = self.K
            V_use = self.V
            imp_use = self.imp_cache
        else:
            # Temporary (no append) for efficiency
            K_use = torch.cat([self.K, k_new], dim=2) if self.K is not None else k_new
            V_use = torch.cat([self.V, v_new], dim=2) if self.V is not None else v_new
            # Mimic _append_imp for temp imp
            imp_new = imp_step if imp_step is not None else torch.ones(B, Tk, device=q.device, dtype=q.dtype)
            imp_use = torch.cat([self.imp_cache, imp_new], dim=1) if self.imp_cache is not None else imp_new

        # Build additive attention mask from (possibly temp) cached importance
        S = K_use.size(2)
        attn_mask = None
        if imp_use is not None:
            # imp_use: [B, S] aligned with K/V positions
            log_imp = torch.log(torch.clamp(imp_use, min=1e-6)).to(qh.dtype)  # [B, S]
            attn_mask = log_imp.view(B, 1, 1, S)  # [B, 1, 1, S]

        y = F.scaled_dot_product_attention(
            qh, K_use, V_use,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False
        )  # [B, h, T_q, d]

        out = self.out(self._merge(y))  # [B, T_q, D]
        return restore_q(out)
 
class TemporalDecoderHierarchical(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layers: int = 4,
        n_heads: int = 4,
        max_len: int = 4096,
        dt: float = 0.02,
        learnable_dt: bool = False,
        dropout: float = 0.0,
        rope_base: float = 1000.0,
        checkpoint_every_layer: bool = True,
        imp_threshold: float = 0.1,  # Mask if imp < threshold

        n_window: int = 16,       # Fixed window size for history
        pooling_kernel: int = 2,  # Kernel and stride for pooling
        pooling_layers: int = 2,  # Number of pooling applications to compress (e.g., 64 -> 32 -> 16)
        # In each iteration, the time dimension reduces by a factor of k=pooling_kernel
    ):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.rope_base = rope_base
        self.imp_threshold = imp_threshold
        self.n_window = n_window
        self.pooling_kernel = pooling_kernel
        self.pooling_layers = pooling_layers

        self.cls_self_attns     = nn.ModuleList([MultiheadSoftmaxAttention(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.cls_cross_attns    = nn.ModuleList([CAU_MultiheadCrossAttention(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.sensor_cross_attns = nn.ModuleList([CAU_MultiheadCrossAttention(d_model, n_heads, dropout) for _ in range(n_layers)])

        self.pre_norms  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers * 2)])  # For CLS and sensors
        self.post_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers * 2)])

        # FIX : Change Normalization to standard Pre-LN layout: norms for Attn input and FFN input
        self.attn_norms_cls = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffn_norms_cls  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.attn_norms_sensor = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.ffn_norms_sensor  = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])

        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 4 * d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(4 * d_model, d_model),
            ) for _ in range(n_layers * 2)  # For CLS and sensors
        ])
        self.chunk_size = 4

        # Heun heads (separate for CLS and sensors)
        self.cls_head    = nn.Linear(d_model, d_model)
        self.sensor_head = nn.Linear(d_model, d_model)

        # Learnable positional embedding (time)
        self.pos_emb = nn.Parameter(torch.randn(max_len, d_model) * 0.02)

        # Learnable spatial embedding for sensors (applied per-sensor)
        self.spatial_emb = nn.Linear(d_model, d_model)  # Projects sensor features with spatial awareness

        # Pooling layers (separate for CLS and sensors; avg and max, then combine)
        self.cls_avg_pool = nn.AvgPool1d(kernel_size=pooling_kernel, stride=pooling_kernel)
        self.cls_max_pool = nn.MaxPool1d(kernel_size=pooling_kernel, stride=pooling_kernel)
        self.sensor_avg_pool = nn.AvgPool1d(kernel_size=pooling_kernel, stride=pooling_kernel)
        self.sensor_max_pool = nn.MaxPool1d(kernel_size=pooling_kernel, stride=pooling_kernel)

        # Post-transformer refinement (1D conv + FC, separate for CLS and sensors)
        conv_dim = d_model  # Assuming after pooling, time dim is small
        self.cls_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.cls_fc = nn.Linear(d_model, d_model)
        self.sensor_conv = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1)
        self.sensor_fc = nn.Linear(d_model, d_model)

        # dt handling 
        self.register_buffer("dt_const", torch.tensor(dt))
        if learnable_dt:
            self.dt_scale = nn.Parameter(torch.zeros(()))
        else:
            self.dt_scale = None

        self.use_ckpt = checkpoint_every_layer

    def apply_rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Accepts:
        x:  [B, T, M, D]  or [B, M, D] or [B, D]
        pos: (1, T, 1) integer/float time indices
        """
        assert x.dim() in (2, 3, 4), f"apply_rope: unsupported x.dim={x.dim()}"
        orig_dim = x.dim()

        # Lift to 4D: [B, T, M, D]
        if orig_dim == 2:          # [B, D] -> [B, 1, 1, D]
            x = x.unsqueeze(1).unsqueeze(1)
        elif orig_dim == 3:        # [B, M, D] -> [B, 1, M, D]
            x = x.unsqueeze(1)
        # else: already [B, T, M, D]

        B, T, M, D = x.shape
        assert D % 2 == 0, f"apply_rope: D must be even, got D={D}"
        d = D // 2

        x_dtype = x.dtype
        x = x.float()
        pos = pos.float()

        # Frequencies and angles
        freq = self.rope_base ** (-torch.arange(d, device=x.device, dtype=x.dtype) / d)  # [d]
        # pos: [1, T, 1] -> [1, T, 1, 1], broadcast with freq -> [1, T, 1, d]
        angle = pos.to(x.dtype).unsqueeze(-1) * freq.view(1, 1, 1, d)

        s, c = angle.sin(), angle.cos()
        x1, x2 = x[..., :d], x[..., d:]  # [B, T, M, d]
        xr = torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)  # [B, T, M, D]

        xr = xr.to(x_dtype)

        # Restore original rank
        if orig_dim == 2: xr = xr.squeeze(1).squeeze(1)    # -> [B, D]
        elif orig_dim == 3: xr = xr.squeeze(1)             # -> [B, M, D]

        return xr

    def _add_pos(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """
        Adds rotary position embeddings to x. Accepts x with shape:
        - [B, T, M, D]  (T steps)
        - [B, M, D]     (T=1)
        - [B, D]        (T=1, M=1)
        """
        if x.dim() == 4:
            T = x.shape[1]
        elif x.dim() in (2, 3):
            T = 1
        else:
            raise ValueError(f"_add_pos: unsupported x.dim={x.dim()}")
        
        # pos = torch.arange(offset, offset + T, device=x.device, dtype=x.dtype).view(1, T, 1)  # [1, T, 1]

        # Fix op-1: always generate positions in float32 
        # Keep sin/cos in float32, then cast back to reduces numerical error
        # pos = torch.arange(offset, offset + T, device=x.device, dtype=torch.float32).view(1, T, 1)

        # Fix op-2: use relative positions within each window to ignore offset during window processing:
        pos = torch.arange(0, T, device=x.device, dtype=torch.float32).view(1, T, 1)

        return self.apply_rope(x, pos)

    def _ensure_imp(self, imp, *, B: int, N_s: int, device, dtype=torch.float32):
        if imp is None:
            return torch.ones(B, N_s, device=device, dtype=dtype)
        return imp

    def _effective_dt(self):
        if self.dt_scale is None:
            return self.dt_const
        return self.dt_const * F.softplus(self.dt_scale)

    def _process_window(self, window: torch.Tensor, offset: int, imp: torch.Tensor) -> tuple:
        """
        window: [B, T_win, 1 + N_s, D]
        Returns pooled cls and sensors: [B, T_pooled, 1, D] and [B, T_pooled, N_s, D]
        """
        B, T_win, M, D = window.shape
        N_s = M - 1

        # Add time embeddings (RoPE)
        window = self._add_pos(window, offset)

        # Separate CLS and sensors
        cls_win = window[:, :, :1, :]      # [B, T, 1, D]
        sensors_win = window[:, :, 1:, :]  # [B, T, N_s, D]

        # Spatial embedding once (per time, per sensor)
        sensors_win = self.spatial_emb(sensors_win)

        # Apply pooling layers separately (multiple times to compress)
        for _ in range(self.pooling_layers):
            T_cur = cls_win.shape[1]
            if T_cur < self.pooling_kernel:
                break

            # ---- CLS path ----
            cls_t = cls_win.squeeze(2).permute(0, 2, 1)
            cls_avg = self.cls_avg_pool(cls_t)
            cls_max = self.cls_max_pool(cls_t)
            cls_pooled = (cls_avg + cls_max) / 2
            cls_win = cls_pooled.permute(0, 2, 1).unsqueeze(2)

            # ---- Sensors path ----
            B_, T_cur_s, N_s_, D_ = sensors_win.shape
            sensors_t = sensors_win.permute(0, 2, 3, 1).reshape(B * N_s, D, T_cur)
            sensors_avg = self.sensor_avg_pool(sensors_t)
            sensors_max = self.sensor_max_pool(sensors_t)
            sensors_pooled = (sensors_avg + sensors_max) / 2
            T_next = sensors_pooled.shape[-1]
            sensors_win = sensors_pooled.reshape(B, N_s, D, T_next).permute(0, 3, 1, 2)

        return cls_win, sensors_win

    def _block(
        self,
        idx: int,
        cls: torch.Tensor,
        sensors: torch.Tensor,
        imp: torch.Tensor,
    ) -> tuple:
        """
        Modified for parallel: cls [B, T_pooled, 1, D], sensors [B, T_pooled, N_s, D]
        Uses forward() of attentions instead of step().
        """
        B, T, _, D = cls.shape
        _, _, N_s, _ = sensors.shape

        # --- CLS stream ---
        cls_residual = cls
        cls_normed = self.pre_norms[idx](cls)
        cls_squeezed = cls_normed.squeeze(2)
        cls_attn = self.cls_self_attns[idx](cls_squeezed).unsqueeze(2)
        cls_cross = self.cls_cross_attns[idx](cls_normed, sensors, imp_weights=imp)
        cls = cls_attn + cls_cross
        cls = self.post_norms[idx](cls_residual + cls)
        cls = cls + self.ffns[idx](cls)

        # --- Sensor stream ---
        # --- Simplified sensor stream logic for clarity ---
        sensors_residual = sensors
        sensors_normed = self.pre_norms[idx + self.n_layers](sensors)
        sensors_cross = self.sensor_cross_attns[idx](sensors_normed, cls, imp_weights=None)
        sensors = sensors_cross # No self-attention was being used
        sensors = self.post_norms[idx + self.n_layers](sensors_residual + sensors)
        sensors = sensors + self.ffns[idx + self.n_layers](sensors)

        return cls, sensors

    # Revised: LayerNorm placement to prevent gradient explosion
    def __block(self, idx: int, cls: torch.Tensor, sensors: torch.Tensor, imp: torch.Tensor) -> tuple:
        """
        Robust Pre-LN Transformer Block.
        Structure: x = x + F(Norm(x))
        """
        # --- CLS Stream ---
        # 1. Attention Block (Pre-LN)
        cls_norm = self.attn_norms_cls[idx](cls)
        cls_sq = cls_norm.squeeze(2)
        
        # Self Attention + Cross Attention
        cls_attn = self.cls_self_attns[idx](cls_sq).unsqueeze(2)
        cls_cross = self.cls_cross_attns[idx](cls_norm, sensors, imp_weights=imp)
        
        # Residual Connection 1
        cls = cls + cls_attn + cls_cross
        
        # 2. FFN Block (Pre-LN)
        cls_norm_ffn = self.ffn_norms_cls[idx](cls)
        cls = cls + self.ffns[idx](cls_norm_ffn)

        # --- Sensor Stream ---
        # 1. Cross Attention Block (Pre-LN)
        sens_norm = self.attn_norms_sensor[idx](sensors)
        sens_cross = self.sensor_cross_attns[idx](sens_norm, cls, imp_weights=None)
        
        # Residual Connection 1
        sensors = sensors + sens_cross
        
        # 2. FFN Block (Pre-LN)
        sens_norm_ffn = self.ffn_norms_sensor[idx](sensors)
        sensors = sensors + self.ffns[idx + self.n_layers](sens_norm_ffn)

        return cls, sensors

    def _block_ckpt(self, idx: int, cls: torch.Tensor, sensors: torch.Tensor, imp: torch.Tensor) -> tuple:
        if not (self.use_ckpt and self.training):
            return self._block(idx, cls, sensors, imp)

        def fn(c, s, i):
            return self._block(idx, c, s, i)
        return cp.checkpoint(fn, cls, sensors, imp, use_reentrant=False)

    def _refine(self, cls: torch.Tensor, sensors: torch.Tensor) -> tuple:
        B, T, _, D = cls.shape
        _, T2, N_s, D2 = sensors.shape
        assert T2 == T and D2 == D, f"Shape mismatch: cls T={T}, sensors T={T2}; D={D}, sensors D={D2}"

        # ---- CLS path ----
        cls_t = cls.squeeze(2).permute(0, 2, 1)
        cls_ref = F.relu(self.cls_conv(cls_t))
        cls_ref = cls_ref.permute(0, 2, 1).unsqueeze(2)
        cls_ref = self.cls_fc(cls_ref)

        # ---- Sensors path ----
        sensors_t = sensors.permute(0, 2, 3, 1).reshape(B * N_s, D, T)
        sensors_ref = F.relu(self.sensor_conv(sensors_t))
        sensors_ref = sensors_ref.permute(0, 2, 1)
        sensors_ref = sensors_ref.reshape(B, N_s, T, D)
        sensors_ref = sensors_ref.permute(0, 2, 1, 3)
        sensors_ref = self.sensor_fc(sensors_ref)

        return cls_ref, sensors_ref

    def forward(self, x_seq: torch.Tensor, imp: torch.Tensor | None = None) -> torch.Tensor:
        """
        Non-autoregressive forward pass.
        x_seq: [B, T, 1 + N_s, D]
        imp: [B, N_s]
        """
        B, T, M, D = x_seq.shape
        N_s = M - 1
        imp = self._ensure_imp(imp, B=B, N_s=N_s, device=x_seq.device, dtype=x_seq.dtype)

        cls_pooled, sensors_pooled = self._process_window(x_seq, 0, imp)

        for idx in range(self.n_layers):
            cls_pooled, sensors_pooled = self._block_ckpt(idx, cls_pooled, sensors_pooled, imp)

        cls_pooled, sensors_pooled = self._refine(cls_pooled, sensors_pooled)

        dt = self._effective_dt()
        k1_cls = self.cls_head(cls_pooled)
        k1_sensors = self.sensor_head(sensors_pooled)
        cls_mid = cls_pooled + dt * k1_cls
        sensors_mid = sensors_pooled + dt * k1_sensors
        
        cls_mid_proc, sensors_mid_proc = self._process_window(cls_mid, 0, imp) # Reprocess midpoint
        for idx in range(self.n_layers):
            cls_mid_proc, sensors_mid_proc = self._block(idx, cls_mid_proc, sensors_mid_proc, imp)
        cls_mid_y, sensors_mid_y = self._refine(cls_mid_proc, sensors_mid_proc)

        k2_cls = self.cls_head(cls_mid_y)
        k2_sensors = self.sensor_head(sensors_mid_y)
        cls_next = cls_pooled + 0.5 * dt * (k1_cls + k2_cls)
        sensors_next = sensors_pooled + 0.5 * dt * (k1_sensors + k2_sensors)

        return torch.cat([cls_next, sensors_next], dim=2)
    
    # --- Added a unified private method for autoregressive rollout ---
    def _rollout_loop(
        self,
        obs_window: torch.Tensor,
        N_fore: int,
        imp: torch.Tensor,
        *,
        is_training: bool,
        truncate_k: int | None = None,
        teacher_force_seq: torch.Tensor | None = None,
        teacher_force_prob: float = 0.0,
    ) -> torch.Tensor:

        B, T_obs, M, D = obs_window.shape
        N_s = M - 1
        dev = obs_window.device
        dt = self._effective_dt()
        
        # Reset state of all attention modules to prevent cache accumulation across epochs
        for m in self.cls_cross_attns:
            m.reset_state(B, obs_window.device)
        for m in self.sensor_cross_attns:
            m.reset_state(B, obs_window.device)

        # Also reset self-attention modules if they have step-caches
        for m in self.cls_self_attns:
            if hasattr(m, "reset_state"):
                m.reset_state(B, obs_window.device)

        history = deque(maxlen=self.n_window)
        for t in range(T_obs):
            history.append(obs_window[:, t])  # Each element is [B, 1 + N_s, D]

        # Start with the observed part of the trajectory
        outputs = [obs_window]
        steps_left = N_fore - T_obs
        if steps_left <= 0:
            return torch.cat(outputs, dim=1)[:, :N_fore]

        # Initial state for the ODE is the last observed token
        latent_cur = history[-1]
        cls_cur = latent_cur[:, :1, :]
        sensors_cur = latent_cur[:, 1:, :]

        for step in range(steps_left):

            # current_time_offset = T_obs + step
            # # Form the context window from the history deque.
            win_tensor = torch.stack(list(history), dim=1)
            # pos_offset = current_time_offset - win_tensor.shape[1] + 1

            # Fix: We explicitly set offset to 0. 
            # The model will always perceive the input window as positions [0, 1, ... W-1].
            # This ensures the embedding distribution matches training regardless of how far we rollout.
            pos_offset = 0 

            # Process the context window to get derivatives (k1)
            # This pipeline is run once per step on the full window.
            cls_win, sensors_win = self._process_window(win_tensor, pos_offset, imp)

            for idx in range(self.n_layers):
                if is_training:
                    cls_win, sensors_win = self._block_ckpt(idx, cls_win, sensors_win, imp)
                else:
                    cls_win, sensors_win = self._block(idx, cls_win, sensors_win, imp)
            cls_y, sensors_y = self._refine(cls_win, sensors_win)

            # Use the last time step of the processed window as the basis for the derivative
            k1_cls = self.cls_head(cls_y[:, -1:, :, :]).squeeze(1)
            k1_sensors = self.sensor_head(sensors_y[:, -1:, :, :]).squeeze(1)

            # Euler update (replaces Heun's midpoint and k2 calculation)
            cls_next = cls_cur + dt * k1_cls
            sensors_next = sensors_cur + dt * k1_sensors

            # ---- Correction ----
            # Using Heun's update

            # cls_pred = cls_cur + dt * k1_cls
            # sensors_pred = sensors_cur + dt * k1_sensors
            # latent_pred = torch.cat([cls_pred, sensors_pred], dim=1)

            # # Create temporary window with Euler prediction at the end
            # temp_history_list = list(history)
            # if len(temp_history_list) >= self.n_window:
            #     temp_history_list.pop(0) # Remove oldest
            # temp_history_list.append(latent_pred) # Add Euler prediction
            # win_tensor_k2 = torch.stack(temp_history_list, dim=1)

            # # Process window for k2
            # cls_win_k2, sensors_win_k2 = self._process_window(win_tensor_k2, 0, imp)
            # for idx in range(self.n_layers):
            #     if is_training:
            #         cls_win_k2, sensors_win_k2 = self._block_ckpt(idx, cls_win_k2, sensors_win_k2, imp)
            #     else:
            #         cls_win_k2, sensors_win_k2 = self._block(idx, cls_win_k2, sensors_win_k2, imp)
            # cls_y_k2, sensors_y_k2 = self._refine(cls_win_k2, sensors_win_k2)
            # # Calculate k2 slopes
            # k2_cls = self.cls_head(cls_y_k2[:, -1:, :, :]).squeeze(1)
            # k2_sensors = self.sensor_head(sensors_y_k2[:, -1:, :, :]).squeeze(1)
            # # Heun Update (Average of slopes)
            # cls_next = cls_cur + 0.5 * dt * (k1_cls + k2_cls)
            # sensors_next = sensors_cur + 0.5 * dt * (k1_sensors + k2_sensors)

            # ---- ----- ---- ----

            latent_next = torch.cat([cls_next, sensors_next], dim=1)
            # Teacher Forcing (if training)
            if is_training and teacher_force_seq is not None and step < teacher_force_seq.size(1):
                if torch.rand((), device=dev) < teacher_force_prob:
                    # print(f'Using teacher forcing at step {step}\n')
                    latent_next = teacher_force_seq[:, step]

            # Update history and outputs
            history.append(latent_next)
            outputs.append(latent_next.unsqueeze(1))

            # Truncated Backpropagation Through Time (BPTT)
            if is_training and truncate_k and ((step + 1) % truncate_k == 0):
                history = deque([h.detach() for h in history], maxlen=self.n_window)
                latent_cur = history[-1] # Re-reference after detaching
            else:
                latent_cur = latent_next

            cls_cur = latent_cur[:, :1, :]
            sensors_cur = latent_cur[:, 1:, :]

            # Truncated BPTT
            # if is_training and truncate_k and ((step + 1) % truncate_k == 0):
            #     history = deque([h.detach() for h in history], maxlen=self.n_window)
            #     # Re-bind current state to detached history to break graph
            #     latent_cur_detached = history[-1]
            #     cls_cur = latent_cur_detached[:, :1, :]
            #     sensors_cur = latent_cur_detached[:, 1:, :]

        return torch.cat(outputs, dim=1)

    # --- `rollout_with_grad` is now a wrapper around `_rollout_loop` ---
    def rollout_with_grad(
        self,
        obs_window: torch.Tensor,
        N_fore: int,
        imp: torch.Tensor | None = None,
        *,
        truncate_k: int | None = 64,
        teacher_force_seq: torch.Tensor | None = None,
        teacher_force_prob: float = 0.0,
    ) -> torch.Tensor:
        assert self.training, "Call only in training mode"
        B, T_obs, M, D = obs_window.shape
        N_s = M - 1
        imp = self._ensure_imp(imp, B=B, N_s=N_s, device=obs_window.device, dtype=obs_window.dtype)

        return self._rollout_loop(
            obs_window=obs_window,
            N_fore=N_fore,
            imp=imp,
            is_training=True,
            truncate_k=truncate_k,
            teacher_force_seq=teacher_force_seq,
            teacher_force_prob=teacher_force_prob,
        )

    # --- `generate` is now a wrapper around `_rollout_loop` ---
    @torch.no_grad()
    def generate(self, obs_window: torch.Tensor, N_fore: int, imp: torch.Tensor | None = None) -> torch.Tensor:
        B, T_obs, M, D = obs_window.shape
        N_s = M - 1
        imp = self._ensure_imp(imp, B=B, N_s=N_s, device=obs_window.device, dtype=obs_window.dtype)

        return self._rollout_loop(
            obs_window=obs_window,
            N_fore=N_fore,
            imp=imp,
            is_training=False,
        )

    def _forward_no_ckpt(self, cls: torch.Tensor, sensors: torch.Tensor, imp: torch.Tensor) -> tuple:
        for idx in range(self.n_layers):
            cls, sensors = self._block(idx, cls, sensors, imp)
        return cls, sensors

# ================================================================
# Temporal Decoder Adapter for all modules
# ================================================================
class TemporalDecoderAdapter(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def forward_autoreg(
        self,
        G_latent          : torch.Tensor,      # (B, T_obs, D)
        N_Fore            : int,
        N_window          : int,
        *,
        imp: torch.Tensor | None = None,       # [B, N_s]
        truncate_k        : int | None = 32,
        teacher_force_seq : torch.Tensor | None = None,
        teacher_force_prob: float = 0.0,
    ):
        """
        Pure autoregressive rollout that *keeps* gradients.
        Call this only with model.train() set.
        """
        assert self.training, "Use only in training mode"
        if imp is not None:
            output = self.core.rollout_with_grad(
                obs_window        = G_latent[:, :N_window],
                N_fore            = N_Fore,
                imp               = imp,
                truncate_k        = truncate_k,
                teacher_force_seq = teacher_force_seq,
                teacher_force_prob= teacher_force_prob,
            )
        else:
            output = self.core.rollout_with_grad(
                obs_window        = G_latent[:, :N_window],
                N_fore            = N_Fore,
                truncate_k        = truncate_k,
                teacher_force_seq = teacher_force_seq,
                teacher_force_prob= teacher_force_prob,
            )       

        # Handle if output is (traj, traj_logvar) or just traj
        if isinstance(output, tuple) and len(output) == 2:
            return output  # (traj, traj_logvar)
        elif isinstance(output, torch.Tensor):
            return output, None  # (traj, None)
        else:
            raise ValueError(f"Unexpected output from core.rollout_with_grad: {type(output)}")

    def forward(
            self, G_latent: torch.Tensor, 
            N_Fore: int, 
            N_window: int, 
            imp: torch.Tensor | None = None,  # [B, N_s]
            ):
        
        if imp is not None:
            output = self.core.generate(G_latent[:, :N_window], N_Fore, imp)
        else:
            output = self.core.generate(G_latent[:, :N_window], N_Fore)
        
        # Handle if output is (traj, traj_logvar) or just traj
        if isinstance(output, tuple) and len(output) == 2:
            return output  # (traj, traj_logvar)
        elif isinstance(output, torch.Tensor):
            return output, None  # (traj, None)
        else:
            raise ValueError(f"Unexpected output from core.generate: {type(output)}")

# ---------------------------------------------------------------------
# Domain Adaptive Reconstructor with soft boundaries
# ---------------------------------------------------------------------
class SoftDomainAdaptiveReconstructor(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        N_channels: int,
        latent_tokens : int,

        num_freqs: int = 64,
        dropout: float = 0.0,

        overlap_ratio: float = 0.02,
        importance_scale: float = 0.50,

        bandwidth_init: float = 0.05,
        top_k: int | None = None,
        per_sensor_sigma: bool = False,
        CalRecVar: bool = False, 
        retain_cls: bool = False,
        retain_lat: bool = False,
        use_checkpoint: bool = True,
        USE_FINAL_MLP: bool = False,

        # --- phi incorporation toggles (set to True for combined use) ---
        use_weighted_fusion: bool = False,   # Toggle Weighted Fusion in Aggregation
        phi_scale: float = 0.5,              # Tunable scale for phi modulation (to avoid over-amplification)
    ):
        super().__init__()

        # ------------------- positional encoder -------------------------
        self.pe = FourierEmbedding(in_dim=2,  
                                   num_frequencies=num_freqs,
                                   learnable=True,
                                   sigma=10.0)

        self.coord_proj = nn.Linear(self.pe.B.shape[1] * 2, d_model)
        self.lat_proj = nn.Linear(d_model, d_model)

        # ------------------ cross-attention --------------------
        self.cross_attn = CrossAttention(d_model, num_heads, dropout)
        self.norm = nn.LayerNorm(d_model)

        # 1-hidden-layer MLP 
        self.mlp = GEGLU(d_model, mult=4)
        self.head = nn.Linear(d_model, N_channels)

        self.d_model       = d_model
        self.N_channels    = N_channels
        self.latent_tokens = latent_tokens

        self.overlap_ratio    = overlap_ratio
        self.importance_scale = importance_scale
        self.CalRecVar = CalRecVar

        self.block_size        = 256         
        self.top_k             = top_k
        self.per_sensor_sigma  = per_sensor_sigma
        self.register_parameter("log_sigma", None)  
        self.bandwidth_init    = bandwidth_init
        self._prev_S           = None

        self.coord_norm = nn.RMSNorm(d_model)
        self.agg_norm   = nn.RMSNorm(d_model)
        self.mlp_norm   = nn.RMSNorm(d_model)

        # flag for retaining CLS and hierarchical reconstruction
        self.retain_cls = retain_cls
        self.retain_lat = retain_lat # Unused
        self.USE_FINAL_MLP = USE_FINAL_MLP

        if self.retain_cls: 
            self.fusion_proj = nn.Linear(2 * d_model, d_model)  # Projects concatenated [local + CLS] back to d_model
            self.cls_gain = nn.Parameter(torch.zeros(d_model))
            self.SpatiallyAwareCLSFusion = SpatiallyAwareCLSFusion(d_model)

        if CalRecVar:
            self.stats_dim = 4  # mean_d, std_d, effective_K, mean_phi
            self.stats_proj = nn.Linear(self.stats_dim, 16)  # Project to embed dim; concatenate to x
            self.var_head = nn.Linear(self.d_model + 16, self.N_channels)  # Input now larger

        # --- params for phi incorporation ---
        self.use_weighted_fusion = use_weighted_fusion
        self.phi_scale = phi_scale  # Scale factor to control phi's influence
        self.use_checkpoint = use_checkpoint

    @staticmethod
    @torch.jit.ignore
    def _topk_aggregate(lat_proj:  torch.Tensor,   # (B,T,S,D)
                        top_idx:   torch.Tensor,   # (B,P,K)
                        weights_k: torch.Tensor,   # (B,P,K)
                        d_k: torch.Tensor,         # (B,P,K)
                        phi_k: torch.Tensor,       # (B,P,K)
                        valid_k: torch.Tensor,     # (B,P,K)
                        S: int) -> tuple[torch.Tensor, torch.Tensor]:

        B, T, _, D = lat_proj.shape
        _, P, K    = top_idx.shape
        dev        = lat_proj.device

        w = torch.zeros(B, P, S, device=dev, dtype=lat_proj.dtype)
        w.scatter_(2, top_idx, weights_k)

        h = torch.einsum('btsd,bps->btpd', lat_proj, w)

        mask = valid_k.float()
        effective_K = mask.sum(dim=-1, keepdim=True)
        
        safe_denom = effective_K + 1e-8
        mean_d = (d_k * weights_k * mask).sum(dim=-1, keepdim=True) / safe_denom
        var_d = ((d_k - mean_d)**2 * weights_k * mask).sum(dim=-1, keepdim=True) / safe_denom
        std_d = torch.sqrt(var_d) # No need for eps here if var_d is non-negative
        mean_phi = (phi_k * weights_k * mask).sum(dim=-1, keepdim=True) / safe_denom

        stats = torch.cat([mean_d, std_d, effective_K, mean_phi], dim=-1)  
        return h, stats

    def forward(self,
                z: torch.Tensor,                # [B, T, S, D_raw] or [B, T, S+1, D_raw] if retain_cls=True
                Y: torch.Tensor,                # [B, P, 2/3]
                sensor_coords: torch.Tensor,    # [B, S, 2/3]
                mask: torch.Tensor,             # [B, T, S] (if retain_cls=True, this is for S+1; we slice below)
                phi_mean: torch.Tensor | None = None,

                padding_mask: torch.Tensor | None = None
                ) -> torch.Tensor:

        B, T, S_or_Sp1, D_raw = z.shape
        P              = Y.size(1)
        C              = self.N_channels
        dev            = z.device
        d_model        = self.lat_proj.out_features
        S = sensor_coords.size(1)  # true number of sensors (excludes CLS)

        if phi_mean is None: phi_mean = torch.ones(B, S, device=dev)  # [B,S]

        if self.per_sensor_sigma:
            if (self.log_sigma is None) or (self.log_sigma.numel() != S):
                self.log_sigma = nn.Parameter(torch.full((S,),
                                            math.log(self.bandwidth_init),
                                            device=dev))
            sigma = self.log_sigma.exp()  # (S,)
        else:
            sigma = torch.tensor(self.bandwidth_init, device=dev)

        # Combine mask and padding_mask; slice out CLS if present later
        effective_mask = mask
        if padding_mask is not None:
            padding_bt = padding_mask.unsqueeze(1).expand(-1, T, -1)  # [B,T,S or S+1]
            effective_mask = mask & padding_bt
        # Project tokens
        lat_proj = self.lat_proj(z)  # (B, T, S or S+1, d_model)

        # --- Split global and local tokens ---
        if self.retain_cls:
            assert lat_proj.size(2) == S + 1, "retain_cls=True requires z to include CLS at index 0."
            cls_proj    = lat_proj[:, :, 0, :]      # (B,T,d)
            sensor_proj = lat_proj[:, :, 1:, :]     # (B,T,S,d)
            sensor_mask = effective_mask[:, :, 1:]  # (B,T,S)
        else:
            global_latents = None
            sensor_proj = lat_proj                  # [B, T, S, D]
            sensor_mask = effective_mask            # [B, T, S]

        # Positional tokens for queries
        coord_tok = self.coord_proj(self.pe(Y))     # (B,P,d)
        coord_tok = self.coord_norm(coord_tok)      # (B,P,d)
        coord_tok = coord_tok.unsqueeze(1).expand(B, T, P, d_model).reshape(B*T, P, d_model)  # (B*T,P,d)
        # Distance and importance scaling
        d = torch.cdist(Y, sensor_coords)  # (B,P,S)

        # Build a [B,S] validity mask for sensors (time-invariant proxy from t=0)
        sensor_valid_bs = sensor_mask[:, 0, :]  # [B,S] (True if valid at t=0)
        if padding_mask is not None and padding_mask.dim() == 2:
            sensor_valid_bs = sensor_valid_bs & padding_mask  # [B,S]

        # Set distances to inf for invalid sensors so they never get into top-k
        gamma = self.importance_scale
        d = d.masked_fill(~sensor_valid_bs.unsqueeze(1), float('inf'))  # [B,P,S]
        phi   = phi_mean.detach()                                       # [B,S]
        d_scaled = d / (phi[:, None, :] ** gamma + 1e-6)                # [B,P,S]

        # ------------------------------------------------------------
        # Unified top-k aggregation (K_eff = min(K, number_of_valid_sensors))
        # If self.top_k is None, K_eff = S (i.e., use all valid sensors)
        # ------------------------------------------------------------
        local_S = sensor_proj.size(2)  # == S
        if self.top_k is None:
            K_eff = local_S
        else:
            K_eff = min(self.top_k, local_S)

        # Top-k over smallest distances; if K_eff==S this is equivalent to "all sensors"
        # For rows where many sensors are invalid (inf), torch.topk returns the S finite ones first anyway.
        _, top_idx = torch.topk(d_scaled, K_eff, dim=2, largest=False)  # (B,P,K_eff)
        d_k = torch.gather(d_scaled, 2, top_idx)                        # (B,P,K_eff)

        # Gather phi and per-sensor sigma (if used) at the same indices
        phi_expan = phi[:, None, :].expand(-1, P, -1)                     # (B,P,S)
        phi_k     = torch.gather(phi_expan, 2, top_idx)                   # (B,P,K_eff)

        if self.per_sensor_sigma:
            sigma_k = sigma[top_idx]                                     # (B,P,K_eff)
        else:
            sigma_k = sigma                                              # scalar

        scores = -d_k / sigma_k                                          # (B,P,K_eff)
        scores -= scores.max(dim=-1, keepdim=True).values
        exp    = torch.exp(scores)

        # Validity at selected indices (time-invariant proxy from t=0)
        valid_k = torch.gather(sensor_valid_bs.unsqueeze(1).expand(-1, P, -1), 2, top_idx)  # (B,P,K_eff), bool
        exp = exp * valid_k.float()
        weights = exp / (exp.sum(dim=-1, keepdim=True) + 1e-6)  # (B,P,K_eff)

        # --- Combined Phi Incorporation (Weighted Fusion + Attention Modulation) ---
        # Weighted Fusion (multiply phi into weights, then re-normalize)
        if self.use_weighted_fusion:
            weights.mul_(phi_k * self.phi_scale)  # In-place multiplication
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)  # Re-normalize

        h, stats = self._topk_aggregate(sensor_proj, top_idx, weights, d_k, phi_k, valid_k, local_S)
        if not self.CalRecVar:
            stats = None # Explicitly set to None if not used

        h = self.agg_norm(h)              # (B,T,P,d)
        lat = h.reshape(B*T, P, d_model)  # (B*T,P,d)
        # Here, lat - [B*T, P, d] is the result of the top-k aggregation. 
        # For each query point, lat holds a feature as a weighted average of the features from its K nearest sensors.

        # this block re-purposes components of a cross-attention layer
        # to act as a two-layer linear transformation block.
        if hasattr(self.cross_attn, 'to_v'):
            v_proj  = self.cross_attn.to_v
            out_proj = self.cross_attn.out
        else:
            mha = self.cross_attn.attn
            dim = mha.embed_dim
            v_weight = mha.in_proj_weight[2 * dim : 3 * dim]
            v_bias   = mha.in_proj_bias[2 * dim : 3 * dim] if mha.in_proj_bias is not None else None
            def v_proj(x): return F.linear(x, v_weight, v_bias)
            out_proj = mha.out_proj

        local_lat = out_proj(v_proj(lat))   # (B*T,P,d)
        # ------------------------------------------------------------------------------
        
        local_out_mean = None
        local_pre      = coord_tok + local_lat    # (B*T,P,d)

        if self.retain_cls:

            fused_pre = self.SpatiallyAwareCLSFusion( local_pre, cls_proj.reshape(B*T, 1, d_model)) 

            fused_x = self.norm(fused_pre)
            if self.USE_FINAL_MLP:
                fused_x = fused_x + self.mlp(fused_x)
            out_mean = self.head(fused_x).view(B, T, P, C)
            x_for_var = fused_x

        else:

            local_x = self.norm(local_pre)
            if self.USE_FINAL_MLP:
                local_x = local_x + self.mlp(local_x)
            local_out_mean = self.head(local_x).view(B, T, P, C)
            out_mean = local_out_mean
            x_for_var = local_x

        # Optional variance head
        out_logvar = None
        if self.CalRecVar:
            stats_emb = self.stats_proj(stats)                     
            stats_emb = stats_emb.unsqueeze(1).expand(-1, T, -1, -1).reshape(B*T, P, -1)
            x_var = torch.cat([x_for_var, stats_emb], dim=-1)      
            logvar = self.var_head(x_var)
            out_logvar = logvar.view(B, T, P, C)

        return local_out_mean, out_mean, out_logvar

    # Revised: Cross-att version
    def _forward(
        self,
        z: torch.Tensor,                # [B, T, S, D_raw] or [B, T, S+1, D_raw] if retain_cls=True
        Y: torch.Tensor,                # [B, P, 2/3]
        sensor_coords: torch.Tensor,    # [B, S, 2/3]
        mask: torch.Tensor,             # [B, T, S] (if retain_cls=True, this is for S+1; we slice below)
        phi_mean: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None
    ):
        B, T, S_or_Sp1, D_raw = z.shape
        P       = Y.size(1)
        C       = self.N_channels
        dev     = z.device
        d_model = self.lat_proj.out_features
        S       = sensor_coords.size(1)  # true number of sensors (excludes CLS)

        if phi_mean is None:
            phi_mean = torch.ones(B, S, device=dev)  # [B,S]

        # -------- sigma --------
        if self.per_sensor_sigma:
            if (self.log_sigma is None) or (self.log_sigma.numel() != S):
                self.log_sigma = nn.Parameter(
                    torch.full((S,), math.log(self.bandwidth_init), device=dev)
                )
            sigma = self.log_sigma.exp()  # (S,)
        else:
            sigma = torch.tensor(self.bandwidth_init, device=dev)

        # -------- combine masks --------
        effective_mask = mask
        if padding_mask is not None:
            padding_bt = padding_mask.unsqueeze(1).expand(-1, T, -1)  # [B,T,S or S+1]
            effective_mask = mask & padding_bt

        # -------- project tokens --------
        lat_proj = self.lat_proj(z)  # (B, T, S or S+1, d_model)

        # -------- split CLS vs sensors --------
        if self.retain_cls:
            assert lat_proj.size(2) == S + 1, "retain_cls=True requires z to include CLS at index 0."
            cls_proj    = lat_proj[:, :, 0, :]      # (B,T,d)
            sensor_proj = lat_proj[:, :, 1:, :]     # (B,T,S,d)
            sensor_mask = effective_mask[:, :, 1:]  # (B,T,S)
        else:
            cls_proj    = None
            sensor_proj = lat_proj                 # (B,T,S,d)
            sensor_mask = effective_mask           # (B,T,S)

        # -------- build coord tokens --------
        coord_tok = self.coord_proj(self.pe(Y))          # (B,P,d)
        coord_tok = self.coord_norm(coord_tok)           # (B,P,d)
        coord_tok = coord_tok.unsqueeze(1).expand(B, T, P, d_model).reshape(B*T, P, d_model)  # (B*T,P,d)

        # -------- compute distances + top-k neighborhood --------
        d = torch.cdist(Y, sensor_coords)  # (B,P,S)

        sensor_valid_bs = sensor_mask[:, 0, :]  # [B,S] (proxy validity from t=0)
        if padding_mask is not None and padding_mask.dim() == 2:
            sensor_valid_bs = sensor_valid_bs & padding_mask  # [B,S]

        gamma = self.importance_scale
        d = d.masked_fill(~sensor_valid_bs.unsqueeze(1), float('inf'))  # [B,P,S]
        phi = phi_mean.detach()                                         # [B,S]
        d_scaled = d / (phi[:, None, :] ** gamma + 1e-6)                # [B,P,S]

        local_S = sensor_proj.size(2)  # == S
        if self.top_k is None:
            K_eff = local_S
        else:
            K_eff = min(self.top_k, local_S)

        _, top_idx = torch.topk(d_scaled, K_eff, dim=2, largest=False)   # (B,P,K)
        d_k = torch.gather(d_scaled, 2, top_idx)                         # (B,P,K)

        phi_expan = phi[:, None, :].expand(-1, P, -1)                    # (B,P,S)
        phi_k     = torch.gather(phi_expan, 2, top_idx)                  # (B,P,K)

        # validity at selected indices (time-invariant proxy from t=0)
        valid_k = torch.gather(
            sensor_valid_bs.unsqueeze(1).expand(-1, P, -1),
            2,
            top_idx
        )  # (B,P,K), bool

        # -------- stats for CalRecVar (keep original kernel weights, but do NOT aggregate features) --------
        stats = None
        if self.CalRecVar:
            if self.per_sensor_sigma:
                sigma_k = sigma[top_idx]          # (B,P,K)
            else:
                sigma_k = sigma                   # scalar

            scores = -d_k / sigma_k
            scores -= scores.max(dim=-1, keepdim=True).values
            exp = torch.exp(scores) * valid_k.float()
            weights = exp / (exp.sum(dim=-1, keepdim=True) + 1e-6)  # (B,P,K)

            if self.use_weighted_fusion:
                weights = weights * (phi_k * self.phi_scale)
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

            mask_f = valid_k.float()
            effective_K = mask_f.sum(dim=-1, keepdim=True)
            denom = effective_K + 1e-8

            mean_d  = (d_k   * weights * mask_f).sum(dim=-1, keepdim=True) / denom
            var_d   = ((d_k - mean_d) ** 2 * weights * mask_f).sum(dim=-1, keepdim=True) / denom
            std_d   = torch.sqrt(var_d)
            mean_phi = (phi_k * weights * mask_f).sum(dim=-1, keepdim=True) / denom

            stats = torch.cat([mean_d, std_d, effective_K, mean_phi], dim=-1)  # (B,P,4)

        # ======================================================================
        # NEW: local cross-attention instead of weighted sum aggregation
        #   - Queries: coord_tok
        #   - Keys/Values: Top-K nearest sensor_proj tokens per query
        # ======================================================================

        # gather local KV tokens: sensor_proj is (B,T,S,d)
        # make it (B,T,P,S,d) then gather along S using top_idx
        sensor_proj_exp = sensor_proj.unsqueeze(2).expand(-1, -1, P, -1, -1)  # (B,T,P,S,d)
        top_idx_bt = top_idx.unsqueeze(1).expand(-1, T, -1, -1)               # (B,T,P,K)
        top_idx_bt = top_idx_bt.unsqueeze(-1).expand(-1, -1, -1, -1, d_model) # (B,T,P,K,d)
        kv = torch.gather(sensor_proj_exp, 3, top_idx_bt)                     # (B,T,P,K,d)

        # optional: scale KV by phi_k (keeps "phi" influence, but now via token magnitudes)
        if self.use_weighted_fusion:
            phi_k_bt = phi_k.unsqueeze(1).expand(-1, T, -1, -1)               # (B,T,P,K)
            kv = kv * (phi_k_bt.unsqueeze(-1) * self.phi_scale)

        # reshape so each query point is its own attention batch element
        q_pt  = coord_tok.reshape(B*T*P, 1, d_model)     # (B*T*P,1,d)
        kv_pt = kv.reshape(B*T*P, K_eff, d_model)        # (B*T*P,K,d)

        valid_pt = valid_k.unsqueeze(1).expand(-1, T, -1, -1).reshape(B*T*P, K_eff)  # (B*T*P,K)
        key_padding_mask = ~valid_pt  # True = ignore

        # run attention using the underlying MultiheadAttention
        if hasattr(self.cross_attn, "attn") and isinstance(self.cross_attn.attn, nn.MultiheadAttention):
            mha = self.cross_attn.attn
            if getattr(mha, "batch_first", False):
                attn_out, _ = mha(q_pt, kv_pt, kv_pt, key_padding_mask=key_padding_mask, need_weights=False)
            else:
                # convert to (L, N, E)
                q_t  = q_pt.transpose(0, 1)   # (1, B*T*P, d)
                kv_t = kv_pt.transpose(0, 1)  # (K, B*T*P, d)
                attn_out, _ = mha(q_t, kv_t, kv_t, key_padding_mask=key_padding_mask, need_weights=False)
                attn_out = attn_out.transpose(0, 1)  # back to (B*T*P,1,d)
        else:
            attn_out = self.cross_attn(q_pt, kv_pt)

        # restore to (B*T, P, d), then apply existing agg_norm
        attn_out = attn_out.reshape(B*T, P, d_model)
        attn_out = self.agg_norm(attn_out.reshape(B, T, P, d_model)).reshape(B*T, P, d_model)

        # merge attended local info into the coord token (residual-style)
        local_pre = coord_tok + attn_out  # (B*T,P,d)

        # -------- CLS fusion (unchanged) --------
        local_out_mean = None
        if self.retain_cls:
            fused_pre = self.SpatiallyAwareCLSFusion(local_pre, cls_proj.reshape(B*T, 1, d_model))
            fused_x   = self.norm(fused_pre)
            out_mean  = self.head(fused_x).view(B, T, P, C)
            x_for_var = fused_x
        else:
            local_x = self.norm(local_pre)
            local_out_mean = self.head(local_x).view(B, T, P, C)
            out_mean  = local_out_mean
            x_for_var = local_x

        # -------- variance head (unchanged; uses stats computed above) --------
        out_logvar = None
        if self.CalRecVar:
            stats_emb = self.stats_proj(stats)  # (B,P,16)
            stats_emb = stats_emb.unsqueeze(1).expand(-1, T, -1, -1).reshape(B*T, P, -1)
            x_var = torch.cat([x_for_var, stats_emb], dim=-1)
            logvar = self.var_head(x_var)
            out_logvar = logvar.view(B, T, P, C)

        return local_out_mean, out_mean, out_logvar

# ==============================
# Complete Model wrapper
# ==============================
class GLU_Bay_DD(nn.Module):
    def __init__(self,
                 cfg: dict,
                 fieldencoder: nn.Module,
                 temporaldecoder: nn.Module,
                 fielddecoder: nn.Module,

                 delta_t: float,
                 N_window: int,
                 CheckPhi: bool = False,
                 stage: int = -1,

                 use_adaptive_selection: bool  = False,
                 CalRecVar             : bool  = False, 
                 retain_cls            : bool  = False, 
                 retain_lat            : bool  = False,
                 Use_imp_in_dyn        : bool  = False,
                 ):

        super().__init__()
        self.cfg                    = cfg
        self.stage                  = stage

        self.fieldencoder           = fieldencoder
        self.temporaldecoder        = temporaldecoder
        self.TemporalDecoderAdapter = TemporalDecoderAdapter(temporaldecoder)
        self.decoder                = fielddecoder
        self.retain_cls             = retain_cls
        self.Use_imp_in_dyn         = Use_imp_in_dyn

        self.delta_t                = delta_t
        self.N_window               = N_window
        self.CheckPhi               = CheckPhi
        
        self.Supervise_Sensors      = cfg.get('Supervise_Sensors', False)

        self.CalRecVar              = CalRecVar
        self.use_adaptive_selection = use_adaptive_selection
        if self.use_adaptive_selection:

            # MLP-1: spatial uncertainty
            self.phi_mlp_1 = nn.Sequential(
                nn.Linear(2, cfg["bayesian_phi"]["phi_mlp_hidden_dim"]), nn.ReLU(),  # Input: (x,y)
                nn.Linear(cfg["bayesian_phi"]["phi_mlp_hidden_dim"], cfg["bayesian_phi"]["phi_mlp_hidden_dim"]), nn.ReLU(),
                nn.Linear(cfg["bayesian_phi"]["phi_mlp_hidden_dim"], 2)  # Output: log(alpha), log(beta) for positivity/stability
            )
            # Initialize with small random weights 
            torch.nn.init.xavier_uniform_(self.phi_mlp_1[0].weight, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_uniform_(self.phi_mlp_1[2].weight, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_uniform_(self.phi_mlp_1[4].weight)

            # MLP-2: temporal uncertainty
            self.phi_mlp_2 = nn.Sequential(
                nn.Linear(2, cfg["bayesian_phi"]["phi_mlp_hidden_dim"]), nn.ReLU(),  
                nn.Linear(cfg["bayesian_phi"]["phi_mlp_hidden_dim"], cfg["bayesian_phi"]["phi_mlp_hidden_dim"]), nn.ReLU(),
                nn.Linear(cfg["bayesian_phi"]["phi_mlp_hidden_dim"], 2)
            )
            torch.nn.init.xavier_uniform_(self.phi_mlp_2[0].weight, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_uniform_(self.phi_mlp_2[2].weight, gain=nn.init.calculate_gain('relu'))
            torch.nn.init.xavier_uniform_(self.phi_mlp_2[4].weight)

            self.avg_residuals = None  # EMA buffer for per-point reconstruction uncertainty [N_pts, 1]

    def forward(self,
                G_down: torch.Tensor,     # [B, N_ts, N_xs, F]  (down-sampled)
                G_full: torch.Tensor,     # [B, T_full, N_pts, 1]
                Y     : torch.Tensor,     # [N_pts, N_dim] 
                U     : torch.Tensor,     # [B, N-para] (can be dummy tensor)
                teacher_force_prob: float,
                ):    

        B, T_full, N_pts, _ = G_full.shape
        _, N_ts, N_xs, F_feat = G_down.shape

        # Adaptive selection is parameterized on the observed sensor coordinates first,
        # then reused downstream by the encoder/decoder and optionally by the dynamics.
        original_phi = None
        if self.use_adaptive_selection:
            current_coords = G_down[:, 0, :, :2]  # [B, K, 2] (per-batch, t=0; shared across t for each case in a batch)
            assert current_coords.shape == (B, N_xs, 2), f"Coords shape mismatch: {current_coords.shape}"

            log_ab_1 = self.phi_mlp_1(current_coords)                         
            alpha_1  = torch.exp(log_ab_1[:, :, 0]) + 1e-3                     
            beta_1   = torch.exp(log_ab_1[:, :, 1]) + 1e-3

            mean_phi_1 = torch.clamp(alpha_1 / (alpha_1 + beta_1) , min=1e-3, max=1-1e-3)  # Clamp for stability
            phi_1 = mean_phi_1

            # Temporal contributions by phi_mlp_2:
            if self.stage == 1 and self.cfg["bayesian_phi"]["update_in_stage1"] == True:
                log_ab_2 = self.phi_mlp_2(current_coords)                        
                alpha_2  = torch.exp(log_ab_2[:, :, 0]) + 1e-3                     
                beta_2   = torch.exp(log_ab_2[:, :, 1]) + 1e-3

                mean_phi_2 = torch.clamp(alpha_2 / (alpha_2 + beta_2), min=1e-3, max=1-1e-3)
                phi_2 = mean_phi_2

            else:
                phi_2 = torch.ones_like(phi_1) # No temporal uncertainty considered

            original_phi = phi_1 * phi_2
        else: 
            original_phi = None # no adaptive sensor selection

        # Encode the sparse observations into latent tokens for each observed frame.
        G_obs, mask_from_encoder, sensor_coords_from_encoder, merged_phi = self.fieldencoder(G_down, U, original_phi) 
        is_multi_token = (G_obs.dim() == 4)

        if is_multi_token:
            B, Tobs, L, D = G_obs.shape
            if self.cfg.get('decoder_type', "CausalTrans") != "UD_Trans":
                latent_seed = G_obs.permute(0, 2, 1, 3).contiguous().view(B * L, Tobs, D)
            else:
                latent_seed = G_obs 

        if self.stage == 0 or self.N_window == T_full: 
            # Stage 0 reconstructs directly from encoded observations without latent rollout.
            latent_traj = G_obs
            latent_traj_logvar = None
        else:
            imp = original_phi.detach() if self.Use_imp_in_dyn is True else None

            if self.training:
                # During training, forecasting is autoregressive with optional teacher forcing.
                output = self.TemporalDecoderAdapter.forward_autoreg(
                    G_latent           = latent_seed,
                    N_Fore             = T_full,
                    imp                = imp,
                    N_window           = self.N_window,
                    teacher_force_seq  = None,      # ground truth tokens: latent_seed[:, self.N_window:], set to None if do not want
                    teacher_force_prob = teacher_force_prob,
                    truncate_k         = 64,               
                )

            else: # evaluation / inference
                # Evaluation uses pure rollout from the observed latent window.
                output = self.TemporalDecoderAdapter(
                    G_latent = latent_seed,
                    N_Fore   = T_full,
                    N_window = self.N_window,
                    imp      = imp,
                    )
            latent_traj, latent_traj_logvar = output

        if is_multi_token and self.cfg.get('decoder_type', "CausalTrans") != "UD_Trans":
            latent_traj = latent_traj.view(B, L, T_full, D).permute(0, 2, 1, 3)  # (B, T_full, L, D)
            if latent_traj_logvar is not None:
                    latent_traj_logvar = latent_traj_logvar.view(B, L, T_full, D).permute(0, 2, 1, 3)

        # Decode the latent trajectory back onto the reconstruction coordinates Y.
        phi_mean = merged_phi if self.use_adaptive_selection else None
        self.phi_mean_ = phi_mean
        self.sensor_coords_ = sensor_coords_from_encoder

        G_u_cls = G_u_cls_sens = None
        G_u_mean_Sens = G_u_logvar_Sens = None
        G_u_mean = G_u_logvar = None

        G_u_cls, G_u_mean, G_u_logvar = self.decoder(latent_traj, Y, sensor_coords=sensor_coords_from_encoder, 
                mask=mask_from_encoder[:, -1:] if mask_from_encoder.dim() > 1 else mask_from_encoder, phi_mean=phi_mean) 
        
        # Optional stage-0 supervision directly at sensor locations.
        if self.stage == 0 and self.Supervise_Sensors:
            G_u_cls_sens, G_u_mean_Sens, G_u_logvar_Sens = self.decoder(latent_traj, sensor_coords_from_encoder, sensor_coords=sensor_coords_from_encoder, 
                    mask=mask_from_encoder[:, -1:] if mask_from_encoder.dim() > 1 else mask_from_encoder, phi_mean=phi_mean)

        return (G_u_mean, G_u_logvar, 
                G_obs, latent_traj, latent_traj_logvar, 
                G_u_cls, 
                G_u_mean_Sens, G_u_logvar_Sens)
