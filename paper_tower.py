"""
paper_tower.py
--------------
PaperTower: SeHGNN-derived encoder that maps pre-propagated metapath features
into a fixed-dim L2-normalised embedding suitable for contrastive retrieval.

Inputs (at forward time):
    feat_dict : dict[str -> Tensor[B, nfeat]]
        Keys must match feat_keys supplied at construction time (e.g. "P", "PP", "PCCon").
        Values are already row-normalised, propagated features — NO adjacency matrices
        needed here; propagation was done offline in step4/step5b.

Output:
    Tensor[B, embed_dim], L2-normalised.

Architecture:
    (optional) LinearPerMetapath  — per-metapath MLP projection  [B, M, nfeat] → [B, M, hidden]
    Transformer                   — cross-metapath semantic fusion
    fc_after_concat               — flatten + project             [B, M*dim] → [B, hidden]
    proj_head                     — Linear(hidden, embed_dim)
    L2 normalisation              — F.normalize(..., dim=-1)

use_mlp=False skips LinearPerMetapath entirely — raw 768-dim features go
directly into the Transformer. Useful as a no-MLP baseline.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _xavier_uniform_(tensor, gain=1.0):
    """Xavier uniform init for 3-D weight tensors (LinearPerMetapath)."""
    fan_in, fan_out = tensor.size()[-2:]
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a   = math.sqrt(3.0) * std
    return torch.nn.init._no_grad_uniform_(tensor, -a, a)


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class LinearPerMetapath(nn.Module):
    """Independent linear projection for each metapath channel."""

    def __init__(self, cin: int, cout: int, num_metapaths: int):
        super().__init__()
        self.W    = nn.Parameter(torch.randn(num_metapaths, cin, cout))
        self.bias = nn.Parameter(torch.zeros(num_metapaths, cout))
        self.reset_parameters()

    def reset_parameters(self):
        _xavier_uniform_(self.W, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, M, cin] → [B, M, cout]
        return torch.einsum("bcm,cmn->bcn", x, self.W) + self.bias.unsqueeze(0)


class Transformer(nn.Module):
    """Cross-metapath self-attention (semantic fusion)."""

    def __init__(self, n_channels: int, num_heads: int, att_drop: float, act: str):
        super().__init__()
        # n_channels: it's the size of each metapath feature vector
        # num_heads: num_heads: number of parallel attention heads.
        # att_drop: dropout probability for attention weights( to prevent overfitting)
        assert n_channels % num_heads == 0, (  # checks that n_channels(feature vector dimension = 768) divides evenly by num_heads so the vector can be split equally across heads with no remainder.
            f"n_channels ({n_channels}) must be divisible by num_heads ({num_heads})"
        )
        self.num_heads = num_heads
        self.query     = nn.Linear(n_channels, n_channels)
        self.key       = nn.Linear(n_channels, n_channels)
        self.value     = nn.Linear(n_channels, n_channels)
        self.gamma     = nn.Parameter(torch.tensor([0.0]))
        self.att_drop  = nn.Dropout(att_drop)

        acts = {"sigmoid": nn.Sigmoid(), "relu": nn.ReLU(),
                "leaky_relu": nn.LeakyReLU(0.2), "none": nn.Identity()}
        if act not in acts:
            raise ValueError(f"Unknown activation '{act}'")
        self.act = acts[act]

        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.query, self.key, self.value]:
            m.reset_parameters()
        nn.init.zeros_(self.gamma)

    def forward(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        B, M, C = x.size()   # batch, num_metapaths, channels
        H = self.num_heads

        # Project input into query/key/value and split across heads → [B, H, M, C//H]
        f = self.query(x).view(B, M, H, -1).permute(0, 2, 1, 3)   # query: "what am I looking for?"
        g = self.key(x).view(B, M, H, -1).permute(0, 2, 3, 1)     # key:   "what do I offer?"
        h = self.value(x).view(B, M, H, -1).permute(0, 2, 1, 3)   # value: "what information do I carry?"

        # Attention scores: how much should each metapath attend to each other → [B, H, M, M]
        # divide by sqrt(C//H) to prevent dot products from becoming too large
        beta = F.softmax(self.act(f @ g / math.sqrt(f.size(-1))), dim=-1)
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.view(B, 1, 1, M)
            beta = beta / (beta.sum(-1, keepdim=True) + 1e-12)

        # Blend value vectors using attention weights.
        # gamma starts at 0 — Transformer contributes nothing at init and gradually learns how much mixing is useful
        o = self.gamma * (beta @ h)

        # Reshape back to [B, M, C] and add residual so original metapath vectors are always preserved
        return o.permute(0, 2, 1, 3).reshape(B, M, C) + x


# ---------------------------------------------------------------------------
# PaperTower
# ---------------------------------------------------------------------------

class PaperTower(nn.Module):

    def __init__(
        self,
        feat_keys:   list,   # metapath keys e.g. ["P", "PP", "PCCon"]
        nfeat:       int,    # input feature dim (768 for SciBERT-based features)
        hidden:      int,    # hidden dim inside Transformer and projection layers
        embed_dim:   int,    # final output dim (CitationBART's vector head is built with this size)
        n_fp_layers: int,    # number of LinearPerMetapath layers (ignored when use_mlp=False)
        dropout:     float,  # activation dropout inside LinearPerMetapath
        input_drop:  float,  # dropout on raw input features before any projection
        att_drop:    float,  # attention dropout inside Transformer
        num_heads:   int,    # number of attention heads
        act:         str,    # Transformer attention activation
        residual:    bool,   # add skip connection from mean(inputs) → hidden
        use_mlp:     bool,   # if False, skip LinearPerMetapath — raw features go straight to Transformer
    ):
        super().__init__()

        self.feat_keys = sorted(feat_keys)
        M              = len(self.feat_keys)
        self.residual  = residual
        self.use_mlp   = use_mlp
        self.input_drop = nn.Dropout(input_drop)

        if use_mlp:
            # Stack of LinearPerMetapath layers: nfeat → hidden (then hidden → hidden)
            assert n_fp_layers >= 1
            layers = [LinearPerMetapath(nfeat, hidden, M),
                      nn.LayerNorm([M, hidden]), nn.PReLU(), nn.Dropout(dropout)]
            for _ in range(n_fp_layers - 1):
                layers += [LinearPerMetapath(hidden, hidden, M),
                           nn.LayerNorm([M, hidden]), nn.PReLU(), nn.Dropout(dropout)]
            self.feature_projection = nn.Sequential(*layers)
            transformer_dim = hidden
        else:
            # No MLP — Transformer receives raw nfeat-dim inputs directly
            self.feature_projection = None
            transformer_dim = nfeat

        # Transformer operates at transformer_dim (= hidden if use_mlp else nfeat)
        self.semantic_fusion = Transformer(transformer_dim, num_heads, att_drop, act)

        # Flatten M metapath vectors → single hidden-dim vector
        self.fc_after_concat = nn.Linear(M * transformer_dim, hidden)

        if residual:
            self.res_fc = nn.Linear(nfeat, hidden, bias=False)

        # Final projection into shared embedding space
        self.proj_head = nn.Linear(hidden, embed_dim)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")

        if self.use_mlp:
            for m in self.feature_projection:
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

        self.semantic_fusion.reset_parameters()
        nn.init.xavier_uniform_(self.fc_after_concat.weight, gain=gain)
        nn.init.zeros_(self.fc_after_concat.bias)

        if self.residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)

        nn.init.xavier_uniform_(self.proj_head.weight, gain=gain)
        nn.init.zeros_(self.proj_head.bias)

    def forward(self, feat_dict: dict) -> torch.Tensor:
        # Stack metapath features → [B, M, nfeat]
        x = torch.stack([feat_dict[k] for k in self.feat_keys], dim=1)
        x = self.input_drop(x)

        if self.residual:
            x_mean = x.mean(dim=1)   # [B, nfeat]

        # Optional MLP projection → [B, M, hidden] (skipped when use_mlp=False)
        if self.use_mlp:
            x = self.feature_projection(x)

        # Cross-metapath attention → same shape as input
        x = self.semantic_fusion(x)

        # Flatten + project → [B, hidden]
        B = x.size(0)
        x = self.fc_after_concat(x.reshape(B, -1))

        if self.residual:
            x = x + self.res_fc(x_mean)

        # Project → [B, embed_dim] and L2 normalise
        return F.normalize(self.proj_head(x), p=2, dim=-1)