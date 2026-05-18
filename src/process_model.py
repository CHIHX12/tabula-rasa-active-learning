"""
PC-BAN: Process Community Bilinear Attention Network
=====================================================
Core idea:
  No domain knowledge required. Given only composition fractions x in Simplex^N,
  BAN automatically learns interactions at all orders. The K mixture components
  in MDN naturally form "communities" — no need to specify the number of groups;
  the data determines it.

BAN interactions (process-domain blackbox Ising):
  1st order  → 4  terms  (individual elements)
  2nd order  → 6  terms  (bilinear pairwise)
  3rd order  → 4  terms  (Hadamard triplet)
  4th order  → 1  term   (global 4-body)
  Total: 15 blackbox features

Multimodal communities:
  MDN with K Gaussian components = K implicit communities
  Unused components auto-collapse to pi_k → 0
  No need to specify the number of communities

Invertibility:
  Decoder → x_hat (reconstructed composition)
  Enables inverse design of optimal composition

Generalizability:
  Any process of the form "N input fractions → measured property Y"
  (alloys, pharmaceuticals, food science, microbiome, semiconductors, ...)
"""

from itertools import combinations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# Element Encoder
# ─────────────────────────────────────────────

class ElementEncoder(nn.Module):
    """Maps scalar x_i → embedding h_i (embed_dim). Fully data-driven."""
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, x_i: torch.Tensor) -> torch.Tensor:
        return self.net(x_i.unsqueeze(-1))   # (B, D)


# ─────────────────────────────────────────────
# BAN Interaction Modules
# ─────────────────────────────────────────────

class BANPair(nn.Module):
    """2nd-order bilinear: h_i^T W h_j -> scalar"""
    def __init__(self, embed_dim):
        super().__init__()
        self.W    = nn.Parameter(torch.randn(embed_dim, embed_dim) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, h_i, h_j):
        return (h_i @ self.W * h_j).sum(-1) + self.bias


class BANTriplet(nn.Module):
    """3rd-order low-rank: (h_i * h_j * h_k) @ U @ v -> scalar"""
    def __init__(self, embed_dim, rank=4):
        super().__init__()
        self.U    = nn.Parameter(torch.randn(embed_dim, rank) * 0.01)
        self.v    = nn.Parameter(torch.randn(rank) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, h_i, h_j, h_k):
        return (h_i * h_j * h_k) @ self.U @ self.v + self.bias


class BANGlobal(nn.Module):
    """4th-order global: (h_1 * h_2 * h_3 * h_4) @ U @ v -> scalar"""
    def __init__(self, embed_dim, rank=4):
        super().__init__()
        self.U    = nn.Parameter(torch.randn(embed_dim, rank) * 0.01)
        self.v    = nn.Parameter(torch.randn(rank) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, *hs):
        h = hs[0]
        for hi in hs[1:]:
            h = h * hi
        return h @ self.U @ self.v + self.bias


# ─────────────────────────────────────────────
# PC-BAN Main Model
# ─────────────────────────────────────────────

class PCBAN(nn.Module):
    """
    PC-BAN: Process Community Bilinear Attention Network

    x_raw (N,)
      -> ElementEncoder x N
      -> BAN (1st + 2nd + 3rd + 4th order) -> 15 blackbox features
      -> Projection (15 -> proj_dim)
      -> MDN (K Gaussian components = K implicit communities) -> p(J10 | x)
      -> Decoder -> x_hat (reconstruction for process invertibility)
    """

    def __init__(
        self,
        n_elem       : int   = 4,
        embed_dim    : int   = 32,
        triplet_rank : int   = 4,
        proj_dim     : int   = 30,
        n_components : int   = 12,
        hidden_dims  : list  = None,
        dropout      : float = 0.1,
    ):
        super().__init__()
        self.n_elem = n_elem
        self.K      = n_components

        # Element embeddings
        self.encoders = nn.ModuleList(
            [ElementEncoder(embed_dim) for _ in range(n_elem)])

        # BAN interactions (auto-generate all combinations)
        pairs    = list(combinations(range(n_elem), 2))
        triplets = list(combinations(range(n_elem), 3))
        self.pairs    = pairs
        self.triplets = triplets
        self.ban_pair    = nn.ModuleList([BANPair(embed_dim)    for _ in pairs])
        self.ban_triplet = nn.ModuleList(
            [BANTriplet(embed_dim, triplet_rank) for _ in triplets])
        self.ban_global  = BANGlobal(embed_dim, triplet_rank)

        feat_dim = n_elem + len(pairs) + len(triplets) + 1   # 15 for N=4

        # Feature projection
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )

        # MDN backbone (K components = implicit communities)
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        layers, prev = [], proj_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h),
                       nn.GELU(), nn.Dropout(dropout)]
            prev = h
        self.backbone   = nn.Sequential(*layers)
        self.pi_head    = nn.Linear(prev, n_components)
        self.mu_head    = nn.Linear(prev, n_components)
        self.sigma_head = nn.Linear(prev, n_components)

        # Decoder (reconstruct N-dim composition for process invertibility)
        self.decoder = nn.Sequential(
            nn.Linear(proj_dim, 32),
            nn.GELU(),
            nn.Linear(32, n_elem),
        )

    def _features(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N) -> z: (B, proj_dim)"""
        h = [self.encoders[i](x[:, i]) for i in range(self.n_elem)]
        order1 = [x[:, i] for i in range(self.n_elem)]
        order2 = [self.ban_pair[k](h[i], h[j])
                  for k, (i, j) in enumerate(self.pairs)]
        order3 = [self.ban_triplet[k](h[i], h[j], h[l])
                  for k, (i, j, l) in enumerate(self.triplets)]
        order4 = [self.ban_global(*h)]
        feats  = torch.stack(order1 + order2 + order3 + order4, dim=-1)
        return self.proj(feats)

    def forward(self, x: torch.Tensor):
        """x: (B, N) -> (pi, mu, sigma) each (B, K)"""
        z   = self._features(x)
        h   = self.backbone(z)
        pi  = F.softmax(self.pi_head(h), dim=-1)
        mu  = self.mu_head(h)
        sig = F.softplus(self.sigma_head(h)) + 1e-4
        return pi, mu, sig

    def predict_mean(self, x):
        pi, mu, _ = self.forward(x)
        return (pi * mu).sum(-1)

    def predict_std(self, x):
        pi, mu, sigma = self.forward(x)
        mean = (pi * mu).sum(-1, keepdim=True)
        var  = (pi * (sigma**2 + mu**2)).sum(-1) - mean.squeeze(-1)**2
        return var.clamp(1e-8).sqrt()

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """x -> x_hat (reconstruct composition from learned representation)"""
        z = self._features(x)
        return F.softmax(self.decoder(z), dim=-1)

    def full_loss(self, x, y, weights=None, lam_mse=0.10, lam_rec=0.05):
        """
        L = NLL + lam_mse * MSE + lam_rec * Reconstruction

        NLL : learn multimodal conditional distribution p(J10 | x);
              K Gaussian components naturally form landscape communities
        MSE : directly optimize R^2
        Rec : reconstruct composition (unsupervised regularization)
        """
        z   = self._features(x)
        h   = self.backbone(z)
        pi  = F.softmax(self.pi_head(h), dim=-1)
        mu  = self.mu_head(h)
        sig = F.softplus(self.sigma_head(h)) + 1e-4

        # NLL
        y_exp     = y.unsqueeze(-1)
        log_gauss = (-0.5 * ((y_exp - mu) / sig)**2
                     - sig.log() - 0.5 * np.log(2 * np.pi))
        log_mix   = (pi + 1e-8).log() + log_gauss
        nll       = -torch.logsumexp(log_mix, dim=-1)
        nll = (nll * weights).mean() if weights is not None else nll.mean()

        # MSE
        mean_pred = (pi * mu).sum(-1)
        mse = (mean_pred - y).pow(2)
        mse = (mse * weights).mean() if weights is not None else mse.mean()

        # Reconstruction
        x_hat = F.softmax(self.decoder(z), dim=-1)
        rec   = F.mse_loss(x_hat, x)

        return nll + lam_mse * mse + lam_rec * rec, nll, mse, rec

    @property
    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
