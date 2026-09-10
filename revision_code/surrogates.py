"""
Surrogate models for the controlled active-learning comparison (revision).

Every surrogate exposes the SAME interface so that the acquisition function and
the query protocol can be held constant while only the surrogate is swapped
(surrogate-isolation study):

    s = Surrogate(...)
    s.fit(X_lab, y_lab, seed)              -> trains / refits
    mu, sigma = s.predict(X)               -> predictive mean and epistemic sd
    s.oob_r2                               -> in-sample OOB R^2 (nan if undefined)

All surrogates consume ONLY the raw composition / mixing-ratio vector, except
`MLPDescriptor`, which consumes hand-engineered elemental descriptors and is
included as the descriptor-based control.
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────
# Elemental descriptors for the descriptor-based control model
# (descriptor comparison: "and, where appropriate, with representative
#  descriptor-based models to demonstrate the benefit of the
#  descriptor-free formulation")
# Values: atomic number, Pauling electronegativity, atomic radius (pm),
#         number of d electrons, first ionisation energy (eV),
#         common oxidation state.
# ─────────────────────────────────────────────────────────────────
ELEM_PROPS = {
    "Ni": [28, 1.91, 149.0, 8, 7.640, 2.0],
    "Fe": [26, 1.83, 156.0, 6, 7.902, 3.0],
    "Co": [27, 1.88, 152.0, 7, 7.881, 2.0],
    "Ce": [58, 1.12, 185.0, 1, 5.539, 3.0],
}


def magpie_like_features(X: np.ndarray, elems=("Ni", "Fe", "Co", "Ce")) -> np.ndarray:
    """Composition-weighted mean / std / min / max of elemental properties.

    This is the standard 'Magpie-style' descriptor construction; it is what a
    conventional descriptor-based surrogate would receive instead of the raw
    mixing ratios.
    """
    P = np.array([ELEM_PROPS[e] for e in elems], dtype=np.float64)  # (n_elem, n_prop)
    w = X / (X.sum(axis=1, keepdims=True) + 1e-12)
    mean = w @ P
    var = w @ (P**2) - mean**2
    std = np.sqrt(np.clip(var, 0.0, None))
    present = (X > 1e-8).astype(np.float64)
    big = np.where(present[:, :, None] > 0, P[None, :, :], -np.inf)
    small = np.where(present[:, :, None] > 0, P[None, :, :], np.inf)
    mx = big.max(axis=1)
    mn = small.min(axis=1)
    mx = np.where(np.isfinite(mx), mx, 0.0)
    mn = np.where(np.isfinite(mn), mn, 0.0)
    return np.hstack([X, mean, std, mx - mn]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────
# Base
# ─────────────────────────────────────────────────────────────────
class BaseSurrogate:
    name = "base"
    #: whether this surrogate provides an out-of-bag estimate
    has_oob = False

    def __init__(self):
        self.oob_r2 = float("nan")
        self.fit_seconds = 0.0

    def fit(self, X, y, seed=0, iteration=0, X_pool=None):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────
# PC-BAN (the proposed descriptor-free surrogate)
# ─────────────────────────────────────────────────────────────────
class PCBANSurrogate(BaseSurrogate):
    """Bootstrap ensemble of PC-BAN with MC-Dropout, identical to the
    published pipeline.  `train_mode` selects the retraining protocol so that
    the contribution of warm-starting and self-knowledge-distillation can be
    isolated (retraining-protocol study).

        train_mode = "warm_kd"  : scratch -> warm start, distil on tier change (published)
        train_mode = "warm"     : scratch -> warm start, plain retrain on tier change
        train_mode = "scratch"  : always retrain from scratch
    """

    name = "pcban"
    has_oob = True

    def __init__(self, n_bootstrap=5, n_mc=20, train_mode="warm_kd",
                 lam_mse=0.10, lam_rec=0.02, kd_alpha=0.5,
                 n_components_override=None, dropout=0.25,
                 sigma_calib=False, warm_epochs=60,
                 y_transform="none", tail_tau=0.0, direction="min"):
        super().__init__()
        #: The loss is dominated by the bulk of the pool (400-450 mV here), so
        #: a 206 mV cliff is fitted as an outlier to be smoothed away.  Two
        #: levers address that, neither of which the published pipeline used:
        #:   y_transform  monotone transform that expands resolution at the
        #:                good end of the range before fitting
        #:   tail_tau     per-sample weights that concentrate the loss on the
        #:                best observations (full_loss already accepts
        #:                `weights`; nothing in the pipeline ever passed any)
        self.y_transform = y_transform
        self.tail_tau = float(tail_tau)
        self.direction = direction
        self._ta = 0.0
        self._t_lo, self._t_hi = -1e9, 1e9
        #: multiply the raw ensemble spread so that it matches the size of the
        #: errors the model actually makes.  Measured on the held-out (OOB)
        #: samples at every refit, so it costs nothing extra.
        self.sigma_calib = sigma_calib
        self.sigma_scale = 1.0
        self.warm_epochs = warm_epochs
        from src.process_model import PCBAN  # noqa: F401  (import check)
        self.n_bootstrap = n_bootstrap
        self.n_mc = n_mc
        self.train_mode = train_mode
        self.lam_mse = lam_mse
        self.lam_rec = lam_rec
        self.kd_alpha = kd_alpha
        self.n_components_override = n_components_override
        self.dropout = dropout
        self.n_elem = None
        self._models = None
        self._tier = None
        self.mean_ = 0.0
        self.scale_ = 1.0

    # -- architecture tiers (unchanged from the published code) --------
    def _warm_epochs(self, n):
        """Fine-tuning budget for a warm start.

        A fixed 60 epochs is applied whether the labelled set holds 20 points or
        220; `warm_epochs < 0` instead scales the budget with the data so the
        number of gradient steps per sample stays roughly constant.
        """
        if self.warm_epochs >= 0:
            return int(self.warm_epochs)
        return int(min(250, max(60, 60 * n / 40)))

    def _tier_of(self, n):
        if n < 40:
            return 0
        if n < 70:
            return 1
        if n < 120:
            return 2
        return 3

    def _make_model(self, n_data):
        from src.process_model import PCBAN
        cfgs = [
            dict(embed_dim=8, triplet_rank=2, proj_dim=12, n_components=4, hidden_dims=[32, 16]),
            dict(embed_dim=16, triplet_rank=2, proj_dim=20, n_components=6, hidden_dims=[64, 32]),
            dict(embed_dim=24, triplet_rank=3, proj_dim=24, n_components=8, hidden_dims=[96, 48]),
            dict(embed_dim=32, triplet_rank=4, proj_dim=30, n_components=10, hidden_dims=[128, 64]),
        ]
        cfg = dict(cfgs[self._tier_of(n_data)])
        if self.n_components_override is not None:
            cfg["n_components"] = int(self.n_components_override)
        return PCBAN(n_elem=self.n_elem, dropout=self.dropout, **cfg).to(DEVICE)

    def _train_one(self, model, X_t, y_t, n_epochs, lr, teacher_soft=None,
                   X_pool_t=None, weights=None):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=n_epochs, eta_min=1e-5 if lr >= 1e-3 else 1e-6)
        model.train()
        for _ in range(n_epochs):
            task_loss, *_ = model.full_loss(X_t, y_t, weights=weights,
                                            lam_mse=self.lam_mse,
                                            lam_rec=self.lam_rec)
            if teacher_soft is not None:
                pi_s, mu_s, _ = model(X_pool_t)
                distill = F.mse_loss((pi_s * mu_s).sum(-1), teacher_soft)
                loss = self.kd_alpha * task_loss + (1.0 - self.kd_alpha) * distill
            else:
                loss = task_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
        model.eval()
        return model

    # -- monotone target transform -------------------------------------
    def _fwd(self, y):
        if self.y_transform == "log":
            return np.log(np.asarray(y, dtype=np.float64) - self._ta)
        return np.asarray(y, dtype=np.float64)

    def _inv(self, t):
        if self.y_transform == "log":
            # Predictions in log space can extrapolate far outside the observed
            # range, and exp() then diverges; clip to the observed span with a
            # margin so the inverse stays on the physical scale.
            t = np.clip(np.asarray(t, dtype=np.float64), self._t_lo, self._t_hi)
            return np.exp(t) + self._ta
        return np.asarray(t, dtype=np.float64)

    def _weights(self, y):
        """Emphasise the best observations; mean weight is kept at 1."""
        if self.tail_tau <= 0:
            return None
        y = np.asarray(y, dtype=np.float64)
        spread = float(y.max() - y.min()) + 1e-9
        gap = (y - y.min()) if self.direction == "min" else (y.max() - y)
        w = np.exp(-gap / (self.tail_tau * spread))
        return w / (w.mean() + 1e-12)

    def fit(self, X, y, seed=0, iteration=0, X_pool=None):
        t0 = time.time()
        self.n_elem = X.shape[1]
        n = len(X)
        if self.y_transform == "log":
            rng_y = float(y.max() - y.min()) + 1e-9
            # a gentle offset: too small an offset makes exp() amplify
            # low-end errors enormously when the prediction is back-transformed
            self._ta = float(y.min()) - 0.6 * rng_y
        y_t = self._fwd(y)
        if self.y_transform == "log":
            span = float(y_t.max() - y_t.min()) + 1e-9
            self._t_lo = float(y_t.min()) - 0.5 * span
            self._t_hi = float(y_t.max()) + 0.5 * span
        self.mean_ = float(y_t.mean())
        self.scale_ = float(y_t.std()) + 1e-8
        y_sc = (y_t - self.mean_) / self.scale_
        w_np = self._weights(y)

        seed_offset = seed * 10000 + iteration
        rng = np.random.default_rng(seed_offset)
        torch.manual_seed(seed_offset)
        torch.cuda.manual_seed_all(seed_offset)

        tier = self._tier_of(n)
        if self.train_mode == "scratch" or self._models is None:
            mode, epochs, lr = "scratch", 250, 1e-3
        elif tier != self._tier:
            mode = "distill" if self.train_mode == "warm_kd" else "scratch"
            epochs, lr = (300, 1e-3)
        else:
            mode, epochs, lr = "warm", self._warm_epochs(n), 3e-4

        teacher_soft = None
        X_pool_t = None
        if mode == "distill":
            mu_t, _ = self._eval_predict(self._models, X_pool)
            teacher_soft = torch.tensor((mu_t - self.mean_) / self.scale_,
                                        dtype=torch.float32, device=DEVICE)
            X_pool_t = torch.tensor(X_pool, dtype=torch.float32, device=DEVICE)

        models, oob_preds = [], np.full((self.n_bootstrap, n), np.nan)
        for b in range(self.n_bootstrap):
            boot = rng.integers(0, n, size=n)
            oob_mask = np.ones(n, dtype=bool)
            oob_mask[np.unique(boot)] = False

            model = self._make_model(n)
            if mode == "warm":
                model.load_state_dict(self._models[b % len(self._models)].state_dict())

            X_t = torch.tensor(X[boot], dtype=torch.float32, device=DEVICE)
            y_t = torch.tensor(y_sc[boot], dtype=torch.float32, device=DEVICE)
            w_t = (torch.tensor(w_np[boot], dtype=torch.float32, device=DEVICE)
                   if w_np is not None else None)
            model = self._train_one(model, X_t, y_t, epochs, lr,
                                    teacher_soft=teacher_soft, X_pool_t=X_pool_t,
                                    weights=w_t)
            models.append(model)

            if oob_mask.sum() > 0:
                Xo = torch.tensor(X[oob_mask], dtype=torch.float32, device=DEVICE)
                with torch.no_grad():
                    oob_preds[b, oob_mask] = self._inv(
                        model.predict_mean(Xo).cpu().numpy()
                        * self.scale_ + self.mean_)

        oob_mean = np.nanmean(oob_preds, axis=0)
        valid = ~np.isnan(oob_mean)
        self.oob_r2 = (float(r2_score(y[valid], oob_mean[valid]))
                       if valid.sum() >= 5 else float("nan"))

        # Calibrate the predictive spread against the errors actually observed
        # out of bag: the raw bootstrap/MC-Dropout spread is far too small
        # (nominal 95% intervals cover ~50% of held-out points), which makes the
        # exploration term of the acquisition function numerically negligible.
        if self.sigma_calib:
            oob_sd = np.nanstd(oob_preds, axis=0)
            ok = valid & np.isfinite(oob_sd) & (oob_sd > 1e-9)
            if ok.sum() >= 5:
                resid = y[ok] - oob_mean[ok]
                num = float(np.sqrt(np.mean(resid ** 2)))
                den = float(np.sqrt(np.mean(oob_sd[ok] ** 2)))
                if den > 1e-9:
                    self.sigma_scale = float(np.clip(num / den, 0.5, 20.0))
        self._models = models
        self._tier = tier
        self.last_mode = mode
        self.fit_seconds = time.time() - t0
        return self

    def _eval_predict(self, models, X):
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        preds = []
        with torch.no_grad():
            for m in models:
                m.eval()
                preds.append(self._inv(
                    m.predict_mean(X_t).cpu().numpy() * self.scale_ + self.mean_))
        preds = np.asarray(preds)
        return preds.mean(0), preds.std(0)

    def predict_mixture(self, X, max_models=None):
        """Return the full Gaussian-mixture predictive law, not just its first
        two moments.

        The published pipeline collapses the K mixture components to a mean and
        a variance and then discards them, which is why K = 1 scores best in a
        sweep: the mixture has nothing to do.  An acquisition function that
        integrates over the mixture can represent what actually matters on this
        landscape -- a composition whose predicted value is mediocre but which
        carries a small probability of being far better, i.e. the probabilistic
        signature of a discontinuity.

        Returns (pi, mu, sigma) each (n_points, n_models * K) in target units.
        """
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        PI, MU, SG = [], [], []
        models = self._models if max_models is None else self._models[:max_models]
        with torch.no_grad():
            for m in models:
                m.eval()
                pi, mu, sg = m(X_t)
                PI.append(pi.cpu().numpy())
                MU.append(mu.cpu().numpy() * self.scale_ + self.mean_)
                SG.append(sg.cpu().numpy() * self.scale_ * self.sigma_scale)
        n = len(models)
        return (np.concatenate(PI, axis=1) / n,
                np.concatenate(MU, axis=1),
                np.concatenate(SG, axis=1))

    def predict(self, X):
        """MC-Dropout + bootstrap ensemble (B x M stochastic passes)."""
        if self.n_mc <= 1:
            return self._eval_predict(self._models, X)
        X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
        preds = []
        with torch.no_grad():
            for m in self._models:
                m.train()
                for _ in range(self.n_mc):
                    preds.append(self._inv(m.predict_mean(X_t).cpu().numpy()
                                           * self.scale_ + self.mean_))
                m.eval()
        preds = np.asarray(preds)
        return preds.mean(0), preds.std(0) * self.sigma_scale


# ─────────────────────────────────────────────────────────────────
# Simple composition-only baselines (descriptor comparison:)
# ─────────────────────────────────────────────────────────────────
class RFSurrogate(BaseSurrogate):
    """Random forest on raw composition; sigma = std over trees."""

    name = "rf"
    has_oob = True

    def __init__(self, n_estimators=300):
        super().__init__()
        self.n_estimators = n_estimators

    def fit(self, X, y, seed=0, iteration=0, X_pool=None):
        t0 = time.time()
        self.m = RandomForestRegressor(n_estimators=self.n_estimators,
                                       oob_score=True, bootstrap=True,
                                       random_state=seed * 10000 + iteration,
                                       n_jobs=2)
        self.m.fit(X, y)
        self.oob_r2 = float(self.m.oob_score_) if len(X) > 5 else float("nan")
        self.fit_seconds = time.time() - t0
        return self

    def predict(self, X):
        per_tree = np.stack([t.predict(X) for t in self.m.estimators_])
        return per_tree.mean(0), per_tree.std(0)


class GPSurrogate(BaseSurrogate):
    """Matern-5/2 GP on raw composition; sigma = GP posterior sd."""

    name = "gp"
    has_oob = False

    def fit(self, X, y, seed=0, iteration=0, X_pool=None):
        t0 = time.time()
        self.ym, self.ys = float(y.mean()), float(y.std()) + 1e-9
        k = (ConstantKernel(1.0) * Matern(nu=2.5, length_scale=np.ones(X.shape[1]))
             + WhiteKernel(1e-2))
        self.m = GaussianProcessRegressor(kernel=k, alpha=1e-6,
                                          n_restarts_optimizer=2,
                                          normalize_y=False,
                                          random_state=seed * 10000 + iteration)
        self.m.fit(X, (y - self.ym) / self.ys)
        self.oob_r2 = float("nan")
        self.fit_seconds = time.time() - t0
        return self

    def predict(self, X):
        mu, sd = self.m.predict(X, return_std=True)
        return mu * self.ys + self.ym, sd * self.ys


class MLPEnsembleSurrogate(BaseSurrogate):
    """Plain MLP deep ensemble; the 'same capacity, no BAN/MDN' control."""

    name = "mlp"
    has_oob = True

    def __init__(self, n_models=5, hidden=(128, 64), descriptor=False):
        super().__init__()
        self.n_models = n_models
        self.hidden = hidden
        self.descriptor = descriptor
        if descriptor:
            self.name = "mlp_descriptor"

    def _feat(self, X):
        return magpie_like_features(X) if self.descriptor else X

    def fit(self, X, y, seed=0, iteration=0, X_pool=None):
        t0 = time.time()
        Xf = self._feat(X)
        self.mu_x, self.sd_x = Xf.mean(0), Xf.std(0) + 1e-8
        Xn = (Xf - self.mu_x) / self.sd_x
        n = len(Xn)
        rng = np.random.default_rng(seed * 10000 + iteration)
        self.ms = []
        oob = np.full((self.n_models, n), np.nan)
        for b in range(self.n_models):
            idx = rng.integers(0, n, size=n)
            oob_mask = np.ones(n, dtype=bool)
            oob_mask[np.unique(idx)] = False
            m = MLPRegressor(hidden_layer_sizes=self.hidden, activation="relu",
                             alpha=1e-3, max_iter=2000, early_stopping=False,
                             random_state=int(rng.integers(0, 2**31 - 1)))
            m.fit(Xn[idx], y[idx])
            self.ms.append(m)
            if oob_mask.sum() > 0:
                oob[b, oob_mask] = m.predict(Xn[oob_mask])
        om = np.nanmean(oob, axis=0)
        v = ~np.isnan(om)
        self.oob_r2 = float(r2_score(y[v], om[v])) if v.sum() >= 5 else float("nan")
        self.fit_seconds = time.time() - t0
        return self

    def predict(self, X):
        Xn = (self._feat(X) - self.mu_x) / self.sd_x
        p = np.stack([m.predict(Xn) for m in self.ms])
        return p.mean(0), p.std(0)


SURROGATES = {
    "pcban": PCBANSurrogate,
    "rf": RFSurrogate,
    "gp": GPSurrogate,
    "mlp": MLPEnsembleSurrogate,
}
