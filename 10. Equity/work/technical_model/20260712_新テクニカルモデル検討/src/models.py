from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor


@dataclass
class ModelBundle:
    predictions: pd.DataFrame
    transformer_model: object
    transformer_meta: dict


class TemporalPatchAttentionRegressor:
    """Fast, transparent Transformer-style temporal patch expert.

    The demo uses 5-day patch tokens and a regularized linear head. Per-sample
    temporal attention is computed from absolute patch contributions and then
    normalized with softmax. This preserves the key PM-facing behavior:
    identifying which historical periods drove a prediction and validating that
    result with temporal occlusion. For production, replace this adapter with
    TFT/PatchTST while keeping the same predict_with_attention interface.
    """

    def __init__(self, seq_len: int, patch_size: int = 5, alpha: float = 20.0, temperature: float = 0.35):
        if seq_len % patch_size != 0:
            raise ValueError("seq_len must be divisible by patch_size")
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.alpha = alpha
        self.temperature = temperature
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = Ridge(alpha=alpha)
        self.input_dim = None

    def _tokenize(self, x: np.ndarray) -> np.ndarray:
        # Chronological 5-day patch tokens, represented by within-patch means.
        b, seq, f = x.shape
        self.input_dim = f
        return x.reshape(b, self.n_patches, self.patch_size, f).mean(axis=2)

    def _flat(self, x: np.ndarray, fit: bool = False) -> np.ndarray:
        tokens = self._tokenize(x).reshape(len(x), -1)
        if fit:
            tokens = self.imputer.fit_transform(tokens)
            return self.scaler.fit_transform(tokens)
        tokens = self.imputer.transform(tokens)
        return self.scaler.transform(tokens)

    def fit(self, x: np.ndarray, y: np.ndarray):
        z = self._flat(x, fit=True)
        self.model.fit(z, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(self._flat(x, fit=False))

    def predict_with_attention(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = self._flat(x, fit=False)
        pred = self.model.predict(z)
        coef = self.model.coef_.reshape(self.n_patches, self.input_dim)
        contrib = (z.reshape(len(x), self.n_patches, self.input_dim) * coef[None, :, :]).sum(axis=2)
        logits = np.abs(contrib) / max(self.temperature, 1e-6)
        logits = logits - logits.max(axis=1, keepdims=True)
        attn = np.exp(logits)
        attn = attn / attn.sum(axis=1, keepdims=True)
        return pred, attn


def _make_sequences(daily: pd.DataFrame, monthly: pd.DataFrame, feature_cols: list[str], lookback: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    daily_idx = daily.set_index(["asset_id", "date"]).sort_index()
    xs, ys, meta = [], [], []
    for row in monthly.dropna(subset=["target_scaled"]).itertuples(index=False):
        try:
            g = daily_idx.loc[row.asset_id]
            loc = g.index.get_loc(row.date)
            if isinstance(loc, slice) or loc < lookback - 1:
                continue
            seq = g.iloc[loc - lookback + 1: loc + 1][feature_cols].astype(float).to_numpy()
            if len(seq) != lookback:
                continue
            seq = np.nan_to_num(seq, nan=0.5, posinf=1.0, neginf=0.0)
            xs.append(seq)
            ys.append(float(row.target_scaled))
            meta.append({"date": row.date, "asset_id": row.asset_id})
        except (KeyError, IndexError):
            continue
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32), pd.DataFrame(meta)


def train_and_predict(root: Path, daily: pd.DataFrame, monthly: pd.DataFrame, signal_cols: list[str], evidence_cols: list[str], cfg: dict) -> ModelBundle:
    seed = int(cfg["project"]["random_seed"])
    np.random.seed(seed)
    feature_cols = signal_cols + evidence_cols
    train_end = pd.Timestamp(cfg["model"]["training_end"])
    test_start = pd.Timestamp(cfg["model"]["test_start"])
    train = monthly[(monthly.date <= train_end) & monthly.target_scaled.notna()].copy()
    test = monthly[(monthly.date >= test_start) & monthly.target_scaled.notna()].copy()

    X_train = train[feature_cols]
    y_train = train.target_scaled
    X_test = test[feature_cols]

    ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=float(cfg["model"]["ridge_alpha"]), solver="lsqr"))
    rf = make_pipeline(SimpleImputer(strategy="median"), RandomForestRegressor(
        n_estimators=int(cfg["model"]["random_forest_estimators"]), max_depth=6, min_samples_leaf=8,
        random_state=seed, n_jobs=1,
    ))
    lgbm = make_pipeline(SimpleImputer(strategy="median"), LGBMRegressor(
        n_estimators=int(cfg["model"]["lightgbm_estimators"]), learning_rate=0.03, max_depth=4,
        num_leaves=15, subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0, random_state=seed, verbosity=-1, n_jobs=1,
    ))
    ridge.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    lgbm.fit(X_train, y_train)

    pred = test[["date", "asset_id", "target_return", "target_rank", "target_scaled", "rv20"] + evidence_cols].copy()
    pred["Rule"] = 2 * (
        0.25 * test.evidence_persistence + 0.10 * test.evidence_correction + 0.15 * test.evidence_volatility_support +
        0.15 * test.evidence_flow + 0.20 * test.evidence_relative_strength + 0.10 * test.evidence_intermarket +
        0.05 * test.evidence_macro_market
    ) - 1
    pred["Ridge"] = ridge.predict(X_test)
    pred["RandomForest"] = rf.predict(X_test)
    pred["LightGBM"] = lgbm.predict(X_test)

    lookback = int(cfg["model"]["transformer_lookback_days"])
    patch = int(cfg["model"]["transformer_patch_days"])
    transformer_features = [
        "ret5_pct", "ret20_pct", "ret60_pct", "ret120_pct", "ma_gap60_pct", "ma_slope60_pct", "breakout120_pct",
        "rv20_pct", "vol_ratio_pct", "volume_ratio_pct", "oi_change20_pct", "ret60_cs", "vix_pct", "move_pct",
        "dxy_ret20_pct", "us10y_chg20_pct", "copper_ret20_pct", "local_fx_support_raw_pct",
    ] + evidence_cols
    all_seq, all_y, all_meta = _make_sequences(daily, monthly, transformer_features, lookback)
    train_mask = (all_meta.date <= train_end).to_numpy()
    test_mask = (all_meta.date >= test_start).to_numpy()
    temporal = TemporalPatchAttentionRegressor(seq_len=lookback, patch_size=patch, alpha=25.0, temperature=0.35)
    temporal.fit(all_seq[train_mask], all_y[train_mask])
    trans_df = all_meta[test_mask].reset_index(drop=True)
    trans_df["Transformer"] = temporal.predict(all_seq[test_mask])
    pred = pred.merge(trans_df, on=["date", "asset_id"], how="left")
    pred["Transformer"] = pred["Transformer"].fillna(pred[["Ridge", "LightGBM"]].mean(axis=1))

    return ModelBundle(predictions=pred, transformer_model=temporal, transformer_meta={
        "features": transformer_features, "lookback": lookback, "patch_size": patch,
        "test_sequences": all_seq[test_mask], "test_meta": trans_df[["date", "asset_id"]],
        "implementation": "Transformer-style temporal patch attention surrogate for demo; TFT/PatchTST adapter in production.",
    })
