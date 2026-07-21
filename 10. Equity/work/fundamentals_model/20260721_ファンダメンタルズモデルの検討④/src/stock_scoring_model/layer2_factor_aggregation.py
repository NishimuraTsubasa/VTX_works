from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .layer2_ic_weighting import rolling_ic_weights
from .master import FactorMeta


def aggregate_layer2_factor_scores(
    data: pd.DataFrame,
    subscores: pd.DataFrame,
    metas: dict[str, FactorMeta],
    group_methods: dict[str, str],
    config: dict[str, Any],
    target_col: str = "NextMonthReturn",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    c = config["columns"]
    groups: dict[str, list[str]] = {}
    for code, meta in metas.items():
        if code in subscores.columns:
            groups.setdefault(meta.group, []).append(code)
    result = pd.DataFrame(index=data.index)
    history_frames: list[pd.DataFrame] = []

    for group, codes in groups.items():
        method = group_methods.get(group, "equal_weight")
        if method == "manual":
            weights = np.asarray([max(metas[k].base_weight, 0.0) for k in codes], float)
            weights = weights / weights.sum() if weights.sum() > 0 else np.ones(len(codes)) / len(codes)
            arr = subscores[codes].to_numpy(float)
            valid = np.isfinite(arr)
            num = np.nansum(arr * weights, axis=1)
            den = np.sum(valid * weights, axis=1)
            result[group] = np.divide(num, den, out=np.full(len(arr), np.nan), where=den > 0)
            history_frames.append(pd.DataFrame({"Factor_Group": group, "FactorCode": codes, "Weight": weights, "Reason": "manual"}))
        elif method == "ic_adjusted":
            weight_by_date, wh = rolling_ic_weights(data, subscores, codes, config, target_col=target_col)
            values = pd.Series(np.nan, index=data.index, dtype=float)
            for date, idx in data.groupby(c["date"]).groups.items():
                if date not in weight_by_date.index:
                    continue
                w = weight_by_date.loc[date, codes].to_numpy(float)
                arr = subscores.loc[idx, codes].to_numpy(float)
                valid = np.isfinite(arr)
                num = np.nansum(arr * w, axis=1)
                den = np.sum(valid * w, axis=1)
                values.loc[idx] = np.divide(num, den, out=np.full(len(idx), np.nan), where=den > 0)
            result[group] = values
            wh.insert(1, "Factor_Group", group)
            history_frames.append(wh)
        elif method == "pca":
            # OOF SubScoreの当月クロスセクション第1主成分。符号は等ウェイト平均と合わせる。
            values = pd.Series(np.nan, index=data.index, dtype=float)
            rows = []
            for date, idx in data.groupby(c["date"]).groups.items():
                frame = subscores.loc[idx, codes].copy()
                frame = frame.fillna(frame.median())
                if len(frame) < max(10, len(codes) + 2):
                    values.loc[idx] = frame.mean(axis=1)
                    continue
                pca = PCA(n_components=1)
                pc = pca.fit_transform(frame.to_numpy(float)).reshape(-1)
                anchor = frame.mean(axis=1).to_numpy(float)
                if np.corrcoef(pc, anchor)[0, 1] < 0:
                    pc = -pc
                    loadings = -pca.components_[0]
                else:
                    loadings = pca.components_[0]
                values.loc[idx] = pc
                for code, loading in zip(codes, loadings):
                    rows.append({"Date": date, "Factor_Group": group, "FactorCode": code, "Weight": loading, "Reason": "pca_loading"})
            result[group] = values
            history_frames.append(pd.DataFrame(rows))
        else:
            result[group] = subscores[codes].mean(axis=1, skipna=True)
            weights = np.ones(len(codes)) / len(codes)
            history_frames.append(pd.DataFrame({"Factor_Group": group, "FactorCode": codes, "Weight": weights, "Reason": "equal_weight"}))
    history = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
    return result, history
