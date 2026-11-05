#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factor Tilt Research Lab -- one-file handoff edition (version 1.0.0)
=================================================================
日本語の引継ぎメモ / READ THIS FIRST
--------------------------------
目的:
  2013-06以降・38ファクター程度の短い月次履歴を想定し、特徴量圧縮、
  情報共有、非線形性、緩やかな時変係数を同一の外側OOSで比較する。
  Kagkadis et al. (2024) の完全再現ではない。標準PCA+PLSを使う比較基盤で、
  原論文のRPPCA、Lasso、銘柄からの特性構築を再現したものではない。
  予測対象は翌1か月だけ。複数月ラベル・重複保有を黙って扱わない。

インストール / 実行:
  python -m pip install numpy pandas scipy scikit-learn threadpoolctl
  python factor_tilt_research.py self-test
  python factor_tilt_research.py demo --out demo_results
  python factor_tilt_research.py demo --models all --factors 38 --months 161 --out all_results
  python factor_tilt_research.py template --out input_template
  python factor_tilt_research.py run --returns returns.csv --features features.csv --out results
  python factor_tilt_research.py list-models
  python factor_tilt_research.py run --help

入力契約 (CSVまたは同じ形のDataFrame、単位は小数: 1%=0.01):
  returns.csv: date,VALUE,QUALITY,...
    dateは月末日。行tのr_tは「前月末から当月末までの実現リターン」。
    月抜け、重複、リターン欠損を拒否。因子の向き・リスク単位を事前に統一。
  features.csv: date,factor,available_at,x_value_spread,x_profitability,...
    各月・各ファクター1行。dateは財務決算期ではなく意思決定スナップショット日。
    x_で始まる列を使用。available_atはその行の全入力が利用可能になった日時。
    available_at <= dateの終日 をチェック。改訂履歴/PITの正しさを独立検証する機能ではない。
    available_atが不明なら明示的に --allow-unverified-pit 。結果に警告を残す。
    財務値の欠損は学習期間内の中央値補完。全期間での補完・標準化はしない。
    個別銘柄の財務データは外部アダプタでファクター特性へ集約すること。

時点の契約:
  月末tで x_t と r_t までを観測し、r_(t+1)を予測。
  学習ペアは (x_s,r_(s+1)), s=0,...,t-1。最新の教師ラベルはr_t。
  同月内のファクターをランダムにtrain/testに分割しない。
  月末情報で月末に約定できるという月次バックテスト上の近似がある。
  実際の発表時刻・引け後情報・翌営業日執行は日次アダプタで対応すること。
  このプログラムだけでリアルタイム約定可能性まで保証するものではない。

特に重要な制約:
  * 各外側予測時点で前処理/PCA/PLS/モデルを過去データだけから再推定。
  * Kalmanは同じ学習窓の同じ座標で最初から再フィルタ。旧PCA座標の状態を流用しない。
    学習期間の全ラベルでPLSを推定するため、内部の過去状態はリアルタイム予測ではない。
    評価するのは学習末の次期予測のみ。平滑化された過去状態を売買に使わない。
  * Q=0とRidgeの等価性をテスト。Qの大きさは内側時系列検証で選択。
  * 外側OOSはモデル比較用。結果を見てモデル/特徴量を選べば最終holdoutではなくなる。
  * 通常のautoencoder/IPCA/銘柄パネル学習は未実装。入力・モデルAPIから拡張可能。
  * バランスした固定因子ユニバースを要求。生存者バイアスや因子採掘は別途監査する。
  * 公開論文を読んで採用した因子を2013年に既知と扱う研究者側の先読みも別途監査する。

return_kindを必ず区別:
  factor_pnl (既定): HML等のLS因子のP&L系列を混合する研究用プロキシ。
    w>=0, sum(w)=1は「因子スリーブへの配分」。原資産のlong-only/市場中立/gross=1を
    意味しない。コストは因子配分変更のproxyのみ。内部回転/借株/調達/原資産相殺は未実装。
    コスト = rate * sum(abs(w_new-w_previous)); 初期配分も課金。
  sleeve_total: 投資可能なスリーブのtotal return。価格変動後の保有比率を計算。
    c=rate*sum(abs((1-c)*w_new-w_pre)) を解き、net=(1-c)*(1+gross)-1。
    スリーブ内部費用は入力リターンに含まれることを想定。別の原資産コストではない。
  rf系列は入力しない。mean_vol_ratio_rf0は無リスク金利ゼロの単純な比率であり、
  実際の超過収益Sharpeとして無条件に引用しない。

次のChatGPTへのファイル分割ガイド (SECTIONはそのままモジュール境界):
  00 config/contracts -> config.py
  01 data/adapters    -> data.py          build_dataset()/load_csv()/snapshots_from_events()
  02 transforms      -> preprocessing.py PanelScaler/PLS圧縮
  03 forecasters     -> models.py        forecast_once()/model_registry()
  04 walk-forward    -> validation.py    select_candidate()/walk_forward()
  05 allocation      -> allocation.py    target_weights()/solve_active_allocation()
  06 backtest        -> backtest.py      simulate_portfolios()/performance_metrics()
  07 reporting       -> reporting.py     export_results()/run_experiment()
  08 synthetic/tests -> tests/,demo.py    make_synthetic()/self_test()
  09 CLI             -> cli.py
  新モデル: registryに候補paramsを登録し、forecast_onceの分岐を追加。
  共通シグナル: predictions.csv のprediction / timing_predictionを使う。
  新戦略: target_weightsの分岐追加。予測を再学習せずにsimulate_portfoliosで比較可能。
  ExperimentResultに全DataFrameを保持し、importしてnotebook等からも利用可能。

主要出力:
  predictions.csv, model_selection.csv, model_diagnostics.csv,
  signal_summary.csv, signal_by_factor.csv, signal_monthly_ic.csv, signal_correlations.csv,
  strategy_returns.csv, weights.csv, strategy_summary.csv, cost_sensitivity.csv,
  bootstrap_vs_equal.csv, expost_mean_weight_diagnostic.csv,
  manifest.json, REPORT.md
  expost_mean_weight_diagnosticは全OOSで平均した重みを使う事後診断で、投資可能戦略でない。
  bootstrapは実現済み経路の条件付き不確実性のみ。再学習/モデル選択/多重検定調整はしない。

参考 (理論の完全再現を主張するものではない):
  Kagkadis et al., Factor Timing with Portfolio Characteristics, 2024,
    DOI:10.1093/rapstu/raad010
  https://scikit-learn.org/stable/modules/generated/sklearn.cross_decomposition.PLSRegression.html
  https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
  https://scikit-learn.org/stable/modules/covariance.html
  https://www.statsmodels.org/stable/statespace.html
  https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html

実データの有効性を保証するものではない。demoは動作検証用の合成データ。
データ送信/API接続/自動発注は一切しない。
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import logging
import math
import platform
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

try:
    import numpy as np
    import pandas as pd
    from scipy.linalg import solve as dense_solve
    from scipy.optimize import minimize
    from scipy.stats import rankdata
    from sklearn.covariance import LedoitWolf
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor
    from threadpoolctl import threadpool_limits
except ImportError as exc:
    raise SystemExit(
        "依存パッケージ不足。python -m pip install numpy pandas scipy scikit-learn threadpoolctl\n"
        f"詳細: {exc}"
    ) from exc

VERSION = "1.0.0"
LOG = logging.getLogger("factor_tilt")
EPS = 1e-12

# ========================= SECTION 00: CONFIG / CONTRACTS =====================
@dataclass(frozen=True)
class ResearchConfig:
    """月次・1期先のみ。候補選択は外側OOSとは分離する。"""
    min_train: int = 60
    train_window: int | None = None
    inner_min_train: int = 36
    inner_validation_months: int = 12
    inner_stride: int = 3
    tune_every: int = 12
    cv_tolerance: float = 0.0  # best*(1+tolerance)内ならregistry先頭の単純な候補を選ぶ
    winsor_quantile: float = 0.01
    min_feature_coverage: float = 0.20
    prediction_clip_sigma: float = 3.0
    covariance_window: int = 60
    factor_max_weight: float | None = None  # None -> max(0.10, 2/N), <=1
    tilt_l1_budget: float = 0.25
    active_risk_aversion: float = 20.0
    anchor_penalty: float = 0.01
    max_active_vol_annual: float = 0.05
    prediction_shrinkage: float = 0.50
    turnover_penalty_bps: float = 10.0  # optimizerのL1 proxy。実現費用はcost_bps。
    cost_bps: float = 10.0  # traded notionalの片道費用。0.5*sum(abs)ではない。
    cost_scenarios_bps: tuple[float, ...] = (0.0, 10.0, 25.0)
    return_kind: str = "factor_pnl"
    bootstrap_samples: int = 500
    bootstrap_block: int = 6
    hac_lags: int = 3
    seed: int = 42
    threads: int = 1

    def validate(self, t: int, n: int) -> None:
        if self.min_train < 12 or self.inner_min_train < 8:
            raise ValueError("min_train>=12, inner_min_train>=8 が必要です。")
        if self.inner_min_train >= self.min_train:
            raise ValueError("inner_min_train は min_train 未満にしてください。")
        if t < self.min_train + 3:
            raise ValueError(f"最低{self.min_train+3}か月必要です。入力は{t}か月。")
        if self.train_window is not None and self.train_window < self.min_train:
            raise ValueError("train_window は min_train 以上にしてください。")
        if min(self.inner_stride, self.tune_every, self.inner_validation_months, self.threads) < 1:
            raise ValueError("分割幅・再選択間隔・スレッド数は正整数です。")
        if not 0 <= self.winsor_quantile < 0.25:
            raise ValueError("winsor_quantile は [0, .25) です。")
        if not 0 < self.min_feature_coverage <= 1:
            raise ValueError("min_feature_coverage は (0,1] です。")
        if self.prediction_clip_sigma <= 0 or self.covariance_window < 6:
            raise ValueError("prediction_clip_sigma>0, covariance_window>=6 が必要です。")
        if self.return_kind not in {"factor_pnl", "sleeve_total"}:
            raise ValueError("return_kind は factor_pnl / sleeve_total です。")
        if self.max_weight(n) * n < 1 - 1e-10:
            raise ValueError("factor_max_weight * 因子数 < 1 で配分制約が実行不能です。")
        if not 0 <= self.prediction_shrinkage <= 1:
            raise ValueError("prediction_shrinkage は [0,1] です。")
        if min(self.active_risk_aversion, self.anchor_penalty, self.max_active_vol_annual) <= 0:
            raise ValueError("リスク/アンカー/active volの設定は正にしてください。")
        if min(self.cost_bps, self.turnover_penalty_bps, *self.cost_scenarios_bps) < 0:
            raise ValueError("費用は非負です。")
        if max(self.cost_bps, *self.cost_scenarios_bps) >= 1000:
            raise ValueError("費用が10%以上です。bps単位を確認してください。")
        if self.tilt_l1_budget < 0 or self.bootstrap_samples < 0 or self.bootstrap_block < 1:
            raise ValueError("budget/bootstrapの設定が不正です。")
        if not 0 <= self.cv_tolerance < 1:
            raise ValueError("cv_tolerance は [0,1) です。")

    def max_weight(self, n: int) -> float:
        cap = min(1.0, max(0.10, 2.0 / n)) if self.factor_max_weight is None else self.factor_max_weight
        if not 0 < cap <= 1:
            raise ValueError("factor_max_weight は (0,1] です。")
        return float(cap)


@dataclass
class ResearchData:
    dates: pd.DatetimeIndex
    factors: tuple[str, ...]
    feature_names: tuple[str, ...]
    returns: np.ndarray  # (T,N)
    features: np.ndarray  # (T,N,P)
    metadata: dict[str, Any] = field(default_factory=dict)

    def prefix(self, length: int) -> "ResearchData":
        return ResearchData(self.dates[:length], self.factors, self.feature_names,
                            self.returns[:length].copy(), self.features[:length].copy(),
                            dict(self.metadata))


@dataclass(frozen=True)
class ModelSpec:
    name: str
    description: str
    candidates: tuple[dict[str, Any], ...]


@dataclass
class ForecastResult:
    prediction: np.ndarray
    raw_prediction: np.ndarray
    historical_mean: np.ndarray
    train_sigma: np.ndarray
    diagnostics: dict[str, Any]


@dataclass
class ExperimentResult:
    predictions: pd.DataFrame
    model_selection: pd.DataFrame
    model_diagnostics: pd.DataFrame
    signal_summary: pd.DataFrame
    signal_by_factor: pd.DataFrame
    signal_monthly_ic: pd.DataFrame
    signal_correlations: pd.DataFrame
    strategy_returns: pd.DataFrame
    weights: pd.DataFrame
    strategy_summary: pd.DataFrame
    cost_sensitivity: pd.DataFrame
    bootstrap_vs_equal: pd.DataFrame
    expost_mean_weight_diagnostic: pd.DataFrame
    manifest: dict[str, Any]


def jsonable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return jsonable(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(x) for x in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, datetime, Path)):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def jdump(value: Any) -> str:
    return json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True, allow_nan=False)


# ========================= SECTION 01: DATA / ADAPTERS =========================
def _month_ends(values: Iterable[Any], name: str) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if idx.tz is not None:
        raise ValueError(f"{name}: タイムゾーンは事前に統一し、日付をnaive月末日にしてください。")
    if idx.hasnans or not np.all(idx == idx.normalize()) or not np.all(idx.is_month_end):
        raise ValueError(f"{name}: 時刻なしの月末日 (例2013-06-30) を指定してください。")
    return idx


def build_dataset(returns_frame: pd.DataFrame, feature_frame: pd.DataFrame,
                  *, feature_cols: Sequence[str] | None = None,
                  allow_unverified_pit: bool = False,
                  metadata: dict[str, Any] | None = None) -> ResearchData:
    """推奨アダプタ入口。入力DataFrameは変更しない。"""
    rf = returns_frame.copy()
    ff = feature_frame.copy()
    if "date" not in rf or not {"date", "factor"}.issubset(ff.columns):
        raise ValueError("returnsにはdate、featuresにはdate,factorが必要です。")
    rf["date"] = _month_ends(rf["date"], "returns.date")
    ff["date"] = _month_ends(ff["date"], "features.date")
    if rf["date"].duplicated().any():
        raise ValueError("returns.dateが重複しています。")
    rf = rf.sort_values("date").set_index("date")
    if len(rf) < 3:
        raise ValueError("月次データが不足しています。")
    months = rf.index.to_period("M").asi8
    if np.any(np.diff(months) != 1):
        raise ValueError("returnsに月抜けがあります。黙って行を詰める処理はしません。")
    factors = tuple(map(str, rf.columns))
    if len(factors) < 2 or len(set(factors)) != len(factors):
        raise ValueError("少なくとも2本の重複しないファクターが必要です。")
    ff["factor"] = ff["factor"].astype(str)
    if ff.duplicated(["date", "factor"]).any():
        raise ValueError("featuresの(date,factor)が重複しています。")
    if set(ff["factor"]) != set(factors):
        raise ValueError("returnsとfeaturesのfactor集合が一致しません。")
    if set(ff["date"]) != set(rf.index):
        raise ValueError("returnsとfeaturesのdate集合が一致しません。明示的に期間を合わせてください。")
    feature_cols = tuple(feature_cols or [c for c in ff if c.startswith("x_")])
    if not feature_cols or not set(feature_cols).issubset(ff.columns):
        raise ValueError("x_で始まる特徴量列、または明示したfeature_colsが必要です。")
    if any(c in {"date", "factor", "available_at"} for c in feature_cols):
        raise ValueError("管理列を特徴量に指定できません。")
    expected = pd.MultiIndex.from_product([rf.index, factors], names=["date", "factor"])
    fidx = ff.set_index(["date", "factor"])
    if len(fidx) != len(expected) or not expected.isin(fidx.index).all():
        raise ValueError("featuresは全月×全ファクターのスナップショットが必要です。欠損値はNaNで保持してください。")
    fidx = fidx.reindex(expected)
    pit = False
    if "available_at" in fidx:
        avail = pd.to_datetime(fidx["available_at"], errors="raise")
        if getattr(avail.dt, "tz", None) is not None:
            raise ValueError("available_atのタイムゾーンをdateと同じ基準に統一してください。")
        cutoff = expected.get_level_values("date") + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        if avail.isna().any():
            if not allow_unverified_pit:
                raise ValueError("available_atに欠損があります。PIT未確認を許可する場合のみフラグを指定してください。")
        if (avail.to_numpy() > cutoff.to_numpy()).any():
            raise ValueError("未来のavailable_atが含まれます。lagで修正し、未来情報を除いてください。")
        pit = not avail.isna().any()
    elif not allow_unverified_pit:
        raise ValueError("available_atが必要です。不明な場合は --allow-unverified-pit を明示してください。")
    r = rf.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    x = fidx[list(feature_cols)].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if not np.isfinite(r).all():
        raise ValueError("returnsのNaN/infは許可しません。欠落因子をゼロ埋めしないでください。")
    if np.isinf(x).any():
        raise ValueError("特徴量のinfは許可しません。定義不能な比率等をNaNにしてください。")
    if np.max(np.abs(r)) > 0.75:
        LOG.warning("リターンの絶対値が75%%を超えます。%%と小数の取り違えがないか確認してください。")
    meta = dict(metadata or {})
    meta.update({"availability_timestamp_checks_passed": pit,
                 "point_in_time_independently_verified": False,
                 "allow_unverified_pit": allow_unverified_pit,
                 "feature_missing_fraction": float(np.isnan(x).mean()),
                 "input_assumption": "fixed balanced factor universe, monthly snapshots"})
    digest = hashlib.sha256()
    for frame in (rf, fidx[list(feature_cols)]):
        digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    meta["numeric_data_sha256"] = digest.hexdigest()
    return ResearchData(rf.index, factors, tuple(feature_cols), r,
                        x.reshape(len(rf), len(factors), len(feature_cols)), meta)


def load_csv(returns_path: str | Path, features_path: str | Path,
             *, feature_cols: Sequence[str] | None = None,
             allow_unverified_pit: bool = False) -> ResearchData:
    paths = [Path(returns_path), Path(features_path)]
    for p in paths:
        if not p.is_file():
            raise FileNotFoundError(p)
    meta = {"synthetic": False, "source_paths": [str(p.resolve()) for p in paths],
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    return build_dataset(pd.read_csv(paths[0]), pd.read_csv(paths[1]),
                         feature_cols=feature_cols, allow_unverified_pit=allow_unverified_pit,
                         metadata=meta)


def snapshots_from_events(events: pd.DataFrame, dates: Sequence[Any],
                          factors: Sequence[str], feature_cols: Sequence[str]) -> pd.DataFrame:
    """利用可能日時でas-of join。eventsは因子単位の完全な特性ベクトル更新イベント。

    events: factor, available_at, x_*。異なる項目の個別更新イベントには直接使わない。
    改訂値の利用可能日時を正しく保持していないデータをこの関数でPIT化はできない。
    未公表期間はNaN。先読みのbackfillはしない。build_datasetで未確認扱いの許可が必要な場合がある。
    """
    event = events.copy()
    event["available_at"] = pd.to_datetime(event["available_at"], errors="raise")
    if event["available_at"].isna().any() or event.duplicated(["factor", "available_at"]).any():
        raise ValueError("eventsの日時欠損/同一因子同時刻の重複を解消してください。")
    idx = _month_ends(dates, "snapshot dates")
    if not idx.is_monotonic_increasing:
        raise ValueError("snapshot datesを昇順にしてください。")
    out = []
    for factor in factors:
        part = event.loc[event["factor"].astype(str) == str(factor),
                         ["available_at", *feature_cols]].sort_values("available_at")
        left = pd.DataFrame({"date": idx, "cutoff": idx + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)})
        if part.empty:
            left["available_at"] = pd.NaT
            for c in feature_cols:
                left[c] = np.nan
        else:
            left = pd.merge_asof(left, part, left_on="cutoff", right_on="available_at",
                                 direction="backward", allow_exact_matches=True)
        left["factor"] = str(factor)
        out.append(left[["date", "factor", "available_at", *feature_cols]])
    return pd.concat(out, ignore_index=True)


# ========================= SECTION 02: PREPROCESSING ==========================
class PanelScaler:
    """因子×特徴量別に学習窓内でwinsorize/median-impute/time-standardize。

    各因子の通常水準を除去して「いつもと比べた状態」を扱う。
    同じ特徴量名でも意味が因子間で異なる場合はpooledモデルを使わないこと。
    学習窓で未観測/一定の列は当該fit内で0化。未来の値から採否を判断しない。
    """
    def __init__(self, winsor: float = .01, min_coverage: float = .20):
        self.winsor = winsor
        self.min_coverage = min_coverage

    def fit(self, x: np.ndarray) -> "PanelScaler":
        if x.ndim != 3 or not len(x):
            raise ValueError("PanelScalerは(T,N,P)配列を受け取ります。")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.median = np.nanmedian(x, axis=0)
            self.low = np.nanquantile(x, self.winsor, axis=0)
            self.high = np.nanquantile(x, 1-self.winsor, axis=0)
        self.median = np.nan_to_num(self.median, nan=0.)
        self.low = np.where(np.isfinite(self.low), self.low, self.median)
        self.high = np.where(np.isfinite(self.high), self.high, self.median)
        filled = np.where(np.isnan(x), self.median, x)
        clipped = np.clip(filled, self.low, self.high)
        self.mean = clipped.mean(axis=0)
        sd = clipped.std(axis=0, ddof=0)
        self.active = (np.isfinite(x).mean(axis=0) >= self.min_coverage) & (sd > 1e-10)
        self.scale = np.where(self.active, sd, 1.)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        filled = np.where(np.isnan(x), self.median, x)
        z = (np.clip(filled, self.low, self.high) - self.mean) / self.scale
        return np.where(self.active, z, 0.)

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        return self.fit(x).transform(x)


def normalize_target(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = y.mean(axis=0)
    scale = y.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-10, scale, 1.)
    return (y-mean)/scale, mean, scale


def ridge_predict(x: np.ndarray, y: np.ndarray, xnew: np.ndarray,
                  lam: float, weights: np.ndarray | None = None) -> np.ndarray:
    """mean squared error + lam*||beta||^2。lamはサンプル数で正規化済み。
    sklearn Ridge(alpha=lam*sum(weights))と同じ目的関数。切片は罰しない。
    """
    if lam <= 0:
        raise ValueError("ridge lambdaは正にしてください。")
    weights = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    xm = weights @ x
    ym = weights @ y
    xc, yc = x-xm, y-ym
    gram = (xc*weights[:, None]).T @ xc
    rhs = (xc*weights[:, None]).T @ yc
    beta = dense_solve(gram + lam*np.eye(x.shape[1]), rhs, assume_a="pos")
    return (xnew-xm) @ beta + ym


def pls_scores(x: np.ndarray, y: np.ndarray, xnew: np.ndarray,
               n_components: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """教師あり圧縮。fitした座標でtrain/newを変換しscoreを標準化する。"""
    active = x.std(axis=0) > 1e-10
    if not active.any() or float(np.std(y)) < 1e-10:
        return np.zeros((len(x), 1)), np.zeros((len(xnew), 1)), {"components": 0, "constant": True}
    xa, xn = x[:, active], xnew[:, active]
    k = min(int(n_components), xa.shape[1], len(x)-1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pls = PLSRegression(n_components=k, scale=False, max_iter=1000, tol=1e-7)
        z = pls.fit_transform(xa, y)[0]
        znew = pls.transform(xn)
    if not np.isfinite(z).all() or not np.isfinite(znew).all():
        raise FloatingPointError("PLSが非有限のscoreを返しました。入力/成分数を確認してください。")
    mean = z.mean(axis=0)
    sd = z.std(axis=0)
    keep = sd > 1e-10
    if not keep.any():
        return np.zeros((len(x), 1)), np.zeros((len(xnew), 1)), {"components": 0, "constant": True}
    return ((z[:, keep]-mean[keep])/sd[keep],
            (znew[:, keep]-mean[keep])/sd[keep],
            {"components": int(keep.sum()), "warnings": [str(w.message) for w in caught]})


def interaction_map(x: np.ndarray) -> np.ndarray:
    """事前に固定した2次項。特徴量の選択を全期間で行わない。"""
    p = x.shape[-1]
    if p*(p+3)//2 > 512:
        raise ValueError("2次項が512列を超えます。事前にfeature_colsを絞るか、このモデルを除外してください。")
    pieces = [x]
    for i in range(p):
        for j in range(i, p):
            pieces.append((x[..., i]*x[..., j])[..., None])
    return np.concatenate(pieces, axis=-1)

# ========================= SECTION 03: FORECASTERS =============================
def kalman_regression_predict(z: np.ndarray, y: np.ndarray, znew: np.ndarray,
                              lam: float, q: float) -> tuple[np.ndarray, dict[str, Any]]:
    """縮小付きrandom-walk係数。観測分散1、Q=q*I、beta_0=0。

    yとzを学習内で中心化/標準化した単位でフィルタする。
    P_0=I/(n*lam) とするのでq=0は平均二乗誤差+lam罰則のRidgeと同じ。
    Qは標準化された係数の月次変動分散。qは内側検証で選ぶ。
    フィルタのJoseph形式で共分散を更新。smoothed stateを使用しない。
    本モデルは全ベイズ階層TVPの実装ではなく、固定qの正則化比較モデル。
    """
    if lam <= 0 or q < 0:
        raise ValueError("lam>0, q>=0 が必要です。")
    zm = z.mean(axis=0)
    zs = np.where(z.std(axis=0) > 1e-10, z.std(axis=0), 1.)
    a, an = (z-zm)/zs, (znew-zm)/zs
    ym = float(y.mean())
    ys = max(float(y.std()), 1e-10)
    response = (y-ym)/ys
    d = a.shape[1]
    eye = np.eye(d)
    beta = np.zeros(d)
    cov = eye/(len(a)*lam)
    innovations = []
    for zi, yi in zip(a, response):
        prior_cov = cov + q*eye
        s = 1.0 + zi @ prior_cov @ zi
        gain = prior_cov @ zi / s
        innovation = yi - zi @ beta
        beta = beta + gain*innovation
        transition = eye - np.outer(gain, zi)
        cov = transition @ prior_cov @ transition.T + np.outer(gain, gain)
        cov = .5*(cov+cov.T)
        innovations.append(innovation/math.sqrt(s))
    prediction = ym + ys*(an @ beta)
    return prediction, {"q": q, "beta_last_standardized": beta.tolist(),
                        "state_cov_diagonal": np.diag(cov).tolist(),
                        "innovation_rms": float(np.sqrt(np.mean(np.square(innovations)))),
                        "observation_variance_standardized": 1.0,
                        "prior_precision": len(a)*lam,
                        "state_cov_is_calibrated_forecast_interval": False}


def partial_pool_predict(x: np.ndarray, y: np.ndarray, xnew: np.ndarray,
                         lam_shared: float, lam_deviation: float) -> tuple[np.ndarray, dict[str, Any]]:
    """Exact block elimination for shared beta plus factor-specific u_k.

    Objective:
      (1/N) sum_k mean_t[(y_kt-x_kt'(beta+u_k))^2]
      + lam_shared*||beta||^2 + (lam_deviation/N)*sum_k||u_k||^2.
    x,yは因子別に中心化済み。共有は各因子の標準偏差単位で行う。
    巨大な(T*N) x (N*P)ダミー行列を作らない。
    """
    if min(lam_shared, lam_deviation) <= 0:
        raise ValueError("共有/差分の正則化係数は正にしてください。")
    t, n, p = x.shape
    gram = np.einsum("tkp,tkq->kpq", x, x)/t
    rhs = np.einsum("tkp,tk->kp", x, y)/t
    identity = np.eye(p)
    inverses_g, inverses_rhs = [], []
    lhs_total = np.zeros((p, p))
    rhs_total = np.zeros(p)
    for k in range(n):
        ak = gram[k]+lam_deviation*identity
        ag = dense_solve(ak, gram[k], assume_a="pos")
        ar = dense_solve(ak, rhs[k], assume_a="pos")
        inverses_g.append(ag)
        inverses_rhs.append(ar)
        lhs_total += gram[k]-gram[k] @ ag
        rhs_total += rhs[k]-gram[k] @ ar
    lhs = .5*(lhs_total+lhs_total.T)/n + lam_shared*identity
    shared = dense_solve(lhs, rhs_total/n, assume_a="pos")
    deviation = np.stack([ar-ag @ shared for ar, ag in zip(inverses_rhs, inverses_g)])
    prediction = np.einsum("kp,kp->k", xnew, shared[None, :]+deviation)
    return prediction, {"shared_coefficient_rms": float(np.sqrt(np.mean(shared**2))),
                        "deviation_coefficient_rms": float(np.sqrt(np.mean(deviation**2)))}


def model_registry() -> dict[str, ModelSpec]:
    """候補数を意図的に小さくする。grid順は同点時の単純さ優先順。"""
    specs = [
        ModelSpec("mean", "各因子の学習期間の過去平均", ({},)),
        ModelSpec("local_ridge", "各因子を個別にRidge回帰", (
            {"lam": 1.0}, {"lam": .1})),
        ModelSpec("local_pls_ridge", "各因子の特性をPLS圧縮しRidge", (
            {"lam": 1.0, "pls_components": 1}, {"lam": 1.0, "pls_components": 2})),
        ModelSpec("local_ew_ridge", "各因子の指数減衰Ridge", (
            {"lam": 1.0, "half_life": 36.0}, {"lam": 1.0, "half_life": 12.0})),
        ModelSpec("pooled_ridge", "因子別切片/尺度と共通傾きのRidge", (
            {"lam": 1.0}, {"lam": .1})),
        ModelSpec("partial_pool", "共通係数+強く縮小する因子固有差", (
            {"lam_shared": .1, "lam_deviation": 10.0},
            {"lam_shared": .1, "lam_deviation": 1.0})),
        ModelSpec("pooled_interactions", "事前固定2次項を共有するRidge", (
            {"lam": 10.0}, {"lam": 1.0})),
        ModelSpec("pooled_histgb", "共有型の浅いHistogram Gradient Boosting", (
            {"max_leaf_nodes": 4, "max_iter": 60},
            {"max_leaf_nodes": 7, "max_iter": 60})),
        ModelSpec("pc_pls_ridge", "リターン相関PCA+PC別PLS+静的Ridge", (
            {"pcs": 3, "pls_components": 2, "lam": 1.0},
            {"pcs": 3, "pls_components": 2, "lam": .1})),
        ModelSpec("pc_pls_ew", "同じPCA/PLS構成+指数減衰Ridge", (
            {"pcs": 3, "pls_components": 2, "lam": 1.0, "half_life": 36.0},
            {"pcs": 3, "pls_components": 2, "lam": 1.0, "half_life": 12.0})),
        ModelSpec("pc_pls_static_kalman", "PCA/PLS+Kalman Q=0の対照実験", (
            {"pcs": 3, "pls_components": 2, "lam": 1.0, "q": 0.0},)),
        ModelSpec("pc_pls_tvp", "PCA/PLS+縮小付き時変係数。Q=0も候補", (
            {"pcs": 3, "pls_components": 2, "lam": 1.0, "q": 0.0},
            {"pcs": 3, "pls_components": 2, "lam": 1.0, "q": .0001},
            {"pcs": 3, "pls_components": 2, "lam": 1.0, "q": .001})),
    ]
    return {s.name: s for s in specs}


def resolve_models(names: str | Sequence[str]) -> list[ModelSpec]:
    registry = model_registry()
    if isinstance(names, str):
        if names == "all":
            selected = list(registry)
        elif names == "core":
            selected = [k for k in registry if k not in {"local_pls_ridge", "pooled_histgb"}]
        else:
            selected = [s.strip() for s in names.split(",") if s.strip()]
    else:
        selected = list(names)
    if len(selected) != len(set(selected)):
        raise ValueError("modelsが重複しています。")
    if not selected or any(k not in registry for k in selected):
        raise ValueError(f"未知のmodels。使用可能: {', '.join(registry)}")
    if "mean" not in selected:
        selected.insert(0, "mean")
    return [registry[k] for k in selected]


def forecast_once(model: str, params: dict[str, Any], x: np.ndarray, y: np.ndarray,
                  xnew: np.ndarray, config: ResearchConfig) -> ForecastResult:
    """独立したfit/predict API。x=(T,N,P), y=(T,N), xnew=(N,P)。

    x,yは既に因果的に対応づけられた学習ペア。未来配列は受け取らない。
    毎回新しく前処理/モデルを作り、前回の状態を流用しない。
    """
    if x.ndim != 3 or y.shape != x.shape[:2] or xnew.shape != x.shape[1:]:
        raise ValueError("forecast_onceの配列shapeが不整合です。")
    if not np.isfinite(y).all() or np.isinf(x).any() or np.isinf(xnew).any():
        raise ValueError("学習データに不正な値があります。")
    n_time, n_factor, _ = x.shape
    hist = y.mean(axis=0)
    train_sigma = np.maximum(y.std(axis=0, ddof=1), 1e-6)
    diag: dict[str, Any] = {"n_train": n_time, "params": params}
    if model == "mean":
        mu = hist.copy()
    else:
        scaler = PanelScaler(config.winsor_quantile, config.min_feature_coverage)
        xs = scaler.fit_transform(x)
        xn = scaler.transform(xnew[None, ...])[0]
        ys, ymean, yscale = normalize_target(y)
        diag["inactive_feature_fraction"] = float(1-scaler.active.mean())
        if model in {"local_ridge", "local_ew_ridge", "local_pls_ridge"}:
            values, dimensions = [], []
            for k in range(n_factor):
                zk, zn = xs[:, k, :], xn[k:k+1, :]
                if model == "local_pls_ridge":
                    zk, zn, info = pls_scores(zk, ys[:, k], zn, params["pls_components"])
                    dimensions.append(info)
                weights = None
                if model == "local_ew_ridge":
                    weights = np.exp2(-(n_time-1-np.arange(n_time))/params["half_life"])
                v = ridge_predict(zk, ys[:, k], zn, params["lam"], weights)[0]
                values.append(v)
            mu = ymean + yscale*np.asarray(values)
            if dimensions:
                diag["pls"] = dimensions
        elif model == "pooled_ridge":
            p = xs.shape[-1]
            values = ridge_predict(xs.reshape(-1, p), ys.reshape(-1), xn, params["lam"])
            mu = ymean+yscale*values
        elif model == "partial_pool":
            values, info = partial_pool_predict(xs, ys, xn, params["lam_shared"], params["lam_deviation"])
            mu = ymean+yscale*values
            diag.update(info)
        elif model == "pooled_interactions":
            poly = interaction_map(xs)
            polynew = interaction_map(xn)
            center = poly.mean(axis=0)
            scale = np.where(poly.std(axis=0) > 1e-10, poly.std(axis=0), 1.)
            z = (poly-center)/scale
            zn = (polynew-center)/scale
            values = ridge_predict(z.reshape(-1, z.shape[-1]), ys.reshape(-1), zn, params["lam"])
            mu = ymean+yscale*values
            diag["interaction_feature_count"] = z.shape[-1]
        elif model == "pooled_histgb":
            learner = HistGradientBoostingRegressor(
                max_iter=params["max_iter"], max_leaf_nodes=params["max_leaf_nodes"],
                max_depth=3, learning_rate=.05, min_samples_leaf=max(20, n_factor*3),
                l2_regularization=10.0, early_stopping=False, random_state=config.seed)
            # early_stopping=False: random validation splitで時系列を混ぜない。
            learner.fit(xs.reshape(-1, xs.shape[-1]), ys.reshape(-1))
            mu = ymean+yscale*learner.predict(xn)
        elif model.startswith("pc_pls_"):
            if float(np.max(ys.std(axis=0))) < 1e-10:
                mu = hist.copy()
                diag["constant_targets"] = True
            else:
                # 各因子をunit-variance化した相関PCA。原論文RPPCAではない。
                k_pc = min(params["pcs"], n_factor, n_time-1)
                pca = PCA(n_components=k_pc, svd_solver="full")
                pc = pca.fit_transform(ys)
                flat, flatnew = xs.reshape(n_time, -1), xn.reshape(1, -1)
                pc_predictions, pc_info = [], []
                for j in range(k_pc):
                    z, zn, info = pls_scores(flat, pc[:, j], flatnew, params["pls_components"])
                    if model in {"pc_pls_tvp", "pc_pls_static_kalman"}:
                        value, state = kalman_regression_predict(z, pc[:, j], zn,
                                                                params["lam"], params["q"])
                        info["kalman"] = state
                    else:
                        weights = None
                        if model == "pc_pls_ew":
                            weights = np.exp2(-(n_time-1-np.arange(n_time))/params["half_life"])
                        value = ridge_predict(z, pc[:, j], zn, params["lam"], weights)
                    pc_predictions.append(float(value[0]))
                    pc_info.append(info)
                mu = ymean+yscale*pca.inverse_transform(np.asarray(pc_predictions)[None, :])[0]
                diag.update({"pc_explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
                             "pc_details": pc_info,
                             "pca_coordinates_refit_and_refiltered": True})
        else:
            raise ValueError(f"未実装モデル: {model}")
    raw = np.asarray(mu, dtype=float).reshape(n_factor)
    if not np.isfinite(raw).all():
        raise FloatingPointError(f"{model}が非有限予測を返しました。")
    bound = config.prediction_clip_sigma*train_sigma
    clipped = np.clip(raw, hist-bound, hist+bound)
    diag["prediction_clip_count"] = int(np.sum(np.abs(clipped-raw) > 1e-12))
    return ForecastResult(clipped, raw, hist, train_sigma, diag)


# ========================= SECTION 04: NESTED WALK-FORWARD =====================
def select_candidate(spec: ModelSpec, x: np.ndarray, y: np.ndarray,
                     label_dates: pd.DatetimeIndex, decision_date: pd.Timestamp,
                     config: ResearchConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """過去の内側originごとにfitし直して1歩先予測。日時単位のrolling-origin。

    内側評価日のラベルは外側decision_dateまでに全て既知。
    スコアは因子ごとの学習標準偏差で割った誤差二乗の平均。
    候補の不具合は記録し除外。全候補失敗なら停止し、黙ってmeanへ戻さない。
    """
    start = max(config.inner_min_train, len(x)-config.inner_validation_months)
    origins = list(range(start, len(x), config.inner_stride))
    if not origins:
        raise ValueError("内側検証の観測数が不足しています。")
    rows, scores = [], []
    for candidate_id, params in enumerate(spec.candidates):
        errors: list[float] = []
        exception: str | None = None
        if len(spec.candidates) == 1:
            score = math.nan  # 固定候補はCVで選択していないことを明示。
        else:
            try:
                for v in origins:
                    result = forecast_once(spec.name, params, x[:v], y[:v], x[v], config)
                    error = np.mean(((y[v]-result.prediction)/result.train_sigma)**2)
                    if not np.isfinite(error):
                        raise FloatingPointError("内側検証scoreが非有限です。")
                    errors.append(float(error))
                score = float(np.mean(errors))
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                score, exception = math.inf, repr(exc)
                LOG.warning("CV候補の失敗: %s %s %s", spec.name, params, exception)
        scores.append(score)
        rows.append({"model": spec.name, "selection_date": decision_date,
                     "candidate_id": candidate_id, "params": jdump(params),
                     "scaled_validation_mse": score,
                     "inner_origins": len(errors), "error": exception,
                     "validation_target_dates": jdump([str(label_dates[v].date()) for v in origins])
                     if len(spec.candidates) > 1 else "[]",
                     "inner_scores": jdump(errors), "selected": False})
    if len(spec.candidates) == 1:
        chosen = 0
    else:
        finite = [i for i, s in enumerate(scores) if np.isfinite(s)]
        if not finite:
            raise RuntimeError(f"{spec.name}: 全CV候補が失敗しました。model_selectionを確認してください。")
        best = min(scores[i] for i in finite)
        threshold = best*(1+config.cv_tolerance) + 1e-12
        chosen = next(i for i in finite if scores[i] <= threshold)
    rows[chosen]["selected"] = True
    return dict(spec.candidates[chosen]), rows


def walk_forward(data: ResearchData, config: ResearchConfig, specs: Sequence[ModelSpec],
                 *, progress: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """外側の各月で再学習。ハイパーパラメータのみtune_everyか月ごとに更新。"""
    config.validate(len(data.dates), len(data.factors))
    predictions, selections, diagnostics = [], [], []
    selected: dict[str, dict[str, Any]] = {}
    start_time = time.perf_counter()
    with threadpool_limits(limits=config.threads):
        for t in range(config.min_train, len(data.dates)-1):
            start = 0 if config.train_window is None else max(0, t-config.train_window)
            x = data.features[start:t]
            y = data.returns[start+1:t+1]
            target_dates = data.dates[start+1:t+1]
            for spec in specs:
                if spec.name not in selected or (t-config.min_train) % config.tune_every == 0:
                    selected[spec.name], records = select_candidate(
                        spec, x, y, target_dates, data.dates[t], config)
                    selections.extend(records)
                params = selected[spec.name]
                result = forecast_once(spec.name, params, x, y, data.features[t], config)
                diagnostics.append({"decision_date": data.dates[t], "target_date": data.dates[t+1],
                                    "model": spec.name, "training_feature_start": data.dates[start],
                                    "training_target_end": data.dates[t], "params": jdump(params),
                                    "diagnostics": jdump(result.diagnostics)})
                for k, factor in enumerate(data.factors):
                    predictions.append({"decision_date": data.dates[t], "target_date": data.dates[t+1],
                                        "model": spec.name, "factor": factor,
                                        "prediction": result.prediction[k],
                                        "raw_prediction": result.raw_prediction[k],
                                        "historical_mean": result.historical_mean[k],
                                        "timing_prediction": result.prediction[k]-result.historical_mean[k],
                                        "train_sigma": result.train_sigma[k],
                                        "realized": data.returns[t+1, k]})
            if progress and ((t-config.min_train) % 12 == 0 or t == len(data.dates)-2):
                LOG.info("OOS %s -> %s / %d models / elapsed %.1fs",
                         data.dates[t].date(), data.dates[t+1].date(), len(specs),
                         time.perf_counter()-start_time)
    return pd.DataFrame(predictions), pd.DataFrame(selections), pd.DataFrame(diagnostics)

# ========================= SECTION 05: ALLOCATION =============================
def project_capped_simplex(v: np.ndarray, cap: float) -> np.ndarray:
    """min ||w-v||^2 subject to 0<=w<=cap,sum(w)=1。"""
    v = np.asarray(v, dtype=float)
    if v.ndim != 1 or not np.isfinite(v).all() or len(v)*cap < 1-1e-12:
        raise ValueError("capped simplexの入力/制約が不正です。")
    if abs(len(v)*cap-1) < 1e-12:
        return np.full(len(v), 1/len(v))
    lo, hi = float(np.min(v-cap)), float(np.max(v))
    for _ in range(80):
        mid = (lo+hi)/2
        if np.clip(v-mid, 0, cap).sum() > 1:
            lo = mid
        else:
            hi = mid
    w = np.clip(v-(lo+hi)/2, 0, cap)
    if abs(w.sum()-1) > 1e-9:
        raise FloatingPointError("simplexへの射影が収束しませんでした。")
    return w


def estimate_covariance(history: np.ndarray) -> np.ndarray:
    """historyは意思決定時点まで。Ledoit-Wolf縮小＋数値安定化。"""
    if len(history) < 3 or not np.isfinite(history).all():
        raise ValueError("共分散推定の観測が不足/不正です。")
    covariance = LedoitWolf().fit(history).covariance_
    ridge = max(float(np.trace(covariance)/len(covariance))*1e-8, 1e-12)
    return .5*(covariance+covariance.T)+ridge*np.eye(len(covariance))


def solve_active_allocation(signal: np.ndarray, covariance: np.ndarray,
                            pretrade: np.ndarray, config: ResearchConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """制約付き凸配分。a=w-equal。平均リターンと共分散は月次単位。

    min -signal'a + gamma/2*a'Sigma*a + anchor/2*||a||^2
        + turnover_proxy * sum(u)
    s.t. sum(w)=1, 0<=w<=cap, u>=|w-pretrade|,
         sqrt(12*a'Sigma*a)<=max_active_vol_annual.
    補助変数uでL1を明示。SLSQP失敗/制約違反時は黙って継続せず停止。
    pretradeは投資可能sleeveならdrift後、factor_pnlなら前回配分。
    """
    n = len(signal)
    base = np.full(n, 1/n)
    cap = config.max_weight(n)
    tc = config.turnover_penalty_bps/10000
    monthly_te2 = config.max_active_vol_annual**2/12
    identity = np.eye(n)
    def fun(v: np.ndarray) -> float:
        a = v[:n]-base
        return float(1000*(-signal @ a + .5*config.active_risk_aversion*(a @ covariance @ a)
                           + .5*config.anchor_penalty*(a @ a)+tc*v[n:].sum()))
    def jac(v: np.ndarray) -> np.ndarray:
        a = v[:n]-base
        return 1000*np.r_[-signal+config.active_risk_aversion*(covariance @ a)
                         + config.anchor_penalty*a, np.full(n, tc)]
    abs_jac = np.block([[-identity, identity], [identity, identity]])
    constraints = [
        {"type": "eq", "fun": lambda v: v[:n].sum()-1,
         "jac": lambda v: np.r_[np.ones(n), np.zeros(n)]},
        {"type": "ineq", "fun": lambda v: np.r_[v[n:]-(v[:n]-pretrade),
                                                    v[n:]+(v[:n]-pretrade)],
         "jac": lambda v: abs_jac},
        {"type": "ineq", "fun": lambda v: 1-((v[:n]-base) @ covariance @ (v[:n]-base))/monthly_te2,
         "jac": lambda v: np.r_[-2*(covariance @ (v[:n]-base))/monthly_te2, np.zeros(n)]},
    ]
    initial = np.r_[base, np.abs(base-pretrade)+1e-8]
    result = minimize(fun, initial, jac=jac, method="SLSQP",
                      bounds=[(0., cap)]*n+[(0., None)]*n,
                      constraints=constraints, options={"ftol": 1e-9, "maxiter": 300})
    w = result.x[:n]
    active_var = float((w-base) @ covariance @ (w-base))
    feasible = (np.isfinite(w).all() and abs(w.sum()-1) <= 1e-7
                and w.min() >= -1e-7 and w.max() <= cap+1e-7
                and active_var <= monthly_te2*(1+1e-5)+1e-12)
    if not result.success or not feasible:
        raise RuntimeError(f"配分最適化失敗: {result.message}; feasible={feasible}")
    # 限界近傍の丸めのみ。大幅な再射影で失敗を隠さない。
    w = np.clip(w, 0., cap)
    w = w/w.sum()
    return w, {"optimizer": "SLSQP", "iterations": int(result.nit), "success": True}


def target_weights(rule: str, prediction: np.ndarray, historical_mean: np.ndarray,
                   train_sigma: np.ndarray, covariance: np.ndarray,
                   pretrade: np.ndarray, config: ResearchConfig) -> tuple[np.ndarray, dict[str, Any]]:
    """予測値から配分への独立API。新ルールはこの関数に追加する。"""
    n = len(prediction)
    base = np.full(n, 1/n)
    cap = config.max_weight(n)
    if rule == "equal":
        return base, {"rule": rule}
    if rule == "inverse_vol":
        raw = 1/np.sqrt(np.maximum(np.diag(covariance), 1e-12))
        return project_capped_simplex(raw/raw.sum(), cap), {"rule": rule}
    timing = prediction-historical_mean
    if rule == "timing_rank":
        score = timing/np.maximum(train_sigma, 1e-6)
        ranks = rankdata(score)-((n+1)/2)
        if np.max(score)-np.min(score) < 1e-12:
            ranks[:] = 0
        tilt = config.tilt_l1_budget*ranks/max(float(np.abs(ranks).sum()), 1.)
        w = project_capped_simplex(base+tilt, cap)
        active = w-base
        active_vol = math.sqrt(max(0., 12*float(active @ covariance @ active)))
        if active_vol > config.max_active_vol_annual:
            w = base+active*(config.max_active_vol_annual/active_vol)
        return w, {"rule": rule, "l1_tilt_before_cap": config.tilt_l1_budget}
    if rule in {"timing_mv", "level_mv"}:
        signal = timing if rule == "timing_mv" else prediction
        return solve_active_allocation(config.prediction_shrinkage*signal,
                                       covariance, pretrade, config)
    raise ValueError(f"未知の配分ルール: {rule}")


def funded_rebalance_cost(target: np.ndarray, pretrade: np.ndarray, rate: float) -> tuple[float, float]:
    """NAV減少を考慮した自己資金制約付きスリーブ売買コスト。
    targetはコスト支払後の配分。pretradeはコスト支払前NAVに対する保有比率。
    """
    if not 0 <= rate < .1:
        raise ValueError("rateは[0,.1)です。")
    if rate == 0:
        return 0., float(np.abs(target-pretrade).sum())
    lo, hi = 0., 1.
    for _ in range(65):
        c = (lo+hi)/2
        f = c-rate*np.abs((1-c)*target-pretrade).sum()
        if f > 0:
            hi = c
        else:
            lo = c
    cost = (lo+hi)/2
    traded = float(np.abs((1-cost)*target-pretrade).sum())
    return cost, traded


# ========================= SECTION 06: BACKTEST / METRICS =======================
def safe_corr(a: np.ndarray, b: np.ndarray, *, rank: bool = False) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    a, b = a[good], b[good]
    if len(a) < 3:
        return math.nan
    if rank:
        a, b = rankdata(a), rankdata(b)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def signal_metrics(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def calculate(frame: pd.DataFrame) -> dict[str, float]:
        actual = frame["realized"].to_numpy()
        pred = frame["prediction"].to_numpy()
        mean = frame["historical_mean"].to_numpy()
        sigma = frame["train_sigma"].to_numpy()
        sse, benchmark_sse = np.sum((actual-pred)**2), np.sum((actual-mean)**2)
        return {"observations": len(frame), "rmse": float(np.sqrt(np.mean((actual-pred)**2))),
                "mae": float(np.mean(np.abs(actual-pred))),
                "oos_r2_vs_past_mean": float(1-sse/benchmark_sse) if benchmark_sse > EPS else math.nan,
                "scaled_mse": float(np.mean(((actual-pred)/sigma)**2)),
                "sign_accuracy": float(np.mean(np.sign(actual) == np.sign(pred))),
                "timing_direction_accuracy": float(np.mean(np.sign(actual-mean) == np.sign(pred-mean))),
                "timing_correlation": safe_corr(actual-mean, pred-mean)}
    per_factor = []
    for (model, factor), frame in predictions.groupby(["model", "factor"], sort=False):
        per_factor.append({"model": model, "factor": factor, **calculate(frame)})
    monthly = []
    for (model, date), frame in predictions.groupby(["model", "target_date"], sort=False):
        monthly.append({"model": model, "target_date": date,
                        "rank_ic_level": safe_corr(frame["prediction"].to_numpy(), frame["realized"].to_numpy(), rank=True),
                        "rank_ic_timing": safe_corr(frame["timing_prediction"].to_numpy(),
                                                   (frame["realized"]-frame["historical_mean"]).to_numpy(), rank=True)})
    monthly_frame = pd.DataFrame(monthly)
    summary = []
    for model, frame in predictions.groupby("model", sort=False):
        ic = monthly_frame.loc[monthly_frame["model"] == model]
        summary.append({"model": model, **calculate(frame),
                        "oos_months": frame["target_date"].nunique(),
                        "rank_ic_level_mean": ic["rank_ic_level"].mean(),
                        "rank_ic_timing_mean": ic["rank_ic_timing"].mean(),
                        "rank_ic_timing_valid_months": ic["rank_ic_timing"].notna().sum()})
    pivot = predictions.pivot(index=["target_date", "factor"], columns="model", values="timing_prediction")
    correlation = pivot.corr().rename_axis("model").reset_index()
    return pd.DataFrame(summary), pd.DataFrame(per_factor), monthly_frame, correlation


def hac_t_stat_mean(values: np.ndarray, lags: int) -> float:
    """Bartlett Newey-West型の平均のt統計。多重検定補正済みではない。"""
    v = np.asarray(values, dtype=float)
    if len(v) < 4:
        return math.nan
    z = v-v.mean()
    n = len(v)
    lag_count = min(lags, n-1)
    long_var = float(z @ z/n)
    for lag in range(1, lag_count+1):
        long_var += 2*(1-lag/(lag_count+1))*float(z[lag:] @ z[:-lag]/n)
    if long_var <= EPS:
        return math.nan
    return float(v.mean()/math.sqrt(long_var/n))


def performance_metrics(returns: np.ndarray, benchmark: np.ndarray | None = None,
                        hac_lags: int = 3) -> dict[str, Any]:
    r = np.asarray(returns, dtype=float)
    if not np.isfinite(r).all() or len(r) < 2:
        raise ValueError("performance_metricsのリターンに不備があります。")
    if np.any(r <= -1):
        raise ValueError("戦略NAVが0以下になる月があります。通常のCAGR計算を停止します。")
    ann_mean = 12*float(r.mean())
    vol = math.sqrt(12)*float(r.std(ddof=1))
    wealth = np.r_[1., np.cumprod(1+r)]
    metrics: dict[str, Any] = {
        "months": len(r), "annualized_mean": ann_mean, "annualized_vol": vol,
        "mean_vol_ratio_rf0": ann_mean/vol if vol > EPS else math.nan,
        "cagr": float(np.expm1(np.log1p(r).sum()*12/len(r))),
        "max_drawdown": float(np.min(wealth/np.maximum.accumulate(wealth)-1)),
        "ending_wealth": float(wealth[-1]),
    }
    if benchmark is not None:
        active = r-np.asarray(benchmark, dtype=float)
        active_vol = math.sqrt(12)*float(active.std(ddof=1))
        metrics.update({"annualized_difference_vs_equal": 12*float(active.mean()),
                        "realized_active_vol_vs_equal": active_vol,
                        "information_ratio_vs_equal": 12*float(active.mean())/active_vol if active_vol > EPS else math.nan,
                        "hac_t_vs_equal_unadjusted": hac_t_stat_mean(active, hac_lags)})
    return metrics


def simulate_portfolios(data: ResearchData, predictions: pd.DataFrame, config: ResearchConfig,
                        *, rules: Sequence[str] = ("timing_rank", "timing_mv", "level_mv")) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """固定予測から戦略比較。新戦略を追加しても予測は再推定しなくてよい。

    cost_sensitivityは同じtarget weightsを保持した費用感応度。費用別の再最適化ではない。
    """
    config.validate(len(data.dates), len(data.factors))
    if config.return_kind == "sleeve_total" and np.any(data.returns <= -1):
        raise ValueError("sleeve_totalでは各資産のreturn>-1が必要です。")
    models = list(dict.fromkeys(predictions["model"]))
    strategies: list[tuple[str, str, str]] = [
        ("baseline_equal", "baseline", "equal"),
        ("baseline_inverse_vol", "baseline", "inverse_vol"),
        ("baseline_initial_fixed", "baseline", "initial_fixed"),
    ]
    for model in models:
        strategies.extend((f"{model}__{rule}", model, rule) for rule in rules)
    n = len(data.factors)
    pretrade = {name: np.zeros(n) for name, _, _ in strategies}
    fixed_initial: np.ndarray | None = None
    date_to_idx = {date: i for i, date in enumerate(data.dates)}
    records, weights, scenarios = [], [], []
    grouped = predictions.groupby("target_date", sort=True)
    expected_dates = data.dates[config.min_train+1:]
    if list(grouped.groups) != list(expected_dates):
        raise ValueError("predictionsのOOS日付がconfigと一致しません。")
    with threadpool_limits(limits=config.threads):
        for target_date, monthly in grouped:
            t = date_to_idx[target_date]-1
            decision_date = data.dates[t]
            if not (monthly["decision_date"] == decision_date).all():
                raise ValueError("予測と実現リターンの時点が対応していません。")
            covariance = estimate_covariance(data.returns[max(0, t+1-config.covariance_window):t+1])
            latest = data.returns[t+1]
            mean_slice = max(1, t+1-(config.train_window or t))
            past_mean = data.returns[mean_slice:t+1].mean(axis=0)
            sigma = np.maximum(data.returns[mean_slice:t+1].std(axis=0, ddof=1), 1e-6)
            if fixed_initial is None:
                fixed_initial, _ = solve_active_allocation(config.prediction_shrinkage*past_mean,
                                                           covariance, np.zeros(n), config)
            model_vectors = {}
            for model in models:
                frame = monthly.loc[monthly["model"] == model].set_index("factor")
                if len(frame) != n or frame.index.duplicated().any() or set(frame.index) != set(data.factors):
                    raise ValueError(f"{model}の予測因子に欠落/重複があります。")
                frame = frame.reindex(data.factors)
                model_vectors[model] = (frame["prediction"].to_numpy(),
                                        frame["historical_mean"].to_numpy(), frame["train_sigma"].to_numpy())
            for name, model, rule in strategies:
                previous = pretrade[name].copy()
                if rule == "initial_fixed":
                    w, info = fixed_initial.copy(), {"rule": "fixed_from_initial_training_only"}
                else:
                    pred, mean, scale = model_vectors.get(model, (past_mean, past_mean, sigma))
                    w, info = target_weights(rule, pred, mean, scale, covariance, previous, config)
                if abs(w.sum()-1) > 1e-7 or w.min() < -1e-8 or w.max() > config.max_weight(n)+1e-7:
                    raise FloatingPointError("配分制約違反を検出しました。")
                gross = float(w @ latest)
                rate = config.cost_bps/10000
                if config.return_kind == "sleeve_total":
                    cost, traded = funded_rebalance_cost(w, previous, rate)
                    net = (1-cost)*(1+gross)-1
                else:
                    traded = float(np.abs(w-previous).sum())
                    cost = rate*traded
                    net = gross-cost
                if net <= -1:
                    raise ValueError(f"{name} {target_date}: NAV破綻。レバレッジ/リターン単位を確認してください。")
                active = w-1/n
                records.append({"target_date": target_date, "decision_date": decision_date,
                                "strategy": name, "model": model, "rule": rule,
                                "gross_return": gross, "net_return": net, "cost_fraction": cost,
                                "traded_notional_fraction": traded,
                                "half_l1_turnover": .5*traded,
                                "predicted_vol_annual": math.sqrt(max(0., 12*float(w @ covariance @ w))),
                                "predicted_active_vol_annual": math.sqrt(max(0., 12*float(active @ covariance @ active))),
                                "max_weight": float(w.max()), "allocation_diagnostics": jdump(info),
                                "return_kind": config.return_kind})
                for k, factor in enumerate(data.factors):
                    weights.append({"target_date": target_date, "decision_date": decision_date,
                                    "strategy": name, "model": model, "rule": rule, "factor": factor,
                                    "weight": w[k], "pretrade_weight": previous[k], "active_weight": active[k]})
                for bps in sorted(set((*config.cost_scenarios_bps, config.cost_bps))):
                    if config.return_kind == "sleeve_total":
                        alt_cost, _ = funded_rebalance_cost(w, previous, bps/10000)
                        alt_net = (1-alt_cost)*(1+gross)-1
                    else:
                        alt_cost = bps/10000*traded
                        alt_net = gross-alt_cost
                    scenarios.append({"target_date": target_date, "strategy": name, "cost_bps": bps,
                                      "net_return": alt_net, "cost_fraction": alt_cost})
                pretrade[name] = (w*(1+latest)/(1+gross)
                                 if config.return_kind == "sleeve_total" else w.copy())
    monthly_returns = pd.DataFrame(records)
    weights_frame = pd.DataFrame(weights)
    benchmark = monthly_returns.loc[monthly_returns["strategy"] == "baseline_equal"].set_index("target_date")["net_return"]
    summaries = []
    for name, frame in monthly_returns.groupby("strategy", sort=False):
        frame = frame.sort_values("target_date")
        summaries.append({"strategy": name, "model": frame["model"].iloc[0], "rule": frame["rule"].iloc[0],
                          **performance_metrics(frame["net_return"].to_numpy(), benchmark.loc[frame["target_date"]].to_numpy(), config.hac_lags),
                          "annualized_cost_fraction": 12*frame["cost_fraction"].mean(),
                          "annualized_traded_notional": 12*frame["traded_notional_fraction"].mean(),
                          "average_predicted_active_vol": frame["predicted_active_vol_annual"].mean(),
                          "return_kind": config.return_kind})
    scenario_frame = pd.DataFrame(scenarios)
    sensitivity = []
    for (name, bps), frame in scenario_frame.groupby(["strategy", "cost_bps"], sort=False):
        sensitivity.append({"strategy": name, "cost_bps": bps,
                            **performance_metrics(frame.sort_values("target_date")["net_return"].to_numpy()),
                            "annualized_cost_fraction": 12*frame["cost_fraction"].mean(),
                            "positions_reoptimized_for_this_cost": False})
    return monthly_returns, weights_frame, pd.DataFrame(summaries), pd.DataFrame(sensitivity)


def block_bootstrap_comparison(strategy_returns: pd.DataFrame, config: ResearchConfig) -> pd.DataFrame:
    """全戦略で同じ月ブロックを抽出するpaired circular moving-block bootstrap。
    学習の再実行やモデル選択誤差、多重比較補正は含まない。
    """
    columns = ["strategy", "annualized_difference_vs_equal", "ci_2_5", "ci_97_5", "samples",
               "block_months", "inference_scope"]
    if config.bootstrap_samples == 0:
        return pd.DataFrame(columns=columns)
    wide = strategy_returns.pivot(index="target_date", columns="strategy", values="net_return").sort_index()
    if wide.isna().any().any():
        raise ValueError("bootstrapには全戦略共通の評価期間が必要です。")
    difference = wide.sub(wide["baseline_equal"], axis=0)
    t = len(wide)
    block = min(config.bootstrap_block, t)
    rng = np.random.default_rng(config.seed)
    starts = rng.integers(0, t, size=(config.bootstrap_samples, math.ceil(t/block)))
    indices = ((starts[..., None]+np.arange(block)) % t).reshape(config.bootstrap_samples, -1)[:, :t]
    output = []
    for name in difference:
        values = difference[name].to_numpy()
        means = 12*values[indices].mean(axis=1)
        output.append({"strategy": name, "annualized_difference_vs_equal": 12*values.mean(),
                       "ci_2_5": float(np.quantile(means, .025)), "ci_97_5": float(np.quantile(means, .975)),
                       "samples": config.bootstrap_samples, "block_months": block,
                       "inference_scope": "conditional fixed OOS paths; NOT multiple-testing-adjusted; NOT refitted"})
    return pd.DataFrame(output, columns=columns)


def mean_weight_diagnostic(data: ResearchData, strategy_returns: pd.DataFrame,
                           weights: pd.DataFrame) -> pd.DataFrame:
    """事後の平均配分に対するgross差を分解。絶対に投資可能なOOS戦略として扱わない。"""
    rframe = pd.DataFrame(data.returns, index=data.dates, columns=data.factors)
    out = []
    for strategy, frame in weights.groupby("strategy", sort=False):
        mean_weight = frame.groupby("factor")["weight"].mean().reindex(data.factors).to_numpy()
        realized = strategy_returns.loc[strategy_returns["strategy"] == strategy].sort_values("target_date")
        static_gross = rframe.loc[realized["target_date"]].to_numpy() @ mean_weight
        dynamic_gross = realized["gross_return"].to_numpy()
        out.append({"strategy": strategy,
                    "dynamic_gross_annualized_mean": 12*float(dynamic_gross.mean()),
                    "expost_mean_weight_gross_annualized_mean": 12*float(static_gross.mean()),
                    "gross_timing_difference_annualized": 12*float((dynamic_gross-static_gross).mean()),
                    "is_investable_oos_benchmark": False,
                    "label": "EX_POST_DIAGNOSTIC_ONLY; average weights use entire OOS; gross only"})
    return pd.DataFrame(out)

# ========================= SECTION 07: REPORTING / PUBLIC API ==================
def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], limit: int | None = None) -> str:
    frame = frame[list(columns)]
    if limit is not None:
        frame = frame.head(limit)
    def text(v: Any) -> str:
        if isinstance(v, (float, np.floating)):
            return f"{v:.5f}" if np.isfinite(v) else "NA"
        return str(v).replace("|", "/")
    header = "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"]*len(columns)) + " |"
    body = "\n".join("| " + " | ".join(text(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    return header+"\n"+body


def build_manifest(data: ResearchData, config: ResearchConfig, specs: Sequence[ModelSpec],
                   rules: Sequence[str], elapsed: float) -> dict[str, Any]:
    dependencies = {}
    for package in ["numpy", "pandas", "scipy", "scikit-learn", "threadpoolctl"]:
        dependencies[package] = importlib.metadata.version(package)
    path = Path(__file__)
    return {"program": "factor_tilt_research", "version": VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed, "python_version": platform.python_version(),
            "dependencies": dependencies,
            "script_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
            "config": dataclasses.asdict(config), "models": [dataclasses.asdict(s) for s in specs],
            "allocation_rules": list(rules), "factor_names": list(data.factors),
            "feature_names": list(data.feature_names),
            "input_start": str(data.dates[0].date()), "input_end": str(data.dates[-1].date()),
            "input_months": len(data.dates), "factor_count": len(data.factors),
            "feature_count": len(data.feature_names),
            "oos_start": str(data.dates[config.min_train+1].date()),
            "oos_end": str(data.dates[-1].date()),
            "oos_months": len(data.dates)-config.min_train-1,
            "data_metadata": data.metadata,
            "assumptions": [
                "one-month next-period forecast; monthly end-of-period execution approximation",
                "PIT source validity and factor discovery dates not independently verified",
                "per-origin refit of transformations and Kalman refilter in new coordinates",
                "no cross-factor random split; hyperparameters selected on past targets only",
                "long-only sleeve allocations are not underlying-security long-only holdings",
                "factor_pnl costs only proxy factor reallocations, not security-level execution",
                "rf0 ratio assumes zero hurdle; no risk-free return input",
                "bootstrap does not refit models or correct multiple testing",
                "expost mean-weight reference is not an investable OOS benchmark",
                "not an exact replication of Kagkadis et al.; no RPPCA or Lasso",
            ]}


def export_results(result: ExperimentResult, out: str | Path) -> None:
    directory = Path(out)
    directory.mkdir(parents=True, exist_ok=True)
    for item in dataclasses.fields(result):
        value = getattr(result, item.name)
        if isinstance(value, pd.DataFrame):
            value.to_csv(directory/f"{item.name}.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    (directory/"manifest.json").write_text(json.dumps(jsonable(result.manifest), ensure_ascii=False,
                                                    indent=2, allow_nan=False), encoding="utf-8")
    m = result.manifest
    synthetic = bool(m["data_metadata"].get("synthetic", False))
    tag = "合成データによる動作検証。実データの運用効果を示す結果ではありません。" if synthetic else "ユーザー指定データの研究用比較。PIT・因子採掘・執行可能性は別途監査してください。"
    sig = result.signal_summary.sort_values("oos_r2_vs_past_mean", ascending=False)
    strategies = result.strategy_summary.sort_values("mean_vol_ratio_rf0", ascending=False)
    report = f"""# Factor Tilt Research Lab 比較結果

**{tag}**

入力: {m['input_start']}～{m['input_end']} / {m['input_months']}か月 /
{m['factor_count']}ファクター / 1因子あたり{m['feature_count']}特性。
OOS: {m['oos_start']}～{m['oos_end']} ({m['oos_months']}か月)。
{len(m['models'])}予測モデル、{len(result.strategy_summary)}配分戦略。
return_kind={m['config']['return_kind']} / 費用={m['config']['cost_bps']}bps per traded notional。

## 予測比較

OOS R2の基準は各予測時点の学習期間平均です。ICは月ごとの横断順位相関です。
表の順序は表示用で、外側OOSを用いたモデル採用を正当化するものではありません。
meanのtiming ICがNAになるのは、timing予測が常にゼロだからです。

{_markdown_table(sig, ['model', 'oos_r2_vs_past_mean', 'rank_ic_level_mean', 'rank_ic_timing_mean', 'rmse'])}

## 戦略比較 (表示は先頭15行、全件はstrategy_summary.csv)

rf0比率は無リスク金利ゼロのmean/volです。多重比較を調整した成績ではありません。
コスト・リスク量が異なるため、この比率だけでモデルを選ばないでください。

{_markdown_table(strategies, ['strategy', 'annualized_mean', 'annualized_vol', 'mean_vol_ratio_rf0', 'max_drawdown', 'annualized_cost_fraction'], 15)}

## 読み方・次のChatGPTへの引継ぎ

- predictions.csv: predictionは水準、timing_predictionは各因子の過去平均からの差。
  target_dateは実現リターンの月、decision_dateは1か月前。
- signal_summary / signal_by_factor / signal_monthly_ic: 予測能力を戦略損益と切り離して比較。
- model_selection: 各候補の内側検証scoreと採用履歴。外側成績を候補選択に使わない。
- model_diagnostics: 学習開始・最終ラベル日・PCA説明率・最後のKalman状態など。
  PCA座標は毎回再推定されるため、係数を座標整合なしで時系列比較しない。
- weights / strategy_returns: 同じ予測から別の配分・コストを評価可能。
- cost_sensitivity: 同一ポジションを費用だけ変えて再計算。費用別の再最適化ではない。
- bootstrap_vs_equal: 同じ月ブロックで全戦略を再標本化。CIは条件付きで、
  モデル再学習・研究者のモデル選択・多重検定の不確実性を含まない。
- expost_mean_weight_diagnostic: 全OOS平均ウェイトを使う**事後診断**。
  これは先読みのある比較対象で、投資可能なベンチマークではない。
- baseline_initial_fixed: 初回学習期間だけで配分を決め、その目標配分へ毎月戻す比較対象。
- 入力・出力・APIは.py先頭のdocstringを参照。SECTION単位でモジュール分割可能。

## 重要な限界

月次の月末情報・月末約定は近似です。発表時刻と実際の執行価格は別途対応が必要です。
ファクターP&Lの場合、原資産の売買相殺、因子内部回転、借株料、資金調達料、
税金、流動性制約は未実装です。38因子の配分合計1は原資産のgross=1ではありません。
結果が良いモデルをこのOOSから選んだ後は、独立holdoutまたは前向き検証が必要です。
データ期間・候補グリッド・特徴量を変えた試行も研究ログへ残してください。
"""
    (directory/"REPORT.md").write_text(report, encoding="utf-8")


def prepare_output_directory(out: str | Path, overwrite: bool = False) -> Path:
    directory = Path(out)
    if directory.exists() and any(directory.iterdir()) and not overwrite:
        raise FileExistsError(f"出力先が空ではありません: {directory}。別フォルダまたは --overwrite を指定してください。")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_experiment(data: ResearchData, config: ResearchConfig | None = None,
                   models: str | Sequence[str] | Sequence[ModelSpec] = "core",
                   *, out: str | Path | None = None, overwrite: bool = False,
                   rules: Sequence[str] = ("timing_rank", "timing_mv", "level_mv"),
                   progress: bool = True) -> ExperimentResult:
    """主API。入力を替え、models/rules/configだけで比較可能。ネットアクセスなし。"""
    config = config or ResearchConfig()
    config.validate(len(data.dates), len(data.factors))
    if not isinstance(models, str) and len(models) and isinstance(models[0], ModelSpec):
        specs = list(models)
    else:
        specs = resolve_models(models)  # type: ignore[arg-type]
    if len({s.name for s in specs}) != len(specs) or not specs:
        raise ValueError("モデル名が重複、またはモデルが空です。")
    if not rules or len(set(rules)) != len(rules) or any(r not in {"timing_rank", "timing_mv", "level_mv"} for r in rules):
        raise ValueError("rulesは重複しないtiming_rank,timing_mv,level_mvから指定してください。")
    directory = prepare_output_directory(out, overwrite) if out is not None else None
    started = time.perf_counter()
    pred, selection, diag = walk_forward(data, config, specs, progress=progress)
    ss, sf, sm, corr = signal_metrics(pred)
    sr, weights, strategy_summary, sensitivity = simulate_portfolios(data, pred, config, rules=rules)
    boot = block_bootstrap_comparison(sr, config)
    expost = mean_weight_diagnostic(data, sr, weights)
    manifest = build_manifest(data, config, specs, rules, time.perf_counter()-started)
    result = ExperimentResult(pred, selection, diag, ss, sf, sm, corr, sr, weights,
                              strategy_summary, sensitivity, boot, expost, manifest)
    if directory is not None:
        export_results(result, directory)
    return result


# ========================= SECTION 08: SYNTHETIC / SELF-TESTS ===================
def make_synthetic(months: int = 161, n_factors: int = 38, n_features: int = 6,
                   seed: int = 42) -> tuple[ResearchData, pd.DataFrame, pd.DataFrame]:
    """相関・係数ドリフト・交互作用を含む動作検証データ。研究上のalphaではない。"""
    if months < 20 or n_factors < 2 or n_features < 2:
        raise ValueError("合成データはmonths>=20, factors>=2, features>=2が必要です。")
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2013-06-30", periods=months, freq=pd.offsets.MonthEnd())
    factors = [f"F{k+1:02d}" for k in range(n_factors)]
    preferred = ["x_value_spread", "x_profitability", "x_fundamental_change",
                 "x_quality", "x_leverage", "x_investment"]
    feature_names = (preferred+[f"x_extra_{j+1}" for j in range(max(0, n_features-6))])[:n_features]
    x = np.zeros((months, n_factors, n_features))
    x[0] = rng.normal(size=(n_factors, n_features))
    for t in range(1, months):
        x[t] = .82*x[t-1]+.38*rng.normal(size=(n_factors, n_features))+.15*rng.normal(size=(1, n_features))
    intercept = .002+rng.normal(0, .001, n_factors)
    base = np.zeros(n_features)
    initial = np.array([.003, .002, .002, .001, -.0015, -.001])
    base[:min(6, n_features)] = initial[:min(6, n_features)]
    deviations = rng.normal(0, .0006, size=(n_factors, n_features))
    loadings = rng.normal(0, .6, size=(n_factors, 3))
    returns = np.empty((months, n_factors))
    returns[0] = intercept+.025*rng.normal(size=n_factors)
    for t in range(1, months):
        dynamic = base.copy()
        dynamic[0] *= .5+np.sin(2*np.pi*t/90)
        linear = np.sum(x[t-1]*(dynamic+deviations), axis=1)
        interaction = .002*np.tanh(x[t-1, :, 0]*x[t-1, :, 1])
        noise = loadings @ rng.normal(0, .015, 3)+rng.normal(0, .02, n_factors)
        returns[t] = intercept+linear+interaction+noise
    observed = x.copy()
    observed[rng.random(observed.shape) < .015] = np.nan
    rf = pd.DataFrame(returns, columns=factors)
    rf.insert(0, "date", dates)
    ff = pd.DataFrame(observed.reshape(months*n_factors, n_features), columns=feature_names)
    ff.insert(0, "available_at", np.repeat((dates-pd.Timedelta(days=1)).to_numpy(), n_factors))
    ff.insert(0, "factor", np.tile(factors, months))
    ff.insert(0, "date", np.repeat(dates.to_numpy(), n_factors))
    data = build_dataset(rf, ff, metadata={"synthetic": True, "synthetic_seed": seed,
                         "synthetic_dgp": "shared linear + drifting slope + interaction + correlated noise"})
    return data, rf, ff


def self_test() -> dict[str, Any]:
    """外部テストフレームワーク不要。境界・数理・先読み・CSV出力を検証する。"""
    import tempfile
    checks: list[tuple[str, Callable[[], None]]] = []
    def check(name: str) -> Callable[[Callable[[], None]], Callable[[], None]]:
        def decorate(function: Callable[[], None]) -> Callable[[], None]:
            checks.append((name, function))
            return function
        return decorate
    def raises(kind: type[BaseException], function: Callable[[], Any]) -> None:
        try:
            function()
        except kind:
            return
        raise AssertionError(f"{kind.__name__}が発生しませんでした。")
    data, rf, ff = make_synthetic(48, 4, 3, 17)
    cfg = ResearchConfig(min_train=24, inner_min_train=16, inner_validation_months=6,
                         inner_stride=3, tune_every=12, bootstrap_samples=40, seed=17)

    @check("input_shape_and_no_mutation")
    def _() -> None:
        before = rf.copy(deep=True)
        d = build_dataset(rf, ff)
        assert d.features.shape == (48, 4, 3)
        assert d.metadata["availability_timestamp_checks_passed"]
        pd.testing.assert_frame_equal(rf, before)

    @check("duplicate_month_rejected")
    def _() -> None:
        raises(ValueError, lambda: build_dataset(pd.concat([rf, rf.iloc[:1]]), ff))

    @check("missing_month_rejected")
    def _() -> None:
        raises(ValueError, lambda: build_dataset(rf.drop(index=5), ff[ff["date"] != rf.loc[5, "date"]]))

    @check("missing_return_rejected")
    def _() -> None:
        bad = rf.copy()
        bad.loc[5, "F01"] = np.nan
        raises(ValueError, lambda: build_dataset(bad, ff))

    @check("future_available_at_rejected")
    def _() -> None:
        bad = ff.copy()
        bad.loc[0, "available_at"] = bad.loc[0, "date"]+pd.Timedelta(days=1)
        raises(ValueError, lambda: build_dataset(rf, bad))

    @check("unverified_pit_requires_opt_in")
    def _() -> None:
        no_stamp = ff.drop(columns="available_at")
        raises(ValueError, lambda: build_dataset(rf, no_stamp))
        unverified = build_dataset(rf, no_stamp, allow_unverified_pit=True)
        assert not unverified.metadata["availability_timestamp_checks_passed"]

    @check("asof_join_does_not_backfill_future")
    def _() -> None:
        events = pd.DataFrame({"factor": ["F01", "F01"],
                               "available_at": pd.to_datetime(["2013-06-15", "2013-07-10"]),
                               "x_a": [1., 2.]})
        snap = snapshots_from_events(events, pd.to_datetime(["2013-05-31", "2013-06-30", "2013-07-31"]), ["F01"], ["x_a"])
        assert np.isnan(snap.iloc[0]["x_a"])
        assert snap.iloc[1]["x_a"] == 1. and snap.iloc[2]["x_a"] == 2.

    @check("training_only_scaler_and_all_missing_column")
    def _() -> None:
        raw = np.arange(40., dtype=float).reshape(10, 2, 2)
        raw[:, 0, 1] = np.nan
        scaler = PanelScaler().fit(raw)
        train = scaler.transform(raw)
        median = scaler.median.copy()
        assert np.isfinite(train).all()
        assert np.all(train[:, 0, 1] == 0)
        scaler.transform(np.full((1, 2, 2), 1e9))
        np.testing.assert_array_equal(median, scaler.median)

    @check("kalman_q0_equals_ridge")
    def _() -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=(60, 3))
        x = (x-x.mean(axis=0))/x.std(axis=0)
        y = x @ np.array([.3, -.2, .1])+rng.normal(size=60)
        xn = rng.normal(size=(1, 3))
        kalman, _ = kalman_regression_predict(x, y, xn, .8, 0.)
        ridge = ridge_predict(x, y, xn, .8)
        np.testing.assert_allclose(kalman, ridge, atol=1e-10, rtol=1e-10)

    @check("partial_pool_matches_dense_objective")
    def _() -> None:
        rng = np.random.default_rng(2)
        t, n, p = 20, 3, 2
        x = rng.normal(size=(t, n, p)); x -= x.mean(axis=0)
        y = rng.normal(size=(t, n)); y -= y.mean(axis=0)
        xn = rng.normal(size=(n, p))
        a, b = .4, 2.
        pred, _ = partial_pool_predict(x, y, xn, a, b)
        design = np.zeros((t*n, p*(n+1)))
        for ti in range(t):
            for k in range(n):
                row = ti*n+k
                design[row, :p] = x[ti, k]
                design[row, p*(k+1):p*(k+2)] = x[ti, k]
        penalty = np.r_[np.full(p, a), np.full(n*p, b/n)]
        beta = np.linalg.solve(design.T @ design/(t*n)+np.diag(penalty), design.T @ y.reshape(-1)/(t*n))
        truth = np.array([xn[k] @ (beta[:p]+beta[p*(k+1):p*(k+2)]) for k in range(n)])
        np.testing.assert_allclose(pred, truth, atol=1e-10)

    @check("partial_pool_strong_deviation_penalty_approaches_pool")
    def _() -> None:
        x = PanelScaler().fit_transform(data.features[:30])
        y, _, _ = normalize_target(data.returns[1:31])
        xn = x[-1]
        shared = ridge_predict(x.reshape(-1, 3), y.reshape(-1), xn, .5)
        partial, _ = partial_pool_predict(x, y, xn, .5, 1e8)
        np.testing.assert_allclose(shared, partial, atol=1e-7)

    @check("capped_simplex_bounds")
    def _() -> None:
        w = project_capped_simplex(np.array([-2., 10., 3., .1]), .4)
        assert abs(w.sum()-1) < 1e-10 and w.min() >= 0 and w.max() <= .4+1e-10
        raises(ValueError, lambda: project_capped_simplex(np.ones(4), .2))

    @check("funded_initial_cost_has_no_half_turnover_error")
    def _() -> None:
        rate = .001
        c, traded = funded_rebalance_cost(np.full(4, .25), np.zeros(4), rate)
        np.testing.assert_allclose(c, rate/(1+rate), atol=1e-12)
        np.testing.assert_allclose(c, rate*traded, atol=1e-12)
        c0, _ = funded_rebalance_cost(np.full(4, .25), np.full(4, .25), rate)
        assert c0 < 1e-12

    @check("optimizer_budget_bounds_and_active_risk")
    def _() -> None:
        covariance = estimate_covariance(data.returns[:24])
        w, info = solve_active_allocation(np.array([.02, -.01, .01, -.02]), covariance, np.full(4, .25), cfg)
        a = w-.25
        assert info["success"] and abs(w.sum()-1) < 1e-7
        assert w.max() <= cfg.max_weight(4)+1e-7 and w.min() >= -1e-8
        assert 12*(a @ covariance @ a) <= cfg.max_active_vol_annual**2*(1+1e-5)

    @check("all_12_model_families_return_finite_predictions")
    def _() -> None:
        for name, spec in model_registry().items():
            forecast = forecast_once(name, spec.candidates[-1], data.features[:30], data.returns[1:31], data.features[30], cfg)
            assert forecast.prediction.shape == (4,), name
            assert np.isfinite(forecast.prediction).all(), name

    @check("pc_pls_static_kalman_matches_same_static_ridge")
    def _() -> None:
        params = {"pcs": 3, "pls_components": 2, "lam": 1.0, "q": 0.}
        a = forecast_once("pc_pls_ridge", params, data.features[:30], data.returns[1:31], data.features[30], cfg)
        b = forecast_once("pc_pls_static_kalman", params, data.features[:30], data.returns[1:31], data.features[30], cfg)
        np.testing.assert_allclose(a.raw_prediction, b.raw_prediction, atol=1e-10)

    @check("nested_oos_prefix_invariance_including_tvp")
    def _() -> None:
        specs = resolve_models("mean,pooled_ridge,pc_pls_tvp")
        full, _, _ = walk_forward(data, cfg, specs, progress=False)
        prefix, _, _ = walk_forward(data.prefix(42), cfg, specs, progress=False)
        selected = full[full["target_date"] <= data.dates[41]].reset_index(drop=True)
        pd.testing.assert_frame_equal(selected, prefix.reset_index(drop=True))
        # 最後の教師ラベルはdecision_dateより未来でない。
        _, selection, diag = walk_forward(data.prefix(30), cfg, specs, progress=False)
        assert (diag["training_target_end"] <= diag["decision_date"]).all()
        for row in selection.itertuples():
            for date in json.loads(row.validation_target_dates):
                assert pd.Timestamp(date) <= row.selection_date

    @check("future_mutation_cannot_change_earlier_forecasts")
    def _() -> None:
        specs = resolve_models("pooled_ridge,pc_pls_tvp")
        original, _, _ = walk_forward(data, cfg, specs, progress=False)
        modified = data.prefix(len(data.dates))
        modified.features[40:] += 1e5
        modified.returns[40:] += .2
        other, _, _ = walk_forward(modified, cfg, specs, progress=False)
        columns = ["decision_date", "model", "factor", "prediction", "raw_prediction"]
        a = original.loc[original["decision_date"] < data.dates[40], columns].reset_index(drop=True)
        b = other.loc[other["decision_date"] < data.dates[40], columns].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, b)

    @check("one_month_alignment_and_proxy_initial_cost")
    def _() -> None:
        pred, _, _ = walk_forward(data.prefix(30), cfg, resolve_models("mean"), progress=False)
        first = pred.iloc[0]
        k = data.factors.index(first["factor"])
        np.testing.assert_allclose(first["realized"], data.returns[25, k])
        monthly, weights, _, _ = simulate_portfolios(data.prefix(30), pred, cfg, rules=("timing_rank",))
        equal = monthly[monthly["strategy"] == "baseline_equal"].iloc[0]
        np.testing.assert_allclose(equal["cost_fraction"], cfg.cost_bps/10000, atol=1e-12)
        np.testing.assert_allclose(equal["net_return"], data.returns[25].mean()-cfg.cost_bps/10000, atol=1e-12)
        sums = weights.groupby(["strategy", "target_date"])["weight"].sum()
        np.testing.assert_allclose(sums.to_numpy(), 1., atol=1e-7)

    @check("funded_sleeve_drift_and_cost_equation")
    def _() -> None:
        funded_cfg = dataclasses.replace(cfg, return_kind="sleeve_total")
        pred, _, _ = walk_forward(data.prefix(30), funded_cfg, resolve_models("mean"), progress=False)
        monthly, weights, _, _ = simulate_portfolios(data.prefix(30), pred, funded_cfg, rules=("timing_rank",))
        equal = monthly[monthly["strategy"] == "baseline_equal"].sort_values("target_date")
        wdf = weights[weights["strategy"] == "baseline_equal"]
        first_r = data.returns[25]
        expected_drift = .25*(1+first_r)/(1+first_r.mean())
        actual = wdf[wdf["target_date"] == data.dates[26]].set_index("factor").reindex(data.factors)["pretrade_weight"].to_numpy()
        np.testing.assert_allclose(actual, expected_drift, atol=1e-12)
        for row in equal.itertuples():
            np.testing.assert_allclose(row.net_return, (1-row.cost_fraction)*(1+row.gross_return)-1, atol=1e-12)

    @check("metrics_include_initial_wealth_drawdown")
    def _() -> None:
        stats = performance_metrics(np.array([-.10, 0., 0.]))
        np.testing.assert_allclose(stats["max_drawdown"], -.10, atol=1e-12)

    @check("export_manifest_tables_bootstrap_and_overwrite_guard")
    def _() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)/"research"
            result = run_experiment(data.prefix(32), cfg, models="mean,pooled_ridge", out=directory,
                                    rules=("timing_rank",), progress=False)
            assert (directory/"REPORT.md").is_file()
            assert (directory/"manifest.json").is_file()
            assert len(list(directory.glob("*.csv"))) == 13
            manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
            assert manifest["data_metadata"]["synthetic"]
            again = block_bootstrap_comparison(result.strategy_returns, cfg)
            pd.testing.assert_frame_equal(result.bootstrap_vs_equal, again)
            assert not result.expost_mean_weight_diagnostic["is_investable_oos_benchmark"].any()
            raises(FileExistsError, lambda: prepare_output_directory(directory, False))

    begin = time.perf_counter()
    passed = []
    with threadpool_limits(limits=1):
        for name, function in checks:
            try:
                function()
            except Exception:
                print(f"FAIL {name}", file=sys.stderr)
                raise
            passed.append(name)
            print(f"PASS {name}")
    result = {"passed": len(passed), "tests": passed, "elapsed_seconds": time.perf_counter()-begin}
    print(jdump({"self_test": "passed", "count": len(passed), "elapsed_seconds": result["elapsed_seconds"]}))
    return result


# ========================= SECTION 09: COMMAND LINE ============================
def write_template(out: str | Path, overwrite: bool = False) -> None:
    directory = prepare_output_directory(out, overwrite)
    dates = pd.date_range("2013-06-30", periods=3, freq=pd.offsets.MonthEnd())
    returns = pd.DataFrame({"date": dates, "VALUE": np.nan, "QUALITY": np.nan, "MOMENTUM": np.nan})
    records = [{"date": date, "factor": factor, "available_at": pd.NaT,
                "x_value_spread": np.nan, "x_profitability": np.nan,
                "x_fundamental_change": np.nan} for date in dates for factor in ["VALUE", "QUALITY", "MOMENTUM"]]
    returns.to_csv(directory/"returns_template.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    pd.DataFrame(records).to_csv(directory/"features_template.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    (directory/"config_example.json").write_text(json.dumps(jsonable(dataclasses.asdict(ResearchConfig())), ensure_ascii=False, indent=2), encoding="utf-8")
    (directory/"INPUT_README.md").write_text(
        "# 入力ひな形\n\n数値は意図的に空欄です。この3か月だけでは実行できません。\n"
        "returnsのdateはリターンが実現した月、featuresのdateは意思決定月末です。\n"
        "available_atにはその行の全特徴量が利用可能になった日時を入れます。\n"
        "単位は小数 (1%=0.01)。全月×全因子の特徴量行を作成してください。\n"
        "例のVALUE/QUALITY/MOMENTUMは自分の38因子名へ置換してください。\n"
        "最低期間・設定・PIT・執行時刻の注意事項は.py冒頭を参照してください。\n"
        "--config config_example.json で設定を読めます。\n", encoding="utf-8")


def config_from_args(args: argparse.Namespace) -> ResearchConfig:
    values: dict[str, Any] = {}
    if getattr(args, "config", None):
        values = json.loads(Path(args.config).read_text(encoding="utf-8-sig"))
        if not isinstance(values, dict):
            raise ValueError("config JSONはオブジェクトである必要があります。")
    allowed = {f.name for f in dataclasses.fields(ResearchConfig)}
    unknown = set(values)-allowed
    if unknown:
        raise ValueError(f"未知のconfig項目: {sorted(unknown)}")
    for name in allowed:
        if hasattr(args, name) and getattr(args, name) is not None:
            values[name] = getattr(args, name)
    if "cost_scenarios_bps" in values:
        values["cost_scenarios_bps"] = tuple(values["cost_scenarios_bps"])
    return ResearchConfig(**values)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="月次ファクター予測・ティルト比較。仕様は.py冒頭のdocstringを参照。")
    parser.add_argument("--verbose", action="store_true", help="詳細ログ")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="合成データで全処理を実行 (実データの証拠ではない)")
    demo.add_argument("--months", type=int, default=161)
    demo.add_argument("--factors", type=int, default=38)
    demo.add_argument("--features-count", type=int, default=6)
    run = sub.add_parser("run", help="ユーザーCSVを使って研究比較")
    run.add_argument("--returns", required=True)
    run.add_argument("--features", required=True)
    run.add_argument("--feature-cols", help="使用列のカンマ区切り。既定はx_列すべて")
    run.add_argument("--allow-unverified-pit", action="store_true", help="available_at不明を明示的に許可。結果に未確認と記録")
    for command in [demo, run]:
        command.add_argument("--out", required=True)
        command.add_argument("--overwrite", action="store_true", help="同名の既存出力を上書き (無関係なファイルは削除しない)")
        command.add_argument("--config", help="ResearchConfigのJSON")
        command.add_argument("--models", default="core", help="core/all またはモデル名のカンマ区切り")
        command.add_argument("--rules", default="timing_rank,timing_mv,level_mv", help="配分ルールのカンマ区切り")
        command.add_argument("--min-train", type=int, default=None)
        command.add_argument("--inner-min-train", type=int, default=None)
        command.add_argument("--inner-stride", type=int, default=None)
        command.add_argument("--tune-every", type=int, default=None)
        command.add_argument("--train-window", type=int, default=None)
        command.add_argument("--cost-bps", type=float, default=None)
        command.add_argument("--return-kind", choices=["factor_pnl", "sleeve_total"], default=None)
        command.add_argument("--bootstrap-samples", type=int, default=None)
        command.add_argument("--seed", type=int, default=None)
        command.add_argument("--threads", type=int, default=None)
    template = sub.add_parser("template", help="CSV/設定ファイルの空ひな形を書き出す")
    template.add_argument("--out", required=True)
    template.add_argument("--overwrite", action="store_true")
    sub.add_parser("self-test", help="内蔵テストを実行。ネット接続不要")
    sub.add_parser("list-models", help="モデル・候補パラメータ一覧")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")
    try:
        if args.command == "self-test":
            self_test()
            return 0
        if args.command == "list-models":
            for spec in model_registry().values():
                print(f"{spec.name}: {spec.description}\n  {jdump(spec.candidates)}")
            return 0
        if args.command == "template":
            write_template(args.out, args.overwrite)
            print(f"入力ひな形: {Path(args.out).resolve()}")
            return 0
        config = config_from_args(args)
        if args.command == "demo":
            data, raw_returns, raw_features = make_synthetic(args.months, args.factors, args.features_count, config.seed)
        else:
            data = load_csv(args.returns, args.features,
                            feature_cols=args.feature_cols.split(",") if args.feature_cols else None,
                            allow_unverified_pit=args.allow_unverified_pit)
        rules = tuple(s.strip() for s in args.rules.split(",") if s.strip())
        result = run_experiment(data, config, args.models, out=args.out, overwrite=args.overwrite, rules=rules)
        if args.command == "demo":
            demo_dir = Path(args.out)/"demo_inputs"
            demo_dir.mkdir(exist_ok=True)
            raw_returns.to_csv(demo_dir/"returns.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
            raw_features.to_csv(demo_dir/"features.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        summary = {"output_directory": str(Path(args.out).resolve()),
                   "synthetic": data.metadata.get("synthetic", False),
                   "models": len(result.manifest["models"]), "strategies": len(result.strategy_summary),
                   "oos_months": result.manifest["oos_months"],
                   "elapsed_seconds": result.manifest["elapsed_seconds"]}
        print(json.dumps(jsonable(summary), ensure_ascii=False, indent=2))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, FloatingPointError) as exc:
        if args.verbose:
            LOG.exception("実行を停止しました")
        else:
            LOG.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
