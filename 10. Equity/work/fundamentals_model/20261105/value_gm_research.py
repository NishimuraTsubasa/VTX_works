#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Value / Growth Momentum: one-file research sample, v1.0.0.

日本語の入力仕様・数式は同梱 INSTRUCTIONS.md / SAMPLE_README.md を参照。
このファイルだけで demo / run / templates / reallocate / self-test を実行できます。
元の factor_tilt_research.py のimport・変更は不要です。ネット接続・発注は行いません。

pip install numpy pandas scipy
python value_gm_research.py self-test
python value_gm_research.py demo --out demo_results
python value_gm_research.py templates --out my_inputs
python value_gm_research.py run --config my_inputs/runtime_config.json --out results
python value_gm_research.py reallocate --config my_inputs/runtime_config.json \
    --spread-predictions results/spread_predictions.csv --out realloc_results

7配分系列: B0固定、B1過去平均、A1個別予測集約、A2=50%縮小、
C1スプレッド直接Ridge、C2グループ直接Ridgeの差、D1=3ブロック予測平均。
実データのA1には既存 predictions.csv を使うのが基本です。
demoのA1は固定lambdaのsample local Ridgeで、ユーザーの現行モデルの再現ではありません。
各モデルを使えない場合はnot_runと理由を記録し、勝手な代替で成功扱いにしません。

学習ペア: (x_s, r_(s+1)); decision tではラベルr_tまで。全前処理を毎回再fit。
固定グループ、月次1期先、同じ連続OOS、原点ごと共通の平均・標本標準偏差。
未来情報チェックは入力日時だけで、改訂履歴や外部予測の選択履歴を保証しません。
閲覧済みOOSは未使用テストに戻りません。合成データの成績は実運用の証拠ではありません。

SECTION単位で分割可能:
00 config, 01 inputs/groups, 02 transforms/models, 03 forecasts,
04 allocation/cost, 05 evaluation, 06 reporting, 07 demo/tests, 08 CLI.
P3のTVP/残差補正/選択器/推定校正は意図的に未実装です。
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import tempfile
import warnings
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
except ImportError as exc:
    raise SystemExit('pip install numpy pandas scipy を実行してください。\n' + str(exc)) from exc

VERSION = '1.0.0'
GROUPS = ('VALUE', 'GROWTH_MOMENTUM')
BLOCKS = ('VALUATION', 'FUNDAMENTAL', 'TREND')
PRED_MODELS = ('B1', 'A1', 'A2', 'C1', 'C2', 'D1')
PRIMARY = ('A1', 'A2', 'C1', 'C2', 'D1')
DESCRIPTIONS = {'B0': '固定グループ配分', 'B1': '過去平均スプレッド',
                'A1': '既存／明示的sample予測の集約', 'A2': 'A1を過去平均へ50%縮小',
                'C1': 'スプレッド直接Ridge', 'C2': 'グループ直接Ridgeの差',
                'D1': '3ブロックの等ウェイト予測平均'}

# ====================== 00 CONFIG / CONTRACTS ======================
@dataclass(frozen=True)
class Config:
    min_train: int = 60
    train_window: int | None = None
    trend_months: int = 12
    ridge_lambda: float = 1.0
    winsor_quantile: float = 0.01
    minimum_feature_coverage: float = 0.20
    spread_clip_sigma: float = 3.0
    base_value_weight: float = 0.50
    max_tilt: float = 0.20
    risk_aversion: float = 20.0
    anchor_penalty: float = 0.01
    variance_window: int = 60
    variance_floor: float = 1e-8
    max_active_vol_annual: float = 0.05
    turnover_penalty_bps: float = 10.0
    cost_bps: float = 10.0
    cost_scenarios_bps: tuple[float, ...] = (0.0, 10.0, 25.0)
    return_kind: str = 'factor_pnl'
    bootstrap_samples: int = 5000
    bootstrap_blocks: tuple[int, ...] = (3, 6, 12)
    hac_lags: tuple[int, ...] = (3, 6)
    seed: int = 42
    core_mode: str = 'saved'  # saved / sample_local_ridge / none
    core_source_model: str | None = None
    allow_unverified_pit: bool = False
    core_scope: str = 'unresolved'
    core_factor_ids: tuple[str, ...] = ()
    feature_orientation_confirmed: bool = False
    feature_columns: dict[str, list[str]] = field(default_factory=lambda: {
        'VALUATION': ['x_value_spread'], 'FUNDAMENTAL': ['x_profitability_change']})
    oos_start: str | None = None  # decision_date, optional, never selected from P&L
    seen_oos_end: str | None = None

    def validate(self, *, real: bool = False) -> None:
        numeric = ['ridge_lambda', 'spread_clip_sigma', 'risk_aversion', 'anchor_penalty',
                   'variance_floor', 'max_active_vol_annual']
        if any(not np.isfinite(getattr(self, x)) or getattr(self, x) <= 0 for x in numeric):
            raise ValueError('正かつ有限のlambda・リスク設定が必要です。')
        for x in ['min_train', 'trend_months', 'variance_window', 'bootstrap_samples', 'seed']:
            if isinstance(getattr(self, x), bool) or not isinstance(getattr(self, x), int):
                raise ValueError(f'{x} must be integer')
        if self.min_train < 12 or self.trend_months < 1 or self.variance_window < 3:
            raise ValueError('min_train>=12, trend_months>=1, variance_window>=3')
        if self.train_window is not None and (not isinstance(self.train_window, int) or self.train_window < self.min_train):
            raise ValueError('train_window must be integer >= min_train')
        if not (0 <= self.winsor_quantile < .25 and 0 < self.minimum_feature_coverage <= 1):
            raise ValueError('前処理設定の範囲が不正です。')
        if not 0 < self.base_value_weight < 1 or not 0 < self.max_tilt <= .5:
            raise ValueError('基準配分またはティルト上限が不正です。')
        costs = (self.cost_bps, self.turnover_penalty_bps, *self.cost_scenarios_bps)
        if any(not np.isfinite(v) or not 0 <= v < 1000 for v in costs):
            raise ValueError('cost must be bps in [0,1000)')
        if self.return_kind not in ('factor_pnl', 'sleeve_total'):
            raise ValueError('return_kind must be factor_pnl or sleeve_total')
        if self.core_mode not in ('saved', 'sample_local_ridge', 'none'):
            raise ValueError('unknown core_mode')
        if self.core_mode == 'saved' and not self.core_source_model:
            raise ValueError('savedモードはcore_source_modelを明示してください。')
        if self.bootstrap_samples < 0 or not self.bootstrap_blocks or not self.hac_lags:
            raise ValueError('bootstrap / HAC settings invalid')
        if any(not isinstance(v, int) or v < 1 for v in self.bootstrap_blocks):
            raise ValueError('bootstrap block must be positive integer')
        if any(not isinstance(v, int) or v < 0 for v in self.hac_lags):
            raise ValueError('HAC lag must be nonnegative integer')
        if set(self.feature_columns) - {'VALUATION', 'FUNDAMENTAL'}:
            raise ValueError('TRENDはコードで生成。feature_columnsのキーはVALUATION/FUNDAMENTALのみ。')
        cols = []
        for block, names in self.feature_columns.items():
            if not isinstance(names, list) or len(names) > 2 or any(not isinstance(c, str) or not c.startswith('x_') for c in names):
                raise ValueError(f'{block}: x_列を最大2つ指定してください。')
            cols.extend(names)
        if len(cols) != len(set(cols)):
            raise ValueError('複数ブロックで同じ特徴列を重複指定できません。')
        if len(self.core_factor_ids) != len(set(self.core_factor_ids)):
            raise ValueError('core_factor_ids duplicated')
        if real and self.core_scope not in ('prediction_targets', 'information_sources', 'tradable_constituents', 'mixed'):
            raise ValueError('core_scope未確定。コード等で13本の役割を確認してください。')
        if real and not self.core_factor_ids:
            raise ValueError('実データには実際のcore_factor_idsを明示してください。')

@dataclass
class GroupData:
    dates: pd.DatetimeIndex
    factors: tuple[str, ...]
    returns: np.ndarray  # oriented T x K, membership factors only
    A: np.ndarray        # K x 2, each column sums to one
    orientation: np.ndarray
    group_returns: np.ndarray  # T x 2
    spread: np.ndarray
    factor_features: np.ndarray | None  # T x K x P, already economically aligned
    feature_names: tuple[str, ...]
    group_features: np.ndarray | None   # T x 2 x P, missing -> NA, no reweighting
    metadata: dict[str, Any]

@dataclass
class Result:
    tables: dict[str, pd.DataFrame]
    manifest: dict[str, Any]


def clean_json(x: Any) -> Any:
    if dataclasses.is_dataclass(x):
        return clean_json(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {str(k): clean_json(v) for k, v in x.items()}
    if isinstance(x, (tuple, list, np.ndarray)):
        return [clean_json(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    if isinstance(x, (pd.Timestamp, datetime, Path)):
        return str(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def dump(x: Any) -> str:
    return json.dumps(clean_json(x), ensure_ascii=False, sort_keys=True, allow_nan=False)


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frame_sha(frame: pd.DataFrame) -> str:
    return hashlib.sha256(frame.to_csv(index=False, lineterminator='\n').encode()).hexdigest()


def month_dates(values: Any, label: str) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(pd.to_datetime(values, errors='raise'))
    if idx.tz is not None or idx.hasnans or not np.all(idx == idx.normalize()) or not np.all(idx.is_month_end):
        raise ValueError(f'{label}: タイムゾーンなし・時刻なしの月末日が必要です。')
    return idx

# ====================== 01 INPUTS / GROUPS ======================
def build_group_panel(returns: pd.DataFrame, membership: pd.DataFrame,
                      features: pd.DataFrame | None, cfg: Config) -> GroupData:
    """No input mutation. Target membership is fixed; unused return columns are not traded."""
    cfg.validate()
    rf, mem = returns.copy(), membership.copy()
    if 'date' not in rf or not {'group','factor_id','weight','return_orientation'}.issubset(mem):
        raise ValueError('returnsのdate / membershipのgroup,factor_id,weight,return_orientationが必要です。')
    rf['date'] = month_dates(rf['date'], 'returns.date')
    if rf['date'].duplicated().any() or rf.columns.duplicated().any():
        raise ValueError('returnsに重複があります。')
    rf = rf.sort_values('date').set_index('date')
    if len(rf) < 3 or np.any(np.diff(rf.index.to_period('M').asi8) != 1):
        raise ValueError('月抜け／月数不足。月を詰めて実行しません。')
    if mem.empty or mem[['group','factor_id']].isna().any().any():
        raise ValueError('グループ定義が空または欠損です。')
    mem['factor_id'] = mem['factor_id'].astype(str)
    if set(mem['group']) != set(GROUPS) or mem['factor_id'].duplicated().any():
        raise ValueError('VALUEとGROWTH_MOMENTUMの非重複・固定構成が必要です。')
    factors = tuple(sorted(mem['factor_id']))
    if not set(factors).issubset(rf.columns):
        raise ValueError('membershipにあるfactorのリターンがありません。')
    m = mem.set_index('factor_id').reindex(factors)
    w = pd.to_numeric(m['weight'], errors='raise').to_numpy(float)
    ori = pd.to_numeric(m['return_orientation'], errors='raise').to_numpy(float)
    if not np.isfinite(w).all() or np.any(w <= 0) or not np.isin(ori, [-1.,1.]).all():
        raise ValueError('weight>0、return_orientationは+1/-1が必要です。')
    if cfg.return_kind == 'sleeve_total' and np.any(ori != 1):
        raise ValueError('total-return商品の単純符号反転は禁止です。')
    A = np.column_stack([np.where(m['group'].to_numpy() == g, w, 0.) for g in GROUPS])
    if not np.allclose(A.sum(axis=0), 1, atol=1e-10, rtol=0):
        raise ValueError('各グループweight合計は1。自動再正規化はしません。')
    r = rf[list(factors)].apply(pd.to_numeric, errors='raise').to_numpy(float)
    if not np.isfinite(r).all():
        raise ValueError('構成因子のリターン欠損／infは禁止です。')
    r = r * ori
    if cfg.return_kind == 'sleeve_total' and np.any(r <= -1):
        raise ValueError('sleeve_totalはreturn>-1が必要です。')
    requested = tuple(c for b in ('VALUATION','FUNDAMENTAL') for c in cfg.feature_columns.get(b, []))
    names: tuple[str,...] = ()
    x = gx = None
    meta: dict[str,Any] = {'membership_sha256': frame_sha(mem), 'returns_frame_sha256': frame_sha(returns),
        'input_return_unit':'decimal', 'target_fixed':True, 'membership_version':
        '|'.join(sorted(map(str,mem['membership_version'].unique()))) if 'membership_version' in mem else 'unspecified',
        'timestamp_check_passed':False, 'pit_independently_verified':False,
        'selection_history_verified':False, 'unused_return_factors': sorted(set(rf.columns)-set(factors)),
        'warnings':[], 'unavailable_feature_columns':list(requested)}
    if np.max(np.abs(r)) > .75:
        meta['warnings'].append('75%を超える月次リターン。百分率と小数の混同を確認。')
    if features is not None:
        if not cfg.feature_orientation_confirmed:
            raise ValueError('feature_orientation_confirmed=trueを、特性の定義・向き確認後に指定してください。')
        ff = features.copy()
        if not {'date','factor'}.issubset(ff):
            raise ValueError('featuresはdate,factorが必要です。')
        ff['date'] = month_dates(ff['date'], 'features.date')
        ff['factor'] = ff['factor'].astype(str)
        ff = ff.loc[ff['factor'].isin(factors)].copy()
        if ff.duplicated(['date','factor']).any():
            raise ValueError('featuresの月×factor重複。')
        expected = pd.MultiIndex.from_product([rf.index,factors],names=['date','factor'])
        fx = ff.set_index(['date','factor'])
        if len(fx) != len(expected) or not expected.isin(fx.index).all():
            raise ValueError('featuresは全月×構成因子の行が必要。値欠損はNaNで保持。')
        fx = fx.reindex(expected)
        cutoff = expected.get_level_values('date') + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
        if 'available_at' in fx:
            avail = pd.to_datetime(fx['available_at'], errors='raise')
            if getattr(avail.dt,'tz',None) is not None:
                raise ValueError('available_atはdateと同じタイムゾーン基準のnaive日時に。')
            if (avail.to_numpy() > cutoff.to_numpy()).any():
                raise ValueError('未来のavailable_atがあります。')
            valid_stamp = not avail.isna().any()
        else:
            valid_stamp = False
        if not valid_stamp and not cfg.allow_unverified_pit:
            raise ValueError('available_at欠損。明示許可がなければ停止します。')
        meta['timestamp_check_passed'] = valid_stamp
        if not valid_stamp:
            meta['warnings'].append('PIT利用可能日時が未確認。')
        names = tuple(c for c in requested if c in fx.columns)
        meta['unavailable_feature_columns'] = sorted(set(requested)-set(names))
        meta['features_frame_sha256'] = frame_sha(features)
        if names:
            raw = fx[list(names)].apply(pd.to_numeric, errors='raise').to_numpy(float)
            if np.isinf(raw).any():
                raise ValueError('特徴量infは禁止。定義不能値はNaNに。')
            x = raw.reshape(len(rf),len(factors),len(names))
            # Only group members; 0 * NaN from unrelated factors must not poison aggregation.
            gx = np.stack([np.sum(x[:,A[:,j]>0,:] * A[A[:,j]>0,j][None,:,None],axis=1) for j in range(2)],axis=1)
            meta['raw_feature_missing_fraction'] = float(np.isnan(x).mean())
    gr = r @ A
    return GroupData(rf.index, factors, r, A, ori, gr, gr[:,0]-gr[:,1], x, names, gx, meta)


def build_spread_features(data: GroupData, cfg: Config) -> tuple[np.ndarray,np.ndarray,dict[str,list[int]],list[str]]:
    """Raw group characteristics FIRST, train-only preprocessing later. Return sums include month t."""
    arrays, names, blocks = [], [], {}
    if data.group_features is not None:
        arrays.append(data.group_features)
        names.extend(data.feature_names)
    for block in ('VALUATION','FUNDAMENTAL'):
        blocks[block] = [names.index(c) for c in cfg.feature_columns.get(block,[]) if c in names]
    trend = pd.DataFrame(data.group_returns).rolling(cfg.trend_months,min_periods=cfg.trend_months).sum().to_numpy()
    arrays.append(trend[:,:,None])
    blocks['TREND'] = [len(names)]
    names.append(f'return_sum_{cfg.trend_months}')
    gf = np.concatenate(arrays,axis=2)
    return gf[:,0]-gf[:,1], gf, blocks, names


def prepare_saved_predictions(frame: pd.DataFrame | None, data: GroupData,
                              cfg: Config) -> tuple[pd.DataFrame | None,str]:
    if cfg.core_mode != 'saved':
        return None, cfg.core_mode
    if frame is None:
        return None, 'not_run_missing_saved_predictions'
    required = {'decision_date','target_date','model','factor','prediction'}
    if not required.issubset(frame):
        raise ValueError(f'保存済み予測には{sorted(required)}が必要。')
    df = frame.loc[frame['model'].astype(str)==str(cfg.core_source_model)].copy()
    if df.empty:
        raise ValueError('指定したcore_source_modelが見つかりません。')
    df['factor'] = df['factor'].astype(str)
    df = df.loc[df['factor'].isin(data.factors)].copy()
    df['decision_date'] = month_dates(df['decision_date'],'prediction.decision_date')
    df['target_date'] = month_dates(df['target_date'],'prediction.target_date')
    if df.duplicated(['decision_date','target_date','factor']).any():
        raise ValueError('予測行が重複しています。')
    if not ((df['decision_date']+pd.offsets.MonthEnd(1))==df['target_date']).all():
        raise ValueError('予測のdecision/targetは1か月差が必要です。')
    if 'train_end' in df:
        train_end = pd.to_datetime(df['train_end'],errors='raise')
        if train_end.isna().any() or (train_end > df['decision_date']).any():
            raise ValueError('保存済み予測のtrain_endが不正／未来です。')
    df['prediction'] = pd.to_numeric(df['prediction'],errors='raise')
    if not np.isfinite(df['prediction']).all():
        raise ValueError('予測NaN/infを代替しません。')
    return df, 'saved_prediction_provenance_not_independently_verified'

# ====================== 02 TRANSFORMS / MODELS ======================
class TrainScaler:
    def __init__(self, q: float=.01, coverage: float=.20):
        self.q, self.coverage = q, coverage

    def fit(self, x: np.ndarray) -> 'TrainScaler':
        x = np.asarray(x,float)
        if x.ndim != 2 or len(x) < 2 or x.shape[1] < 1 or np.isinf(x).any():
            raise ValueError('scaler expects 2D finite-or-NaN training features')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',RuntimeWarning)
            self.median = np.nanmedian(x,axis=0)
            self.low = np.nanquantile(x,self.q,axis=0)
            self.high = np.nanquantile(x,1-self.q,axis=0)
        self.median = np.nan_to_num(self.median,nan=0.)
        self.low = np.where(np.isfinite(self.low),self.low,self.median)
        self.high = np.where(np.isfinite(self.high),self.high,self.median)
        filled = np.clip(np.where(np.isnan(x),self.median,x),self.low,self.high)
        self.mean, sd = filled.mean(axis=0), filled.std(axis=0,ddof=0)
        self.active = (np.isfinite(x).mean(axis=0)>=self.coverage) & (sd>1e-10)
        self.scale = np.where(self.active,sd,1.)
        return self

    def transform(self,x: np.ndarray) -> np.ndarray:
        x = np.asarray(x,float)
        if np.isinf(x).any():
            raise ValueError('inf feature')
        return np.where(self.active,(np.clip(np.where(np.isnan(x),self.median,x),self.low,self.high)-self.mean)/self.scale,0.)


def ridge_predict(x: np.ndarray,y: np.ndarray,new: np.ndarray,lam: float) -> np.ndarray:
    """mean squared error + lam * ||beta||^2; intercept unpenalized."""
    x,y,new = np.asarray(x,float),np.asarray(y,float),np.asarray(new,float)
    if lam <= 0 or not all(np.isfinite(v).all() for v in (x,y,new)):
        raise ValueError('invalid Ridge input')
    xm,ym = x.mean(axis=0),y.mean(axis=0)
    xc,yc = x-xm,y-ym
    beta = np.linalg.solve(xc.T@xc/len(x)+lam*np.eye(x.shape[1]),xc.T@yc/len(x))
    return (new-xm)@beta+ym


def forecast_spread_once(x: np.ndarray,y: np.ndarray,new: np.ndarray,cfg: Config) -> tuple[float,dict[str,Any]]:
    scaler = TrainScaler(cfg.winsor_quantile,cfg.minimum_feature_coverage).fit(x)
    z,zn = scaler.transform(x),scaler.transform(np.asarray(new).reshape(1,-1))
    ym,ys = float(np.mean(y)),float(np.std(y,ddof=0))
    if not np.isfinite(y).all():
        raise ValueError('target missing')
    info = {'inactive_columns':int((~scaler.active).sum()), 'feature_count':x.shape[1],
            'missing_fraction':float(np.isnan(x).mean()), 'lambda':cfg.ridge_lambda,
            'current_missing_fraction':float(np.isnan(new).mean()),
            'current_outside_training_quantiles':int(np.sum((new<scaler.low)|(new>scaler.high))),
            'status':'ok'}
    if not scaler.active.any() or ys < 1e-12:
        info['status'] = 'no_active_features' if not scaler.active.any() else 'constant_target'
        return ym,info
    pred = ridge_predict(z,(y-ym)/ys,zn,cfg.ridge_lambda)[0]
    return float(ym+ys*pred),info


def shrink_prediction(prediction: float,mean: float,gamma: float) -> float:
    if not 0 <= gamma <= 1:
        raise ValueError('gamma must be [0,1]')
    return float(mean+gamma*(prediction-mean))


def aggregate_core_predictions(values: np.ndarray,data: GroupData) -> float:
    if values.shape != (len(data.factors),) or not np.isfinite(values).all():
        raise ValueError('構成因子の予測が欠けています。')
    group = (values*data.orientation) @ data.A
    return float(group[0]-group[1])

# ====================== 03 WALK FORWARD / FORECASTS ======================
def walk_forward_spread(data: GroupData,cfg: Config,saved: pd.DataFrame | None=None) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """No hyperparameter search in this sample. External A1 uses only supplied historical forecasts."""
    x,gx,blocks,names = build_spread_features(data,cfg)
    supplied,source_status = prepare_saved_predictions(saved,data,cfg)
    direct_available = data.group_features is not None
    a_available = (supplied is not None or (cfg.core_mode=='sample_local_ridge' and data.factor_features is not None))
    d_available = direct_available and all(blocks[b] for b in BLOCKS)
    enabled = ['B1'] + (['A1','A2'] if a_available else []) + (['C1','C2'] if direct_available else []) + (['D1'] if d_available else [])
    registry = []
    for model in ['B0',*PRED_MODELS]:
        status = 'executed' if model=='B0' or model in enabled else 'not_run'
        reason = ''
        if status=='not_run':
            reason = 'missing_feature_block' if model=='D1' else ('missing_saved_predictions_or_explicit_sample_core' if model in ('A1','A2') else 'missing_features')
        registry.append({'model':model,'description':DESCRIPTIONS[model],'status':status,'reason':reason})
    # Common warmup excludes the same initial pair dates for every model, including B1/A1.
    warmup = cfg.trend_months-1 if direct_available else 0
    first = warmup+cfg.min_train
    if supplied is not None:
        earliest = supplied['decision_date'].min()
        first = max(first,int(data.dates.searchsorted(earliest)))
    if cfg.oos_start is not None:
        ds = month_dates([cfg.oos_start],'oos_start')[0]
        first = max(first,int(data.dates.searchsorted(ds)))
    if first >= len(data.dates)-1:
        raise ValueError('ウォームアップと共通60学習ペアの後にOOSがありません。')
    if supplied is not None:
        required = pd.MultiIndex.from_product([data.dates[first:-1],data.factors],names=['decision_date','factor'])
        selected = supplied.set_index(['decision_date','factor'])
        if not required.isin(selected.index).all():
            missing = required[~required.isin(selected.index)]
            raise ValueError('共通OOSの保存予測に欠落。ウェイトを変更しません: '+str(list(missing[:3])))
    else:
        selected = None
    rows,components,diagnostics = [],[],[]
    for t in range(first,len(data.dates)-1):
        start = warmup if cfg.train_window is None else max(warmup,t-cfg.train_window)
        train_x,train_y = x[start:t],data.spread[start+1:t+1]
        hist,sd = float(train_y.mean()),float(train_y.std(ddof=1))
        if not np.isfinite(sd):
            raise ValueError('学習スプレッドの分散が非有限。')
        raw = {'B1':hist}
        origin_info: dict[str,Any] = {'B1':{'status':'historical_mean'}}
        if 'A1' in enabled:
            if selected is not None:
                part = selected.loc[(data.dates[t],slice(None)),:].droplevel(0).reindex(data.factors)
                vals = part['prediction'].to_numpy(float)
                raw['A1'] = aggregate_core_predictions(vals,data)
                origin_info['A1'] = {'status':source_status,'upstream_preprocessing':'unknown_or_already_applied'}
            else:
                vals = []
                for k in range(len(data.factors)):
                    value,info = forecast_spread_once(data.factor_features[start:t,k,:],data.returns[start+1:t+1,k],data.factor_features[t,k,:],cfg)
                    vals.append(value)
                # sample targets are already oriented, unlike saved raw-factor predictions.
                group_pred = np.asarray(vals) @ data.A
                raw['A1'] = float(group_pred[0]-group_pred[1])
                origin_info['A1'] = {'status':'sample_local_ridge_NOT_user_model','lambda':cfg.ridge_lambda}
        if 'C1' in enabled:
            raw['C1'],origin_info['C1'] = forecast_spread_once(train_x,train_y,x[t],cfg)
            gp,gi = [],[]
            for j in range(2):
                p,di = forecast_spread_once(gx[start:t,j,:],data.group_returns[start+1:t+1,j],gx[t,j,:],cfg)
                gp.append(p);gi.append(di)
            raw['C2'] = gp[0]-gp[1]
            origin_info['C2'] = {'group_predictions':gp,'groups':gi}
        if 'D1' in enabled:
            block_preds=[]
            for block in BLOCKS:
                idx = blocks[block]
                p,di = forecast_spread_once(train_x[:,idx],train_y,x[t,idx],cfg)
                block_preds.append(p)
                components.append({'decision_date':data.dates[t],'target_date':data.dates[t+1],
                                   'block':block,'raw_prediction':p,'realized':data.spread[t+1],
                                   'train_end':data.dates[t], 'input_columns':dump([names[i] for i in idx]),'diagnostics':dump(di)})
            raw['D1'] = float(np.mean(block_preds))
            origin_info['D1'] = {'equal_weight_components':block_preds}
        bound = cfg.spread_clip_sigma*sd
        processed = {m:float(np.clip(v,hist-bound,hist+bound)) for m,v in raw.items()}
        if 'A2' in enabled:
            processed['A2'] = shrink_prediction(processed['A1'],hist,.5)
            raw['A2'] = processed['A2']  # shrink AFTER A1 common clip; not a second hidden clip
            origin_info['A2'] = {'status':'shrink_once_after_A1_clip','gamma':.5}
        for model in enabled:
            source_model = str(cfg.core_source_model) if selected is not None else cfg.core_mode
            row={'decision_date':data.dates[t],'target_date':data.dates[t+1],'model':model,
                 'raw_prediction':raw[model],'prediction':processed[model],'historical_mean':hist,
                 'timing_prediction':processed[model]-hist,'train_sigma':sd,'shrink_gamma':.5 if model=='A2' else (0. if model=='B1' else 1.),
                 'realized':data.spread[t+1], 'source_model':source_model if model in ('A1','A2') else model,
                 'train_start':data.dates[start+1],'train_end':data.dates[t], 'n_train':len(train_y),
                 'feature_set_id':hashlib.sha256(dump(names).encode()).hexdigest()[:16],
                 'input_cutoff':data.dates[t], 'membership_version':data.metadata['membership_version'],
                 'prediction_source_verified':False, 'upstream_train_window_known': model not in ('A1','A2') or selected is None,
                 'common_clip_applied':abs(raw[model]-processed[model])>1e-12}
            rows.append(row)
            diagnostics.append({'decision_date':data.dates[t],'model':model,'details':dump(origin_info[model])})
    return pd.DataFrame(rows),pd.DataFrame(components,columns=['decision_date','target_date','block','raw_prediction','realized','train_end','input_columns','diagnostics']),pd.DataFrame(diagnostics),pd.DataFrame(registry)

# ====================== 04 ALLOCATION / COST ======================
def allocate_two_group_tilt(prediction: float,variance: float,A: np.ndarray,pretrade: np.ndarray,cfg: Config) -> tuple[np.ndarray,dict[str,float]]:
    """Exact scalar convex piecewise-quadratic minimization; no numerical optimizer fallback.
    min -mu*b + (risk_aversion*v/2+anchor_penalty)*b^2 + tau*||A*q(b)-pre||_1.
    Check every absolute-value kink, interval stationary point, and feasible endpoint.
    """
    if not np.isfinite(prediction) or not np.isfinite(variance) or variance<0:
        raise ValueError('allocation nonfinite/negative input')
    v=max(variance,cfg.variance_floor)
    q0=np.array([cfg.base_value_weight,1-cfg.base_value_weight])
    c,d=A@q0,A[:,0]-A[:,1]
    cap=min(cfg.max_tilt,cfg.max_active_vol_annual/math.sqrt(12*v))
    lo,hi=max(-cap,-q0[0]),min(cap,q0[1])
    nonzero=np.abs(d)>1e-14
    kinks=(pretrade[nonzero]-c[nonzero])/d[nonzero]
    points=sorted(set([lo,hi,*[float(k) for k in kinks if lo<k<hi]]))
    candidates=points.copy()
    curvature=cfg.risk_aversion*v+2*cfg.anchor_penalty
    rate=cfg.turnover_penalty_bps/10000
    for a,b in zip(points[:-1],points[1:]):
        mid=(a+b)/2
        slope=float(d @ np.sign(c+d*mid-pretrade))
        stationary=(prediction-rate*slope)/curvature
        candidates.append(float(np.clip(stationary,a,b)))
    def objective(b: float) -> float:
        return float(-prediction*b+.5*curvature*b*b+rate*np.abs(c+d*b-pretrade).sum())
    b=min(candidates,key=lambda u:(objective(u),abs(u)))
    q=q0+np.array([b,-b])
    if abs(q.sum()-1)>1e-10 or np.min(q)<-1e-10 or abs(b)>cfg.max_tilt+1e-10:
        raise RuntimeError('allocation constraint violation')
    return q,{'tilt':b,'variance_used':v,'variance_floored':variance<cfg.variance_floor,
              'predicted_active_vol_annual':abs(b)*math.sqrt(12*v),'objective':objective(b)}


def funded_rebalance_cost(target: np.ndarray,pretrade: np.ndarray,rate: float) -> tuple[float,float]:
    if not 0<=rate<.1:
        raise ValueError('rate out of range')
    if rate==0:
        return 0.,float(np.abs(target-pretrade).sum())
    lo,hi=0.,1.
    for _ in range(65):
        c=(lo+hi)/2
        if c-rate*np.abs((1-c)*target-pretrade).sum()>0:
            hi=c
        else:
            lo=c
    cost=(lo+hi)/2
    return cost,float(np.abs((1-cost)*target-pretrade).sum())


def cost_and_net(w: np.ndarray,pre: np.ndarray,r: np.ndarray,bps: float,kind: str) -> tuple[float,float,float,float]:
    gross=float(w@r)
    if kind=='factor_pnl':
        traded=float(np.abs(w-pre).sum());cost=bps/10000*traded;net=gross-cost
    elif kind=='sleeve_total':
        cost,traded=funded_rebalance_cost(w,pre,bps/10000);net=(1-cost)*(1+gross)-1
    else:
        raise ValueError('unknown return kind')
    if net<=-1 or not np.isfinite(net):
        raise ValueError('NAV<=0 or nonfinite; no silent leverage repair')
    return gross,net,cost,traded


def validate_spread_predictions(pred: pd.DataFrame,data: GroupData) -> pd.DataFrame:
    p=pred.copy()
    required={'decision_date','target_date','model','prediction','historical_mean','realized'}
    if not required.issubset(p):
        raise ValueError('spread prediction columns missing')
    for c in ['decision_date','target_date']:
        p[c]=month_dates(p[c],c)
    if p.duplicated(['target_date','model']).any() or not ((p.decision_date+pd.offsets.MonthEnd(1))==p.target_date).all():
        raise ValueError('spread predictions duplicated or not one-step aligned')
    if not np.isfinite(p[['prediction','historical_mean','realized']].to_numpy(float)).all():
        raise ValueError('spread predictions nonfinite')
    models=list(p.model.unique())
    if 'B1' not in models or 'B0' in models or set(models)-set(PRED_MODELS):
        raise ValueError('B1 required and only registered prediction models allowed')
    dates=sorted(p.target_date.unique())
    if np.any(np.diff(pd.DatetimeIndex(dates).to_period('M').asi8)!=1):
        raise ValueError('OOS dates not consecutive')
    for model,part in p.groupby('model'):
        if list(part.sort_values('target_date').target_date)!=dates:
            raise ValueError('all models must share contiguous OOS')
    pos=data.dates.get_indexer(p.target_date)
    if np.any(pos<1) or not np.allclose(p.realized,data.spread[pos],atol=1e-10,rtol=1e-8):
        raise ValueError('prediction.realized differs from fixed target returns')
    if 'train_end' in p:
        td=pd.to_datetime(p.train_end,errors='raise')
        if td.isna().any() or (td>p.decision_date).any():
            raise ValueError('future training end')
    if p.groupby('target_date').historical_mean.agg(lambda s:s.max()-s.min()).max()>1e-10:
        raise ValueError('common historical mean differs by model')
    b1=p.loc[p.model=='B1']
    if not np.allclose(b1.prediction,b1.historical_mean,atol=1e-12):
        raise ValueError('B1 must predict common past mean')
    return p.sort_values(['target_date','model']).reset_index(drop=True)


def simulate_portfolios(data: GroupData,predictions: pd.DataFrame,cfg: Config) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    pred=validate_spread_predictions(predictions,data)
    models=['B0',*[m for m in PRED_MODELS if m in set(pred.model)]]
    pre={m:np.zeros(len(data.factors)) for m in models}
    records,weights,scenarios=[],[],[]
    q0=np.array([cfg.base_value_weight,1-cfg.base_value_weight])
    for date,part in pred.groupby('target_date',sort=True):
        t=int(data.dates.get_loc(date))-1
        hist=data.spread[max(0,t+1-cfg.variance_window):t+1]
        if len(hist)<3:
            raise ValueError('variance history too short')
        variance=float(np.var(hist,ddof=1))
        preds=part.set_index('model').prediction.to_dict()
        r=data.returns[t+1]
        for m in models:
            old=pre[m].copy()
            if m=='B0':
                q=q0.copy();info={'tilt':0.,'predicted_active_vol_annual':0.,'variance_used':max(variance,cfg.variance_floor),'variance_floored':variance<cfg.variance_floor}
            else:
                q,info=allocate_two_group_tilt(float(preds[m]),variance,data.A,old,cfg)
            w=data.A@q
            gross,net,cost,traded=cost_and_net(w,old,r,cfg.cost_bps,cfg.return_kind)
            records.append({'decision_date':data.dates[t],'target_date':date,'strategy':m,'model':m,
                'gross_return':gross,'net_return':net,'cost_fraction':cost,'traded_notional_fraction':traded,
                'half_l1_turnover':.5*traded,'value_weight':q[0],'growth_momentum_weight':q[1],
                **info,'return_kind':cfg.return_kind})
            for j,k in enumerate(data.factors):
                weights.append({'decision_date':data.dates[t],'target_date':date,'strategy':m,'factor':k,
                    'weight':w[j],'pretrade_weight':old[j], 'factor_return':r[j],
                    'gross_contribution':w[j]*r[j]})
            for bps in sorted(set([cfg.cost_bps,*cfg.cost_scenarios_bps])):
                _,alt,ac,at=cost_and_net(w,old,r,bps,cfg.return_kind)
                scenarios.append({'target_date':date,'strategy':m,'cost_bps':bps,'net_return':alt,
                                  'cost_fraction':ac,'positions_reoptimized':False})
            pre[m]=w*(1+r)/(1+gross) if cfg.return_kind=='sleeve_total' else w.copy()
    return pd.DataFrame(records),pd.DataFrame(weights),pd.DataFrame(scenarios)

# ====================== 05 EVALUATION / INFERENCE ======================
def safe_corr(a: np.ndarray,b: np.ndarray) -> float:
    if len(a)<3 or np.std(a)<1e-12 or np.std(b)<1e-12:
        return math.nan
    return float(np.corrcoef(a,b)[0,1])


def prediction_metrics(frame: pd.DataFrame) -> dict[str,Any]:
    y,p,h=frame.realized.to_numpy(),frame.prediction.to_numpy(),frame.historical_mean.to_numpy()
    den=float(np.sum((y-h)**2));num=float(np.sum((y-p)**2))
    s=p-h;e=y-h
    return {'months':len(frame),'mse':float(np.mean((y-p)**2)),'mae':float(np.mean(np.abs(y-p))),
        'oos_r2_vs_past_mean':1-num/den if den>1e-18 else math.nan,
        'oos_r2_na_reason':'' if den>1e-18 else 'zero_benchmark_sse',
        'sign_accuracy':float(np.mean(np.sign(y)==np.sign(p))),
        'timing_direction_accuracy':float(np.mean(np.sign(e)==np.sign(s))),
        'timing_correlation':safe_corr(e,s),
        'correlation_na_reason':'constant_or_insufficient' if not np.isfinite(safe_corr(e,s)) else '',
        'timing_prediction_rms':float(np.sqrt(np.mean(s*s)))}


def performance_metrics(r: np.ndarray) -> dict[str,float]:
    r=np.asarray(r,float)
    if len(r)<2 or not np.isfinite(r).all() or np.any(r<=-1):
        raise ValueError('invalid performance series')
    avg=float(np.mean(r)*12);vol=float(np.std(r,ddof=1)*math.sqrt(12))
    nav=np.r_[1.,np.cumprod(1+r)]
    return {'months':len(r),'annualized_mean':avg,'annualized_vol':vol,
        'mean_vol_ratio_rf0':avg/vol if vol>1e-12 else math.nan,
        'cagr':float(np.expm1(np.sum(np.log1p(r))*12/len(r))),
        'max_drawdown':float(np.min(nav/np.maximum.accumulate(nav)-1)), 'ending_wealth':float(nav[-1])}


def hac_mean(values: np.ndarray,lags: int=3) -> dict[str,Any]:
    x=np.asarray(values,float);n=len(x);mean=float(x.mean())
    base={'monthly_mean':mean,'annualized_difference':12*mean,'lags':lags,'months':n,
          'small_sample_correction':False,'p_approximation':'two_sided_asymptotic_normal'}
    if n<4:
        return {**base,'hac_t':math.nan,'hac_p':math.nan,'hac_se_mean':math.nan,'status':'insufficient_months'}
    z=x-mean;L=min(lags,n-1);lr=float(z@z/n)
    for l in range(1,L+1):
        lr+=2*(1-l/(L+1))*float(z[l:]@z[:-l]/n)
    if np.max(np.abs(x))<1e-14:
        return {**base,'hac_t':math.nan,'hac_p':1.,'hac_se_mean':0.,'ci_annual_low':0.,'ci_annual_high':0.,'status':'identical_paths'}
    if lr<=1e-20:
        return {**base,'hac_t':math.nan,'hac_p':math.nan,'hac_se_mean':math.nan,'status':'degenerate_variance'}
    se=math.sqrt(lr/n);t=mean/se
    return {**base,'hac_t':t,'hac_p':float(2*norm.sf(abs(t))),'hac_se_mean':se,
            'ci_annual_low':12*(mean-1.95996398454*se),'ci_annual_high':12*(mean+1.95996398454*se),'status':'ok'}


def holm_adjust(pvals: np.ndarray) -> np.ndarray:
    p=np.asarray(pvals,float)
    if not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('Holm requires finite p in [0,1]')
    order=np.argsort(p,kind='stable');m=len(p)
    adj=np.minimum(1.,np.maximum.accumulate((m-np.arange(m))*p[order]))
    result=np.empty(m);result[order]=adj
    return result


def block_bootstrap(differences: np.ndarray,block: int,samples: int,seed: int) -> np.ndarray:
    """Paired circular moving blocks. All columns share the SAME sampled dates."""
    x=np.asarray(differences,float)
    if x.ndim==1:x=x[:,None]
    n=len(x)
    if not samples:return np.full((2,x.shape[1]),np.nan)
    b=min(block,n);rng=np.random.default_rng(seed)
    starts=rng.integers(0,n,size=(samples,math.ceil(n/b)))
    indices=((starts[:,:,None]+np.arange(b))%n).reshape(samples,-1)[:,:n]
    # per-column to keep memory bounded
    return np.stack([np.quantile(12*x[indices,j].mean(axis=1),[.025,.975]) for j in range(x.shape[1])],axis=1)


def compare_to_mean_baseline(sr: pd.DataFrame,cfg: Config) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    wide=sr.pivot(index='target_date',columns='strategy',values='net_return').sort_index()
    if wide.isna().any().any() or 'B1' not in wide or 'B0' not in wide:
        raise ValueError('all strategies require paired dates and baselines')
    rows=[]
    for baseline in ['B1','B0']:
        for m in wide:
            if m==baseline:continue
            diff=(wide[m]-wide[baseline]).to_numpy()
            active_vol=float(np.std(diff,ddof=1)*math.sqrt(12))
            for lag in cfg.hac_lags:
                h=hac_mean(diff,lag)
                rows.append({'strategy':m,'baseline':baseline,'primary':baseline=='B1' and m in PRIMARY,
                             'active_vol_annual':active_vol,'information_ratio':12*np.mean(diff)/active_vol if active_vol>1e-12 else math.nan,**h})
    comparisons=pd.DataFrame(rows)
    multiplicity=[]
    for lag in cfg.hac_lags:
        selected=comparisons.loc[(comparisons.baseline=='B1')&(comparisons.lags==lag)].set_index('strategy')
        vals=[]
        for m in PRIMARY:
            p=selected.loc[m,'hac_p'] if m in selected.index else math.nan
            vals.append(float(p) if np.isfinite(p) else 1.)
        adj=holm_adjust(np.asarray(vals))
        for m,p,a in zip(PRIMARY,vals,adj):
            exists=m in selected.index
            realp=float(selected.loc[m,'hac_p']) if exists else math.nan
            multiplicity.append({'strategy':m,'baseline':'B1','lags':lag,'family_size':len(PRIMARY),
                'raw_p':realp,'holm_adjusted_p':float(a) if np.isfinite(realp) else math.nan,
                'adjustment_placeholder':not np.isfinite(realp),'status':'evaluated' if exists else 'not_run',
                'scope':'fixed five current comparisons; NOT past researcher search adjusted'})
    boot=[]
    for baseline in ['B1','B0']:
        cols=[m for m in wide if m!=baseline]
        mat=wide[cols].sub(wide[baseline],axis=0).to_numpy()
        for block in cfg.bootstrap_blocks:
            ci=block_bootstrap(mat,block,cfg.bootstrap_samples,cfg.seed)
            for j,m in enumerate(cols):
                boot.append({'strategy':m,'baseline':baseline,'block_months':block,'samples':cfg.bootstrap_samples,
                    'annualized_difference':float(12*mat[:,j].mean()),'ci_2_5':ci[0,j],'ci_97_5':ci[1,j],
                    'scope':'conditional fixed OOS paths; NOT refitted; NOT multiple-testing-adjusted'})
    return comparisons,pd.DataFrame(multiplicity),pd.DataFrame(boot)


def period_masks(dates: pd.DatetimeIndex) -> list[tuple[str,np.ndarray]]:
    n=len(dates);idx=np.arange(n)
    result=[('ALL',np.ones(n,dtype=bool)),('FIRST_HALF',idx<n//2),('SECOND_HALF',idx>=n//2)]
    for year in sorted(set(dates.year)):
        result.append((f'YEAR_{year}',dates.year==year))
    for i in range(23,n):
        result.append((f'ROLL24_END_{dates[i].date()}',(idx>=i-23)&(idx<=i)))
    return [(name,mask) for name,mask in result if int(mask.sum())>=2]


def evaluate(pred: pd.DataFrame,sr: pd.DataFrame,scenarios: pd.DataFrame,components: pd.DataFrame,cfg: Config) -> dict[str,pd.DataFrame]:
    pred_summary=[];period_strategies=[]
    dates=pd.DatetimeIndex(sorted(sr.target_date.unique()))
    sw=sr.pivot(index='target_date',columns='strategy',values='net_return').sort_index()
    for period,mask in period_masks(dates):
        use=dates[mask]
        for m,part in pred.loc[pred.target_date.isin(use)].groupby('model',sort=False):
            pred_summary.append({'model':m,'period':period,'start':use[0],'end':use[-1],**prediction_metrics(part)})
        for m in sw:
            arr=sw.loc[use,m].to_numpy()
            diff=arr-sw.loc[use,'B1'].to_numpy()
            period_strategies.append({'strategy':m,'period':period,'start':use[0],'end':use[-1],
                **performance_metrics(arr),'annualized_difference_vs_B1':float(12*diff.mean())})
    summary=[]
    for m,part in sr.groupby('strategy',sort=False):
        part=part.sort_values('target_date')
        summary.append({'strategy':m,**performance_metrics(part.net_return.to_numpy()),
            'annualized_cost':float(12*part.cost_fraction.mean()),'annualized_traded_notional':float(12*part.traded_notional_fraction.mean()),
            'average_active_vol_predicted':float(part.predicted_active_vol_annual.mean()),
            'mean_value_weight':float(part.value_weight.mean())})
    costs=[]
    for bps,sdf in scenarios.groupby('cost_bps',sort=True):
        bw=sdf.pivot(index='target_date',columns='strategy',values='net_return').sort_index()
        for m in bw:
            costs.append({'strategy':m,'cost_bps':bps,**performance_metrics(bw[m].to_numpy()),
                'annualized_difference_vs_B1':float(12*(bw[m]-bw.B1).mean()),'positions_reoptimized':False})
    comps,multi,boot=compare_to_mean_baseline(sr,cfg)
    navs=(1+sw).cumprod();nav=[]
    for date,row in navs.iterrows():
        for m in sw:
            nav.append({'target_date':date,'strategy':m,'nav':row[m], 'relative_nav_vs_B1':row[m]/row.B1-1,
                'arithmetic_cumulative_difference_vs_B1':float((sw.loc[:date,m]-sw.loc[:date,'B1']).sum())})
    corr=pd.DataFrame(columns=['component',*BLOCKS])
    if len(components):
        cp=components.pivot(index='target_date',columns='block',values='raw_prediction')
        yy=components.groupby('target_date').realized.first()
        corr=cp.sub(yy,axis=0).corr().rename_axis('component').reset_index()
    return {'prediction_metrics':pd.DataFrame(pred_summary),'strategy_summary':pd.DataFrame(summary),
        'period_strategy_metrics':pd.DataFrame(period_strategies),'strategy_comparison':comps,
        'multiple_testing':multi,'bootstrap_comparison':boot,'cost_sensitivity':pd.DataFrame(costs),
        'nav_paths':pd.DataFrame(nav),'component_error_correlations':corr}

# ====================== 06 PUBLIC API / EXPORT ======================
def run_experiment(data: GroupData,cfg: Config,saved_predictions: pd.DataFrame | None=None,
                   *, synthetic: bool=False) -> Result:
    cfg.validate()
    pred,components,diag,registry=walk_forward_spread(data,cfg,saved_predictions)
    sr,w,scenarios=simulate_portfolios(data,pred,cfg)
    tables={'spread_predictions':pred,'component_predictions':components,'diagnostics':diag,
            'experiment_registry':registry,'strategy_returns':sr,'weights':w,'cost_scenario_returns':scenarios}
    tables.update(evaluate(pred,sr,scenarios,components,cfg))
    group_rows=[]
    for i,date in enumerate(data.dates):
        for j,g in enumerate(GROUPS):
            group_rows.append({'target_date':date,'group':g,'realized_return':data.group_returns[i,j],
                               'membership_version':data.metadata['membership_version']})
    tables['group_returns']=pd.DataFrame(group_rows)
    tables['model_selection']=pd.DataFrame([{'model':m,'selection':'fixed_no_new_grid',
        'detail':'external A1 selection history not audited' if m in ('A1','A2') and cfg.core_mode=='saved' else 'fixed lambda / specified formulas'} for m in pred.model.unique()])
    manifest={'program':'value_gm_research','version':VERSION,'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'synthetic':synthetic,'config':dataclasses.asdict(cfg),'input_metadata':data.metadata,
        'python_version':platform.python_version(),'dependencies':{p:importlib.metadata.version(p) for p in ('numpy','pandas','scipy')},
        'script_sha256':sha(__file__),'factor_count':len(data.factors),'factors':list(data.factors),
        'first_target_date':str(pred.target_date.min().date()),'last_target_date':str(pred.target_date.max().date()),
        'oos_months':int(pred.target_date.nunique()),'strategy_count':int(sr.strategy.nunique()),
        'core_sample_is_not_existing_user_model':cfg.core_mode=='sample_local_ridge',
        'known_limits':['no new feature selection or hyperparameter grid','external saved prediction history not independently verified',
            'availability timestamp check is not point-in-time restatement audit',
            'monthly same-close execution approximation','factor P&L costs are reallocation proxies, not security execution costs',
            'rf0 mean/vol is not an actual excess-return Sharpe','inference conditional on observed OOS paths',
            'P3 TVP / calibration estimation / residual corrections NOT implemented',
            'already inspected OOS and selected core universe are exploratory, not virgin holdout']}
    return Result(tables,manifest)


def table_md(df: pd.DataFrame,cols: list[str]) -> str:
    def f(v: Any) -> str:
        if isinstance(v,(float,np.floating)):return f'{v:.6f}' if np.isfinite(v) else 'NA'
        return str(v).replace('|','/')
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(f(v) for v in row)+' |' for row in df[cols].itertuples(index=False,name=None))


def reserve_output(out: str | Path) -> Path:
    path=Path(out)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'出力先が空ではありません: {path}。既存結果を上書きしません。')
    path.mkdir(parents=True,exist_ok=True)
    return path


def export_results(result: Result,out: str | Path) -> None:
    path=Path(out);path.mkdir(parents=True,exist_ok=True)
    for name,df in result.tables.items():
        df.to_csv(path/f'{name}.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
    (path/'audit_manifest.json').write_text(json.dumps(clean_json(result.manifest),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    syn=result.manifest['synthetic'];sig=result.tables['prediction_metrics']
    sig=sig.loc[sig.period=='ALL']
    status='合成データの動作確認。実際の投資効果ではありません。' if syn else '指定データの探索的比較。PIT・選択履歴・執行可能性は別途確認してください。'
    report=f'''# Value / Growth Momentum 比較結果\n\n**{status}**\n\nOOS {result.manifest['oos_months']}か月、{result.manifest['strategy_count']}配分系列。\n\n## 実行状況\n\n{table_md(result.tables['experiment_registry'],['model','description','status','reason'])}\n\n## 予測比較（表示は単位小数）\n\n{table_md(sig,['model','months','oos_r2_vs_past_mean','mse','timing_correlation'])}\n\n## 戦略比較\n\n{table_md(result.tables['strategy_summary'],['strategy','annualized_mean','annualized_vol','mean_vol_ratio_rf0','annualized_cost'])}\n\n## 必ず確認すること\n\nB1比は strategy_comparison.csv。比較対象のt値同士を引き算しない。\nmultiple_testing.csvは現在の5比較のみのHolm調整で、過去の探索を解消しない。\nbootstrap_comparison.csvは実現済みOOS経路に条件付けた区間で、モデルを再推定しない。\nperiod_strategy_metrics.csvとprediction_metrics.csvの期間別結果で一部期間への依存を確認する。\nnav_paths.csvの相対NAVはNAV比であり、リターン差を複利にしていない。\nデモのA1はsample local Ridgeであり、ユーザーの13因子モデルを再現したものではない。\n\n## 未実装・制約\n\nP3の時変係数、残差補正、校正係数推定、追加因子選択は未実装。\nコスト感応度は同じポジションの費用差し替えであり費用別再最適化ではない。\n原資産の内部回転・借株・資金調達・相殺・実際の約定は別途対応する。\n結果を見てモデル・ユニバースを選んだ期間は未使用テストとして扱えない。\n'''
    (path/'REPORT.md').write_text(report,encoding='utf-8')


def read_runtime(path: str | Path) -> tuple[Config,dict[str,Path | None],dict[str,Any]]:
    p=Path(path);raw=json.loads(p.read_text(encoding='utf-8-sig'))
    if set(raw)-{'settings','inputs','notes','dataset_kind'} or 'settings' not in raw or 'inputs' not in raw:
        raise ValueError('runtime設定はsettings/inputs/notes/dataset_kindのみ。config_spec.jsonは使えません。')
    if raw.get('dataset_kind','real') not in ('real','synthetic'):
        raise ValueError('dataset_kind must be real or synthetic')
    params=dict(raw['settings'])
    unknown=set(params)-{f.name for f in dataclasses.fields(Config)}
    if unknown:raise ValueError('unknown settings: '+str(sorted(unknown)))
    for c in ['cost_scenarios_bps','bootstrap_blocks','hac_lags','core_factor_ids']:
        if c in params:params[c]=tuple(params[c])
    cfg=Config(**params);cfg.validate(real=raw.get('dataset_kind','real')=='real')
    allowed={'returns','features','membership','predictions'}
    if set(raw['inputs'])-allowed:raise ValueError('unknown input key')
    paths={k:(None if not raw['inputs'].get(k) else (p.parent/str(raw['inputs'][k])).resolve()) for k in allowed}
    for k in ('returns','membership'):
        if paths[k] is None:raise ValueError(f'{k} input required')
    for k,v in paths.items():
        if v is not None and not v.is_file():raise FileNotFoundError(v)
    return cfg,paths,raw

# ====================== 07 SYNTHETIC / TESTS ======================
def make_synthetic(months: int=161,n_value: int=7,n_gm: int=6,seed: int=42) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    """13 factors by default. Shared shocks + drifting relative signal, not empirical data."""
    if months<24 or min(n_value,n_gm)<1:raise ValueError('invalid synthetic dimensions')
    rng=np.random.default_rng(seed);n=n_value+n_gm
    dates=pd.date_range('2013-06-30',periods=months,freq=pd.offsets.MonthEnd())
    factors=[f'VALUE_{i+1:02}' for i in range(n_value)]+[f'GM_{i+1:02}' for i in range(n_gm)]
    x=np.zeros((months,n,2));x[0]=rng.normal(size=(n,2))
    for t in range(1,months):
        x[t]=.85*x[t-1]+.4*rng.normal(size=(n,2))+.2*rng.normal(size=(1,2))
    r=np.zeros((months,n));r[0]=rng.normal(.002,.025,n)
    for t in range(1,months):
        drift=.5+np.sin(t/20)
        r[t]=.002+.003*drift*x[t-1,:,0]+.002*x[t-1,:,1]+rng.normal(0,.012)+rng.normal(0,.020,n)
    observed=x.copy();observed[rng.random(x.shape)<.01]=np.nan
    rf=pd.DataFrame(r,columns=factors);rf.insert(0,'date',dates)
    ff=pd.DataFrame(observed.reshape(-1,2),columns=['x_value_spread','x_profitability_change'])
    ff.insert(0,'available_at',np.repeat((dates-pd.Timedelta(days=1)).to_numpy(),n))
    ff.insert(0,'factor',np.tile(factors,months));ff.insert(0,'date',np.repeat(dates.to_numpy(),n))
    mem=pd.DataFrame({'group':['VALUE']*n_value+['GROWTH_MOMENTUM']*n_gm,'factor_id':factors,
        'weight':[1/n_value]*n_value+[1/n_gm]*n_gm,'return_orientation':1,
        'membership_version':'synthetic_v1','definition_decided_at':'SYNTHETIC_NOT_REAL'})
    return rf,ff,mem


def write_templates(out: str | Path) -> None:
    path=reserve_output(out)
    pd.DataFrame(columns=['date','YOUR_VALUE_FACTOR','YOUR_GM_FACTOR']).to_csv(path/'returns.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(columns=['date','factor','available_at','x_value_spread','x_profitability_change']).to_csv(path/'features.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(columns=['group','factor_id','weight','return_orientation','membership_version','definition_decided_at']).to_csv(path/'group_membership.csv',index=False,encoding='utf-8-sig')
    cfg=dataclasses.asdict(Config(core_source_model='REPLACE_WITH_ACTUAL_MODEL_NAME'))
    raw={'dataset_kind':'real','settings':cfg,'inputs':{'returns':'returns.csv','features':'features.csv','membership':'group_membership.csv','predictions':None},
        'notes':'空ひな形。core_scope, core_factor_ids, feature_orientation_confirmed, core_source_modelを監査後に設定。既存予測のパスを追加。'}
    (path/'runtime_config.json').write_text(json.dumps(clean_json(raw),ensure_ascii=False,indent=2),encoding='utf-8')


def self_test() -> dict[str,Any]:
    """Deterministic checks; no requirement that any strategy make money."""
    tests=[]
    def test(name):
        def deco(fn):tests.append((name,fn));return fn
        return deco
    def raises(kind,fn):
        try:fn()
        except kind:return
        raise AssertionError(f'Expected {kind.__name__}')
    rf,ff,mem=make_synthetic(55,2,2,7)
    cfg=Config(min_train=24,trend_months=6,core_mode='sample_local_ridge',core_scope='prediction_targets',
        core_factor_ids=tuple(mem.factor_id),feature_orientation_confirmed=True,bootstrap_samples=60)
    data=build_group_panel(rf,mem,ff,cfg)
    cache={}
    def experiment():
        if 'r' not in cache:cache['r']=run_experiment(data,cfg,synthetic=True)
        return cache['r']
    @test('input_not_mutated_and_group_weight_identity')
    def _():
        before=rf.copy(deep=True);d=build_group_panel(rf,mem,ff,cfg)
        pd.testing.assert_frame_equal(before,rf)
        np.testing.assert_allclose(d.group_returns,d.returns@d.A,atol=1e-12)
    @test('unresolved_real_core_rejected')
    def _():raises(ValueError,lambda:Config(core_source_model='x').validate(real=True))
    @test('duplicate_month_rejected')
    def _():raises(ValueError,lambda:build_group_panel(pd.concat([rf,rf.iloc[:1]]),mem,ff,cfg))
    @test('missing_month_rejected')
    def _():raises(ValueError,lambda:build_group_panel(rf.drop(index=5),mem,ff,cfg))
    @test('missing_return_rejected')
    def _():
        bad=rf.copy();bad.iloc[2,1]=np.nan
        raises(ValueError,lambda:build_group_panel(bad,mem,ff,cfg))
    @test('bad_group_weight_rejected')
    def _():
        bad=mem.copy();bad.loc[0,'weight']=.7
        raises(ValueError,lambda:build_group_panel(rf,bad,ff,cfg))
    @test('overlap_membership_rejected')
    def _():raises(ValueError,lambda:build_group_panel(rf,pd.concat([mem,mem.iloc[:1]]),ff,cfg))
    @test('future_availability_rejected')
    def _():
        bad=ff.copy();bad.loc[0,'available_at']=bad.loc[0,'date']+pd.Timedelta(days=1)
        raises(ValueError,lambda:build_group_panel(rf,mem,bad,cfg))
    @test('unknown_pit_requires_explicit_opt_in')
    def _():
        bad=ff.drop(columns='available_at')
        raises(ValueError,lambda:build_group_panel(rf,mem,bad,cfg))
        d=build_group_panel(rf,mem,bad,replace(cfg,allow_unverified_pit=True))
        assert not d.metadata['timestamp_check_passed']
    @test('group_feature_nan_not_renormalized')
    def _():
        bad=ff.copy();bad.loc[(bad.date==rf.date.iloc[0])&(bad.factor==mem.factor_id.iloc[0]),'x_value_spread']=np.nan
        d=build_group_panel(rf,mem,bad,cfg)
        assert np.isnan(d.group_features[0,0,0])
    @test('total_return_sign_flip_rejected')
    def _():
        bad=mem.copy();bad.loc[0,'return_orientation']=-1
        raises(ValueError,lambda:build_group_panel(rf,bad,ff,replace(cfg,return_kind='sleeve_total')))
    @test('ridge_difference_linearity')
    def _():
        rng=np.random.default_rng(1);x=rng.normal(size=(40,3));new=rng.normal(size=(1,3));a=rng.normal(size=40);b=rng.normal(size=40)
        direct,_=forecast_spread_once(x,a-b,new[0],cfg)
        pa,_=forecast_spread_once(x,a,new[0],cfg);pb,_=forecast_spread_once(x,b,new[0],cfg)
        np.testing.assert_allclose(direct,pa-pb,atol=1e-10)
    @test('ridge_mean_loss_replication_invariance')
    def _():
        rng=np.random.default_rng(2);x=rng.normal(size=(30,3));y=rng.normal(size=30);new=x[:2]
        np.testing.assert_allclose(ridge_predict(x,y,new,1.),ridge_predict(np.tile(x,(2,1)),np.tile(y,2),new,1.),atol=1e-10)
    @test('ridge_matches_penalized_normal_equations')
    def _():
        rng=np.random.default_rng(3);x=rng.normal(size=(24,4));y=rng.normal(size=24);new=x[:3]
        xx=np.column_stack([np.ones(len(x)),x]);pen=np.diag([0]+[len(x)*.4]*4)
        coef=np.linalg.solve(xx.T@xx+pen,xx.T@y)
        np.testing.assert_allclose(ridge_predict(x,y,new,.4),np.column_stack([np.ones(3),new])@coef,atol=1e-10)
    @test('all_missing_constant_features_mean_fallback_logged')
    def _():
        x=np.full((24,2),np.nan);y=np.arange(24.)
        p,di=forecast_spread_once(x,y,np.array([np.nan,np.nan]),cfg)
        assert p==y.mean() and di['status']=='no_active_features'
    @test('scaler_future_extreme_does_not_change_fit')
    def _():
        z=np.arange(60.).reshape(30,2);s=TrainScaler().fit(z);before=s.mean.copy()
        s.transform(np.full((1,2),1e9));np.testing.assert_array_equal(before,s.mean)
    @test('shrinkage_endpoints_and_midpoint')
    def _():
        assert shrink_prediction(.02,.01,0)==.01 and shrink_prediction(.02,.01,1)==.02
        np.testing.assert_allclose(shrink_prediction(.02,.01,.5),.015)
    @test('all_seven_strategies_run_and_d1_mean')
    def _():
        out=experiment();p=out.tables['spread_predictions'];c=out.tables['component_predictions']
        assert out.manifest['strategy_count']==7
        avg=c.groupby('target_date').raw_prediction.mean()
        d=p[p.model=='D1'].set_index('target_date').raw_prediction
        np.testing.assert_allclose(d.sort_index(),avg.sort_index(),atol=1e-12)
    @test('common_history_and_one_month_alignment')
    def _():
        p=experiment().tables['spread_predictions'];first=p.iloc[0];t=data.dates.get_loc(first.decision_date)
        np.testing.assert_allclose(first.historical_mean,data.spread[cfg.trend_months:t+1].mean(),atol=1e-12)
        assert first.target_date==data.dates[t+1] and first.n_train==cfg.min_train
        y=data.returns[cfg.trend_months:t+1].mean(axis=0)@data.A
        np.testing.assert_allclose(y[0]-y[1],first.historical_mean,atol=1e-12)
    @test('sample_local_unrelated_factor_invariance')
    def _():
        rr=rf.copy();rr['UNRELATED']=np.linspace(-.1,.1,len(rr))
        d=build_group_panel(rr,mem,ff,cfg)
        p,*_=walk_forward_spread(d,cfg)
        pd.testing.assert_frame_equal(p,experiment().tables['spread_predictions'])
    @test('missing_d1_block_recorded_not_replaced')
    def _():
        f=ff.drop(columns='x_profitability_change');d=build_group_panel(rf,mem,f,cfg)
        p,c,di,reg=walk_forward_spread(d,cfg)
        assert 'D1' not in set(p.model)
        assert reg.set_index('model').loc['D1','status']=='not_run'
    @test('prefix_invariance_forecasts_and_weights')
    def _():
        end=47;d=build_group_panel(rf.iloc[:end],mem,ff[ff.date<=rf.date.iloc[end-1]],cfg)
        short=run_experiment(d,cfg,synthetic=True);full=experiment()
        for key,datecol in [('spread_predictions','target_date'),('weights','target_date')]:
            a=full.tables[key].loc[full.tables[key][datecol]<=rf.date.iloc[end-1]].reset_index(drop=True)
            pd.testing.assert_frame_equal(a,short.tables[key].reset_index(drop=True))
    @test('future_mutation_does_not_change_past_predictions_or_weights')
    def _():
        cutoff=rf.date.iloc[44];rr=rf.copy();rr.loc[rr.date>=cutoff,rr.columns[1:]]+=.1
        f=ff.copy();f.loc[f.date>=cutoff,['x_value_spread','x_profitability_change']]+=100
        d=build_group_panel(rr,mem,f,cfg);other=run_experiment(d,cfg,synthetic=True)
        for key,cols in [('spread_predictions',['decision_date','model','prediction']),('weights',['decision_date','strategy','factor','weight'])]:
            a=experiment().tables[key];b=other.tables[key]
            pd.testing.assert_frame_equal(a.loc[a.decision_date<cutoff,cols].reset_index(drop=True),b.loc[b.decision_date<cutoff,cols].reset_index(drop=True))
    @test('negative_fixture_full_sample_scaler_leak_is_detectable')
    def _():
        a=np.arange(100.).reshape(50,2);b=a.copy();b[40:]+=1e6
        la=TrainScaler().fit(a).transform(a[:1]);lb=TrainScaler().fit(b).transform(b[:1])
        assert not np.allclose(la,lb)  # demonstrates what prefix test would catch
    @test('allocation_exact_solution_vs_dense_grid')
    def _():
        A=data.A;pre=A@np.array([.6,.4]);mu=.003;v=.0008
        q,info=allocate_two_group_tilt(mu,v,A,pre,cfg)
        grid=np.linspace(-.2,.2,20001);curv=cfg.risk_aversion*v+2*cfg.anchor_penalty
        obj=np.array([-mu*b+.5*curv*b*b+.001*np.abs(A@np.array([.5+b,.5-b])-pre).sum() for b in grid])
        assert info['objective']<=obj.min()+1e-10
    @test('allocation_bounds_and_gross_identity')
    def _():
        s=experiment().tables['strategy_returns']
        assert np.allclose(s.value_weight+s.growth_momentum_weight,1)
        assert s.value_weight.between(.3-1e-10,.7+1e-10).all()
        assert (s.predicted_active_vol_annual<=cfg.max_active_vol_annual+1e-10).all()
        for row in s.itertuples():
            i=data.dates.get_loc(row.target_date)
            base=.5*data.group_returns[i].sum()
            np.testing.assert_allclose(row.gross_return-base,row.tilt*data.spread[i],atol=1e-10)
    @test('l1_two_times_delta_b_and_initial_proxy_cost')
    def _():
        w1=data.A@np.array([.6,.4]);w2=data.A@np.array([.45,.55])
        np.testing.assert_allclose(np.abs(w1-w2).sum(),2*.15,atol=1e-12)
        first=experiment().tables['strategy_returns'].groupby('strategy',sort=False).head(1)
        np.testing.assert_allclose(first.cost_fraction,.001,atol=1e-12)
    @test('zero_signal_shrink_equals_b1_whole_path')
    def _():
        p=experiment().tables['spread_predictions'].copy();p=p[p.model.isin(['B1','A1'])].copy()
        p.loc[p.model=='A1','prediction']=p.loc[p.model=='A1','historical_mean']
        s,_,_=simulate_portfolios(data,p,cfg)
        a=s[s.strategy=='A1'].reset_index(drop=True);b=s[s.strategy=='B1'].reset_index(drop=True)
        np.testing.assert_allclose(a[['value_weight','net_return']],b[['value_weight','net_return']],atol=1e-12)
    @test('funded_initial_cost_and_total_return_drift')
    def _():
        total=replace(cfg,return_kind='sleeve_total');p=experiment().tables['spread_predictions']
        s,w,_=simulate_portfolios(data,p,total)
        first=s.groupby('strategy').head(1);np.testing.assert_allclose(first.cost_fraction,.001/1.001,atol=1e-12)
        b=s[s.strategy=='B0'].sort_values('target_date');date=b.iloc[0].target_date;i=data.dates.get_loc(date)
        base=data.A@np.array([.5,.5]);drift=base*(1+data.returns[i])/(1+base@data.returns[i])
        actual=w[(w.strategy=='B0')&(w.target_date==b.iloc[1].target_date)].set_index('factor').reindex(data.factors).pretrade_weight
        np.testing.assert_allclose(actual,drift,atol=1e-12)
    @test('funded_net_bridge_cross_term')
    def _():
        gm,gb,cm,cb=.03,.01,.002,.001
        direct=(1-cm)*(1+gm)-(1-cb)*(1+gb)
        bridge=(gm-gb)-(cm-cb)-(cm*gm-cb*gb)
        np.testing.assert_allclose(direct,bridge,atol=1e-12)
    @test('zero_sse_and_constant_correlation_are_na')
    def _():
        f=pd.DataFrame({'prediction':[0.,0.,0.],'realized':[0.,0.,0.],'historical_mean':[0.,0.,0.]})
        met=prediction_metrics(f);assert np.isnan(met['oos_r2_vs_past_mean']) and np.isnan(met['timing_correlation'])
    @test('hac_zero_lag_and_identical_paths')
    def _():
        x=np.array([.01,-.02,.03,.04,.01]);h=hac_mean(x,0)
        np.testing.assert_allclose(h['hac_se_mean'],np.std(x,ddof=0)/np.sqrt(len(x)),atol=1e-12)
        assert hac_mean(np.zeros(10),3)['hac_p']==1.
    @test('holm_known_values')
    def _():np.testing.assert_allclose(holm_adjust(np.array([.01,.04,.03])),[.03,.06,.06],atol=1e-12)
    @test('paired_bootstrap_reproducibility')
    def _():
        x=np.arange(30.)/1000;mat=np.column_stack([x,2*x])
        a=block_bootstrap(mat,6,100,1);b=block_bootstrap(mat,6,100,1)
        np.testing.assert_array_equal(a,b);np.testing.assert_allclose(a[:,1],2*a[:,0],atol=1e-12)
    @test('relative_nav_is_nav_ratio_not_compounded_difference')
    def _():
        out=experiment();nav=out.tables['nav_paths'];sr=out.tables['strategy_returns']
        pivot=sr.pivot(index='target_date',columns='strategy',values='net_return');truth=(1+pivot).cumprod()
        a=nav[nav.strategy=='A1'].set_index('target_date')
        np.testing.assert_allclose(a.relative_nav_vs_B1,truth.A1/truth.B1-1,atol=1e-12)
    @test('saved_predictions_alignment_missing_rows_rejected')
    def _():
        rows=[]
        for t in range(cfg.trend_months-1+cfg.min_train,len(data.dates)-1):
            for k in data.factors:rows.append({'decision_date':data.dates[t],'target_date':data.dates[t+1],'model':'saved_core','factor':k,'prediction':.001})
        saved=pd.DataFrame(rows);sc=replace(cfg,core_mode='saved',core_source_model='saved_core')
        p,*_=walk_forward_spread(data,sc,saved)
        assert set(p.model)==set(PRED_MODELS)
        raises(ValueError,lambda:walk_forward_spread(data,sc,saved.drop(index=1)))
    @test('no_features_saved_only_is_partial_not_fabricated')
    def _():
        c=replace(cfg,core_mode='none');d=build_group_panel(rf,mem,None,c)
        p,com,di,reg=walk_forward_spread(d,c)
        assert set(p.model)=={'B1'} and len(reg[reg.status=='not_run'])==5
    @test('reallocation_from_saved_spread_forecasts_matches')
    def _():
        out=experiment();s,w,sc=simulate_portfolios(data,out.tables['spread_predictions'],cfg)
        pd.testing.assert_frame_equal(s,out.tables['strategy_returns'])
    @test('export_and_nonoverwrite_guard')
    def _():
        with tempfile.TemporaryDirectory() as td:
            path=reserve_output(Path(td)/'r');export_results(experiment(),path)
            assert (path/'REPORT.md').is_file() and (path/'audit_manifest.json').is_file()
            assert len(list(path.glob('*.csv')))>=15
            raises(FileExistsError,lambda:reserve_output(path))
    passed=[]
    for name,fn in tests:
        fn();passed.append(name);print('PASS',name)
    result={'passed':len(passed),'tests':passed,'status':'passed','test_scope':'sample implementation, not complete project acceptance or economic validity'}
    print(dump(result));return result

# ====================== 08 CLI ======================
def main(argv: list[str] | None=None) -> int:
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=parser.add_subparsers(dest='command',required=True)
    test=sub.add_parser('self-test');test.add_argument('--out',help='optional new directory for test report')
    temp=sub.add_parser('templates');temp.add_argument('--out',required=True)
    demo=sub.add_parser('demo');demo.add_argument('--out',required=True);demo.add_argument('--months',type=int,default=161)
    demo.add_argument('--seed',type=int,default=42);demo.add_argument('--return-kind',choices=['factor_pnl','sleeve_total'],default='factor_pnl')
    demo.add_argument('--bootstrap-samples',type=int,default=5000)
    for name in ['run','reallocate']:
        p=sub.add_parser(name);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
        if name=='reallocate':p.add_argument('--spread-predictions',required=True)
    args=parser.parse_args(argv)
    try:
        if args.command=='self-test':
            directory=reserve_output(args.out) if args.out else None
            result=self_test()
            if directory:
                (directory/'TEST_REPORT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
                (directory/'TEST_REPORT.md').write_text('# サンプル自動テスト\n\n'+f"{result['passed']}項目 PASS。投資効果は検証していません。\n\n"+'\n'.join('- '+v for v in result['tests']),encoding='utf-8')
            return 0
        if args.command=='templates':write_templates(args.out);return 0
        out=reserve_output(args.out)
        if args.command=='demo':
            rf,ff,mem=make_synthetic(args.months,seed=args.seed)
            cfg=Config(core_mode='sample_local_ridge',core_scope='prediction_targets',core_factor_ids=tuple(mem.factor_id),
                feature_orientation_confirmed=True,seed=args.seed,return_kind=args.return_kind,bootstrap_samples=args.bootstrap_samples)
            data=build_group_panel(rf,mem,ff,cfg)
            result=run_experiment(data,cfg,synthetic=True)
            inputs=out/'demo_inputs';inputs.mkdir()
            rf.to_csv(inputs/'returns.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
            ff.to_csv(inputs/'features.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d')
            mem.to_csv(inputs/'group_membership.csv',index=False,encoding='utf-8-sig')
            runtime={'dataset_kind':'synthetic','settings':dataclasses.asdict(cfg),'inputs':{'returns':'returns.csv','features':'features.csv','membership':'group_membership.csv','predictions':None},
                     'notes':'SYNTHETIC DEMO INPUTS; not actual market data'}
            (inputs/'runtime_config.json').write_text(json.dumps(clean_json(runtime),ensure_ascii=False,indent=2),encoding='utf-8')
        else:
            cfg,paths,raw=read_runtime(args.config)
            rf=pd.read_csv(paths['returns']);mem=pd.read_csv(paths['membership'])
            ff=pd.read_csv(paths['features']) if paths['features'] else None
            saved=pd.read_csv(paths['predictions']) if paths['predictions'] else None
            data=build_group_panel(rf,mem,ff,cfg)
            data.metadata['source_file_sha256']={k:sha(p) for k,p in paths.items() if p is not None}
            data.metadata['runtime_config_sha256']=sha(args.config)
            if args.command=='run':
                result=run_experiment(data,cfg,saved,synthetic=raw.get('dataset_kind','real')=='synthetic')
            else:
                predictions=validate_spread_predictions(pd.read_csv(args.spread_predictions),data)
                sr,w,sc=simulate_portfolios(data,predictions,cfg)
                empty=pd.DataFrame(columns=['block','target_date','raw_prediction','realized'])
                tables={'spread_predictions':predictions,'strategy_returns':sr,'weights':w,'cost_scenario_returns':sc,
                    'experiment_registry':pd.DataFrame([{'model':m,'description':DESCRIPTIONS[m],'status':'reallocated','reason':'predictions_not_refitted'} for m in sr.strategy.unique()])}
                tables.update(evaluate(predictions,sr,sc,empty,cfg))
                result=Result(tables,{'program':'value_gm_research','version':VERSION,'synthetic':raw.get('dataset_kind','real')=='synthetic',
                    'oos_months':predictions.target_date.nunique(),'strategy_count':sr.strategy.nunique(),
                    'config':dataclasses.asdict(cfg),'input_metadata':data.metadata,
                    'source_spread_predictions_sha256':sha(args.spread_predictions),'script_sha256':sha(__file__),
                    'mode':'reallocate_only; source origin/training provenance must be audited separately'})
        export_results(result,out)
        print(dump({'output':str(out.resolve()),'synthetic':result.manifest['synthetic'],
            'oos_months':result.manifest['oos_months'],'strategies':result.manifest['strategy_count']}))
        return 0
    except (ValueError,KeyError,FileNotFoundError,FileExistsError,RuntimeError,np.linalg.LinAlgError) as exc:
        print('ERROR: '+str(exc),file=sys.stderr)
        if 'out' in locals() and Path(out).is_dir():
            (Path(out)/'FAILED.json').write_text(dump({'status':'failed','error':str(exc)}),encoding='utf-8')
        return 2

if __name__=='__main__':
    raise SystemExit(main())
