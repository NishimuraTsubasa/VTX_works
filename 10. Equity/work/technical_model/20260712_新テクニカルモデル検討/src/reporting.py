from __future__ import annotations

from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import font_manager

from .ensemble import MODEL_COLS, EVIDENCE_COLS


def setup_font() -> None:
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams['font.family'] = 'Noto Sans CJK JP'
    plt.rcParams['axes.unicode_minus'] = False


def transformer_explain(model, meta: dict, latest_date, asset_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = meta['test_meta']
    mask = (pd.to_datetime(m.date) == pd.Timestamp(latest_date)) & (m.asset_id == asset_id)
    if not mask.any():
        mask = pd.to_datetime(m.date) == pd.Timestamp(latest_date)
    idx = np.where(mask.to_numpy())[0]
    if len(idx) == 0:
        idx = np.array([len(m) - 1])
    seq = meta['test_sequences'][idx]
    base_pred, attn = model.predict_with_attention(seq)
    attention = attn.mean(axis=0)
    attention = attention / attention.sum()
    n_patches = len(attention)
    patch = int(meta['patch_size'])
    rows = []
    for p, val in enumerate(attention):
        oldest_lag = (n_patches - p) * patch
        newest_lag = oldest_lag - patch + 1
        rows.append({'patch': p + 1, 'lag_start': newest_lag, 'lag_end': oldest_lag, 'attention': float(val)})
    patch_df = pd.DataFrame(rows)

    buckets = [(1, 5, '1-5日'), (6, 20, '6-20日'), (21, 60, '21-60日'), (61, 120, '61-120日'), (121, 252, '121-252日')]
    bucket_rows = []
    for lo, hi, label in buckets:
        bucket_attn = patch_df[(patch_df.lag_start >= lo) & (patch_df.lag_end <= hi)].attention.sum()
        occluded = seq.copy()
        start = max(0, seq.shape[1] - hi)
        end = max(0, seq.shape[1] - lo + 1)
        occluded[:, start:end, :] = 0.5
        impact = float(np.mean(np.abs(base_pred - model.predict(occluded))))
        bucket_rows.append({'period': label, 'attention_share': float(bucket_attn), 'occlusion_impact': impact})
    return patch_df, pd.DataFrame(bucket_rows)


def _style_axis(ax) -> None:
    ax.grid(True, alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)


def _context_table(daily: pd.DataFrame, latest_date, asset_id: str, attention_bucket: pd.DataFrame) -> list[list[object]]:
    g = daily[(daily.asset_id == asset_id) & (daily.date <= latest_date)].sort_values('date')
    att_map = dict(zip(attention_bucket.period, attention_bucket.attention_share))
    periods = [('1-5日',1,5), ('6-20日',6,20), ('21-60日',21,60), ('61-120日',61,120), ('121-252日',121,252)]

    def state(v: float, vol: bool = False) -> str:
        if vol:
            return 'Calm' if v < 0.30 else 'Neutral' if v < 0.60 else 'Elevated' if v < 0.80 else 'Stress'
        return 'Strong' if v >= 0.80 else 'Bullish' if v >= 0.60 else 'Neutral' if v >= 0.40 else 'Weak' if v >= 0.20 else 'Bearish'

    rows = []
    for label, lo, hi in periods:
        sub = g.iloc[max(0, len(g)-hi):len(g)-lo+1]
        mom = float(sub[['ret20_pct','ret60_pct','ret120_pct']].mean().mean())
        trend = float(sub[['ma_gap60_pct','ma_slope60_pct','breakout120_pct']].mean().mean())
        vol = float(sub[['rv20_pct','vol_ratio_pct','vix_pct']].mean().mean())
        rows.append([label, float(att_map.get(label, 0.0)), state(mom), state(trend), state(vol, True)])
    return rows


def build_pdf(root: Path, pred: pd.DataFrame, model_weights: pd.DataFrame, model_ic: pd.DataFrame,
              evidence_weights: pd.DataFrame, bt: pd.DataFrame, latest_portfolio: pd.DataFrame,
              metrics: pd.DataFrame, attention_bucket: pd.DataFrame, patch_attention: pd.DataFrame,
              daily: pd.DataFrame | None = None, attention_asset: str = 'NKY') -> Path:
    setup_font()
    out = root / 'outputs' / 'report' / 'Model_Report.pdf'
    latest_date = pred.date.max()
    latest_pred = pred[pred.date == latest_date].copy()
    evidence_labels = ['Persistence','Correction','Volatility Support','Flow','Relative Strength','Intermarket','Macro Market']

    with PdfPages(out) as pdf:
        # Page 1: portfolio/performance only
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27))
        fig.suptitle('Portfolio & Performance Summary', fontsize=18, fontweight='bold')
        p = latest_portfolio.sort_values('dynamic_weight')
        axes[0,0].barh(p.asset_id, p.dynamic_weight * 100); axes[0,0].axvline(0, color='k', lw=.7)
        axes[0,0].set_title(f'Latest portfolio weights ({pd.Timestamp(latest_date):%Y-%m-%d})'); axes[0,0].set_xlabel('Weight (%)'); _style_axis(axes[0,0])
        axes[0,1].plot(bt.date, bt.cum_dynamic_return_net, label='Dynamic'); axes[0,1].plot(bt.date, bt.cum_static_return_net, label='Static'); axes[0,1].plot(bt.date, bt.cum_benchmark_return, label='Benchmark', alpha=.7)
        axes[0,1].set_title('Cumulative performance'); axes[0,1].legend(); _style_axis(axes[0,1])
        axes[1,0].plot(bt.date, bt.dynamic_drawdown, label='Dynamic'); axes[1,0].plot(bt.date, bt.static_drawdown, label='Static'); axes[1,0].set_title('Drawdown'); axes[1,0].legend(); _style_axis(axes[1,0])
        r = bt.dynamic_return_net.rolling(12); axes[1,1].plot(bt.date, r.mean()/r.std()*np.sqrt(12)); axes[1,1].axhline(0, color='k', lw=.7); axes[1,1].set_title('Dynamic 12-month rolling Sharpe'); _style_axis(axes[1,1])
        fig.tight_layout(rect=[0,0,1,.94]); pdf.savefig(fig); plt.close(fig)

        # Page 2: evidence only
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27)); fig.suptitle('Evidence State & Attribution', fontsize=18, fontweight='bold')
        mat = latest_pred.set_index('asset_id')[EVIDENCE_COLS].T
        im = axes[0,0].imshow(mat, aspect='auto', vmin=0, vmax=1, cmap='RdYlGn'); axes[0,0].set_yticks(range(7), evidence_labels); axes[0,0].set_xticks(range(len(mat.columns)), mat.columns, rotation=45); axes[0,0].set_title('Latest evidence heatmap'); fig.colorbar(im, ax=axes[0,0], fraction=.03)
        last = evidence_weights.iloc[-1]; axes[0,1].barh(evidence_labels, [last[c]*100 for c in EVIDENCE_COLS]); axes[0,1].set_title('Latest evidence weights'); axes[0,1].set_xlabel('Weight (%)'); _style_axis(axes[0,1])
        score = latest_pred.set_index('asset_id')[EVIDENCE_COLS].sub(.5).mul(last[EVIDENCE_COLS].values, axis=1).sum(axis=1).sort_values(); axes[1,0].barh(score.index, score.values); axes[1,0].axvline(0, color='k', lw=.7); axes[1,0].set_title('Evidence-only composite score'); _style_axis(axes[1,0])
        axes[1,1].axis('off'); axes[1,1].text(.03,.95,'How to read',fontsize=14,fontweight='bold',va='top'); axes[1,1].text(.03,.82,'0 = bearish, 0.5 = neutral, 1 = bullish\n\nEvidence scores translate raw features into PM-readable market states.\nWeights combine economic prior and trailing OOS reliability.\nThe final portfolio also reflects model ensemble and risk controls.',fontsize=11.5,va='top',linespacing=1.5,bbox=dict(boxstyle='round,pad=.7',facecolor='#F3F6FA',edgecolor='#CBD5E1'))
        fig.tight_layout(rect=[0,0,1,.94]); pdf.savefig(fig); plt.close(fig)

        # Page 3: model ensemble only
        fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.27)); fig.suptitle('Model Ensemble - Dynamic Voting Power', fontsize=18, fontweight='bold')
        for m in MODEL_COLS: axes[0,0].plot(model_ic.date, model_ic[m].rolling(6,min_periods=1).mean(), label=m)
        axes[0,0].axhline(0,color='k',lw=.7); axes[0,0].set_title('6-month average Rank IC'); axes[0,0].legend(fontsize=8); _style_axis(axes[0,0])
        for m in MODEL_COLS: axes[0,1].plot(model_weights.date, model_weights[m], label=m)
        axes[0,1].set_title('Dynamic model weight history'); axes[0,1].legend(fontsize=8); _style_axis(axes[0,1])
        avg = model_ic[MODEL_COLS].mean().sort_values(); axes[1,0].barh(avg.index, avg.values); axes[1,0].axvline(0,color='k',lw=.7); axes[1,0].set_title('Average test-period Rank IC'); _style_axis(axes[1,0])
        corr = pred[MODEL_COLS].corr(); im=axes[1,1].imshow(corr,vmin=-1,vmax=1,cmap='coolwarm'); axes[1,1].set_xticks(range(len(MODEL_COLS)),MODEL_COLS,rotation=45); axes[1,1].set_yticks(range(len(MODEL_COLS)),MODEL_COLS); axes[1,1].set_title('Prediction correlation / diversity'); fig.colorbar(im,ax=axes[1,1],fraction=.03)
        fig.tight_layout(rect=[0,0,1,.94]); pdf.savefig(fig); plt.close(fig)

        # Page 4: transformer only
        fig = plt.figure(figsize=(11.69, 8.27)); gs=fig.add_gridspec(2,2,height_ratios=[1,1.35]); fig.suptitle('Transformer Explanation = Time Band x Evidence State', fontsize=18, fontweight='bold')
        a1=fig.add_subplot(gs[0,0]); a1.bar(attention_bucket.period, attention_bucket.attention_share*100); a1.set_title('Temporal importance'); a1.set_ylabel('Share (%)'); a1.tick_params(axis='x',labelsize=8,rotation=15); _style_axis(a1)
        a2=fig.add_subplot(gs[0,1]); a2.bar(attention_bucket.period, attention_bucket.occlusion_impact); a2.set_title('Occlusion impact'); a2.set_ylabel('Prediction change'); a2.tick_params(axis='x',labelsize=8,rotation=15); _style_axis(a2)
        a3=fig.add_subplot(gs[1,:]); a3.axis('off')
        context = _context_table(daily, latest_date, attention_asset, attention_bucket) if daily is not None else [[r.period,r.attention_share,'N/A','N/A','N/A'] for r in attention_bucket.itertuples()]
        cells=[[r[0],f'{r[1]:.0%}',r[2],r[3],r[4]] for r in context]
        table=a3.table(cellText=cells,colLabels=['Time band','Importance','Momentum','Trend structure','Volatility'],cellLoc='center',loc='upper center',bbox=[.02,.28,.96,.68]); table.auto_set_font_size(False); table.set_fontsize(10.5)
        for (r,c),cell in table.get_celld().items():
            if r==0:
                cell.set_facecolor('#0F2747'); cell.get_text().set_color('white'); cell.get_text().set_weight('bold')
            elif c==1:
                cell.set_facecolor('#DCE9F7')
            elif c in [2,3,4]:
                txt=cell.get_text().get_text(); cell.set_facecolor('#C6F1DD' if txt in ['Strong','Bullish','Calm'] else '#F9D6D5' if txt in ['Weak','Bearish','Stress'] else '#EEF2F6')
        top=max(context,key=lambda x:x[1]); explanation=f'Example: {attention_asset}. The model focused most on {top[0]}. In that band, Momentum={top[2]}, Trend={top[3]}, Volatility={top[4]}. Attention is not treated as causal; occlusion impact is shown above as a robustness check.'
        a3.text(.02,.16,'\n'.join(textwrap.wrap(explanation,120)),fontsize=10.5,va='top',bbox=dict(boxstyle='round,pad=.5',facecolor='#F8FAFC',edgecolor='#CBD5E1'))
        fig.tight_layout(rect=[0,0,1,.94]); pdf.savefig(fig); plt.close(fig)

        # Page 5: flow/data requirements only
        fig, axes = plt.subplots(1,2,figsize=(11.69,8.27)); fig.suptitle('Implementation Flow & Data Requirements',fontsize=18,fontweight='bold')
        axes[0].axis('off'); flow='Excel / Bloomberg market inputs\n↓\nPoint-in-time alignment & futures roll controls\n↓\nDerived features\n↓\nPercentile / rank signals\n↓\nEvidence states\n↓\nRule / Ridge / RF / LightGBM / Temporal model\n↓\nOOS Rank IC / ICIR / Hit ratio\n↓\nDynamic ensemble\n↓\nRisk & turnover constraints\n↓\nSummary.xlsx + Model_Report.pdf'; axes[0].text(.5,.5,flow,ha='center',va='center',fontsize=12,linespacing=1.5,bbox=dict(boxstyle='round,pad=1',facecolor='#E8F0FE',edgecolor='#64748B'))
        axes[1].axis('off'); req='Required core fields\n- date, asset_id\n- PX_OPEN, PX_HIGH, PX_LOW, PX_LAST\n- VIX, DXY, US 2Y, US 10Y\n\nOptional / confidence-adjusted\n- PX_VOLUME, OPEN_INT\n- MOVE, HY spread\n- Oil, copper, gold\n- Local FX pairs\n\nControls\n- one-business-day lag for cross assets\n- adjusted continuous futures prices\n- roll-window penalties for volume/OI\n- monthly non-overlapping evaluation'; axes[1].text(.03,.95,req,va='top',fontsize=12,linespacing=1.45,bbox=dict(boxstyle='round,pad=.8',facecolor='#FFF7ED',edgecolor='#D97706'))
        fig.tight_layout(rect=[0,0,1,.94]); pdf.savefig(fig); plt.close(fig)
    return out
