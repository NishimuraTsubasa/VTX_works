from pathlib import Path
import argparse
import pandas as pd
import yaml

from src.data_generation import generate_dummy_data
from src.features import build_daily_dataset, build_monthly_dataset
from src.models import train_and_predict
from src.ensemble import dynamic_ensemble, MODEL_COLS, EVIDENCE_COLS
from src.portfolio import build_portfolios, performance_metrics
from src.reporting import transformer_explain, build_pdf
from src.excel_io import write_model_outputs, write_single_sheet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true', help='Regenerate dummy Excel inputs before running')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    with open(root / 'config' / 'config.yaml', 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    if args.rebuild:
        print('[1/8] Generate dummy Excel input workbooks')
        generate_dummy_data(root)
    else:
        print('[1/8] Use existing Excel input workbooks')

    print('[2/8] Build daily derived features, percentile/rank signals and evidence')
    daily = build_daily_dataset(root)
    print('[3/8] Build monthly model dataset')
    monthly, signal_cols, evidence_cols = build_monthly_dataset(root, daily)

    print('[4/8] Train Rule/Ridge/RF/LightGBM/Temporal model')
    bundle = train_and_predict(root, daily, monthly, signal_cols, evidence_cols, cfg)
    print('[5/8] Dynamic model and evidence ensemble')
    pred, model_weights, model_ic, evidence_weights = dynamic_ensemble(bundle.predictions, cfg)
    print('[6/8] Portfolio and backtest')
    bt, latest_portfolio = build_portfolios(pred, cfg)
    metrics = performance_metrics(bt)

    latest_date = pred.date.max()
    attention_asset = cfg['report']['latest_attention_asset']
    patch_attention, attention_bucket = transformer_explain(bundle.transformer_model, bundle.transformer_meta, latest_date, attention_asset)
    model_perf = pd.DataFrame({
        'model': MODEL_COLS,
        'average_rank_ic': [model_ic[m].mean() for m in MODEL_COLS],
        'rank_ic_std': [model_ic[m].std() for m in MODEL_COLS],
        'icir': [model_ic[m].mean() / (model_ic[m].std() + 1e-6) for m in MODEL_COLS],
        'latest_weight': [model_weights.iloc[-1][m] for m in MODEL_COLS],
    })

    print('[7/8] Save all important/intermediate results to XLSX')
    manifest = {
        'latest_date': str(pd.Timestamp(latest_date).date()),
        'attention_asset': attention_asset,
        'dummy_data': True,
        'temporal_model': bundle.transformer_meta['implementation'],
        'note': 'All displayed performance is based on synthetic data and is not investable.',
    }
    write_model_outputs(root / 'outputs' / 'data' / 'Model_Outputs.xlsx', {
        'Predictions': pred,
        'Model_Weights': model_weights,
        'Model_RankIC': model_ic,
        'Evidence_Weights': evidence_weights,
        'Backtest': bt,
        'Performance': metrics,
        'Latest_Portfolio': latest_portfolio,
        'Evidence_Latest': pred[pred.date == latest_date][['date','asset_id'] + EVIDENCE_COLS],
        'Transformer_Attention': attention_bucket,
        'Transformer_Patches': patch_attention,
        'Model_Performance': model_perf,
    }, manifest)

    sample_days = int(cfg.get('outputs', {}).get('daily_signal_sample_days', 252))
    sample_dates = sorted(daily.date.unique())[-sample_days:]
    sample = daily[daily.date.isin(sample_dates)].copy()
    write_single_sheet(root / 'data' / 'processed' / 'Daily_Signal_Sample.xlsx', 'Daily_Signal_Sample', sample)

    print('[8/8] Build grouped multi-chart PDF')
    build_pdf(root, pred, model_weights, model_ic, evidence_weights, bt, latest_portfolio, metrics, attention_bucket, patch_attention, daily=daily, attention_asset=attention_asset)
    print('Completed. Run `python src/build_excel_summary.py` to refresh Summary.xlsx.')


if __name__ == '__main__':
    main()
