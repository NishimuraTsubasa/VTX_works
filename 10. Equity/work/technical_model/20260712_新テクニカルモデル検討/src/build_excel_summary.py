from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_OUTPUTS = ROOT / 'outputs' / 'data' / 'Model_Outputs.xlsx'
SUMMARY = ROOT / 'outputs' / 'summary' / 'Summary.xlsx'


def main() -> None:
    book = pd.ExcelFile(MODEL_OUTPUTS)
    data = {name: pd.read_excel(MODEL_OUTPUTS, sheet_name=name) for name in book.sheet_names}

    preserved = {}
    if SUMMARY.exists():
        current = pd.ExcelFile(SUMMARY)
        for name in ['Feature_Dictionary', 'Input_Specification', 'Transformer_Context']:
            if name in current.sheet_names:
                preserved[name] = pd.read_excel(SUMMARY, sheet_name=name)

    portfolio = data['Latest_Portfolio'].sort_values('dynamic_weight', ascending=False)
    perf = data['Performance']
    model_perf = data['Model_Performance']
    evidence = data['Evidence_Latest']
    attention = data['Transformer_Attention']

    kpi = pd.DataFrame({
        'KPI': ['Latest date', 'Dynamic annual return', 'Dynamic annual volatility', 'Dynamic Sharpe', 'Dynamic max drawdown', 'Static Sharpe', 'Net exposure'],
        'Value': [
            portfolio['date'].iloc[0],
            perf.loc[perf.strategy == 'Dynamic Ensemble', 'annual_return'].iloc[0],
            perf.loc[perf.strategy == 'Dynamic Ensemble', 'annual_volatility'].iloc[0],
            perf.loc[perf.strategy == 'Dynamic Ensemble', 'sharpe'].iloc[0],
            perf.loc[perf.strategy == 'Dynamic Ensemble', 'max_drawdown'].iloc[0],
            perf.loc[perf.strategy == 'Static Ensemble', 'sharpe'].iloc[0],
            portfolio.dynamic_weight.sum(),
        ],
    })

    with pd.ExcelWriter(SUMMARY, engine='xlsxwriter', datetime_format='yyyy-mm-dd') as writer:
        wb = writer.book
        header = wb.add_format({'bold': True, 'font_color': 'white', 'bg_color': '#17365D', 'align': 'center'})
        pct = wb.add_format({'num_format': '0.0%;[Red](0.0%);-'})
        num = wb.add_format({'num_format': '0.000;[Red](0.000);-'})

        kpi.to_excel(writer, sheet_name='Executive_Summary', index=False, startrow=3)
        ws = writer.sheets['Executive_Summary']
        ws.merge_range('A1:F1', 'Global Equity Futures - Evidence Ensemble Summary', wb.add_format({'bold': True, 'font_color': 'white', 'bg_color': '#17365D', 'font_size': 18, 'align': 'center'}))
        ws.merge_range('A2:F2', 'Synthetic demo results - not for investment use', wb.add_format({'bold': True, 'bg_color': '#FFF2CC', 'font_color': '#9C6500', 'align': 'center'}))
        ws.set_row(3, 22, header)
        ws.set_column('A:A', 30); ws.set_column('B:B', 20)
        model_perf.to_excel(writer, sheet_name='Executive_Summary', index=False, startrow=3, startcol=3)
        ws.set_row(3, 22, header)
        ws.set_column('D:H', 20)
        portfolio.head(3).to_excel(writer, sheet_name='Executive_Summary', index=False, startrow=13, startcol=0)
        portfolio.tail(3).sort_values('dynamic_weight').to_excel(writer, sheet_name='Executive_Summary', index=False, startrow=13, startcol=8)

        mapping = {
            'Portfolio_Latest': 'Latest_Portfolio',
            'Model_Performance': 'Model_Performance',
            'Model_Weights_History': 'Model_Weights',
            'Model_RankIC_History': 'Model_RankIC',
            'Evidence_Latest': 'Evidence_Latest',
            'Evidence_Weights': 'Evidence_Weights',
            'Transformer_Attention': 'Transformer_Attention',
            'Transformer_Patches': 'Transformer_Patches',
            'Backtest_Monthly': 'Backtest',
            'Performance_Metrics': 'Performance',
            'Predictions': 'Predictions',
            'Config': 'Run_Manifest',
        }
        for out_sheet, source_sheet in mapping.items():
            frame = data[source_sheet]
            frame.to_excel(writer, sheet_name=out_sheet[:31], index=False)
            sh = writer.sheets[out_sheet[:31]]
            sh.freeze_panes(1, 0); sh.set_row(0, 22, header); sh.autofilter(0, 0, max(len(frame), 1), max(len(frame.columns)-1, 0))
            sh.set_column(0, min(len(frame.columns)-1, 25), 16)

        for name, frame in preserved.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sh = writer.sheets[name]; sh.freeze_panes(1, 0); sh.set_row(0, 22, header); sh.set_column(0, min(len(frame.columns)-1, 20), 20)

    print(f'Created {SUMMARY}')


if __name__ == '__main__':
    main()
