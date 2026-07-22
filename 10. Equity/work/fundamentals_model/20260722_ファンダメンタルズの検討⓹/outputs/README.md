# 出力フォルダ

このフォルダのExcelは合成データによる出力サンプルです。
PDFはv0.13.1の日本語フォント設定で再生成し、日本語表示を確認しています。

実データでの実行前に、次を実行してください。

```powershell
py scripts\check_pdf_font.py --config config\model_config.py
```

`japanese_font_check.pdf`が正常なら、通常パイプラインを実行します。

```powershell
py scripts\run_pipeline.py --config config\model_config.py
```

主な出力:

- `analysis_summary.xlsx`
- `scenario_comparison.pdf`
- `quintile_cumulative_returns.pdf`
- `factor_return_weight_diagnostics.xlsx/pdf`
- `aggregate_factor_diagnostics.xlsx/pdf`
- `layer3_model_diagnostics.xlsx/pdf`
- `country_factor_score_trends.xlsx/pdf`
- `model_parameter_summary.xlsx`
- `stock_score_patterns/*.xlsx`
