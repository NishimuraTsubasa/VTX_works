# 個別銘柄スコアリングモデル v0.12.6 ファイル一覧

本一覧は統合完全版の実ファイルから自動生成しています。差分パッチは不要です。

## ルート

```text
CHANGELOG.md
FILE_LIST.txt
README.md
REPLACE_AND_RUN.txt
VERSION
pyproject.toml
requirements.txt
run_binscatter.bat
run_scoring.bat
```

## 設定

```text
config/model_config.py
```

## 入力・テンプレート

```text
data/input/factor_master.xlsx
data/input/factors_and_returns.xlsx
data/templates/factor_master_template.xlsx
data/templates/factors_and_returns_template.xlsx
```

## ドキュメント

```text
docs/binscatter_time_averaged_spec.md
docs/common_oos_and_s07_estimators.md
docs/config_guide.md
docs/copilot_prompt_create_factor_master.txt
docs/factor_master_excel_creation_instructions.txt
docs/factor_master_excel_creation_instructions_for_copilot.md
docs/factor_master_excel_creation_instructions_for_copilot.txt
docs/factor_master_reference.md
docs/file_inventory.md
docs/file_inventory.txt
docs/input_generation_readme.md
docs/layer3_scope_model_spec.md
docs/model_logic_and_analysis_flow.md
docs/output_specification.md
docs/three_layer_model_spec.md
```

## 実行スクリプト

```text
scripts/generate_demo_data.py
scripts/run_binscatter.py
scripts/run_pipeline.py
scripts/validate_inputs.py
```

## モデルコード

```text
src/stock_scoring_model/__init__.py
src/stock_scoring_model/binscatter.py
src/stock_scoring_model/binscatter_cli.py
src/stock_scoring_model/binscatter_runner.py
src/stock_scoring_model/cli.py
src/stock_scoring_model/config_loader.py
src/stock_scoring_model/country_factor_score_reporting.py
src/stock_scoring_model/evaluation.py
src/stock_scoring_model/factor_score_performance_reporting.py
src/stock_scoring_model/feature_engineering.py
src/stock_scoring_model/interaction_features.py
src/stock_scoring_model/io.py
src/stock_scoring_model/layer1_model_selection.py
src/stock_scoring_model/layer1_oof.py
src/stock_scoring_model/layer1_single_factor.py
src/stock_scoring_model/layer2_factor_aggregation.py
src/stock_scoring_model/layer2_ic_weighting.py
src/stock_scoring_model/layer3_country_model.py
src/stock_scoring_model/layer3_country_reporting.py
src/stock_scoring_model/layer3_cross_sectional.py
src/stock_scoring_model/layer3_design_matrix.py
src/stock_scoring_model/layer3_partial_pooling.py
src/stock_scoring_model/layer3_pooled.py
src/stock_scoring_model/layer3_regional_model.py
src/stock_scoring_model/layer3_scope_selector.py
src/stock_scoring_model/master.py
src/stock_scoring_model/model_fit_reporting.py
src/stock_scoring_model/nonlinear_basis.py
src/stock_scoring_model/pipeline.py
src/stock_scoring_model/preprocessing.py
src/stock_scoring_model/regression_metrics.py
src/stock_scoring_model/regularization.py
src/stock_scoring_model/reporting.py
src/stock_scoring_model/scenario_registry.py
src/stock_scoring_model/scenarios.py
src/stock_scoring_model/sector_grouping.py
```

## テスト

```text
tests/test_binscatter.py
tests/test_common_oos.py
tests/test_country_factor_score_reporting.py
tests/test_factor_score_performance_reporting.py
tests/test_feature_engineering.py
tests/test_japanese_pdf_font.py
tests/test_layer1_fit_diagnostics.py
tests/test_layer1_single_factor.py
tests/test_layer2_aggregation.py
tests/test_layer3_country_reporting.py
tests/test_layer3_design_matrix.py
tests/test_layer3_estimators.py
tests/test_layer3_partial_pooling.py
tests/test_layer3_standardization.py
tests/test_model_fit_reporting.py
tests/test_no_lookahead.py
tests/test_outputs.py
tests/test_sector_interactions.py
```

## 出力フォルダ説明

```text
outputs/README.md
```

## 主要な追加診断モジュール

- `layer3_country_reporting.py`: 国別係数・パフォーマンス・実効セクター傾き
- `model_fit_reporting.py`: S06/S07係数、R²、誤差、分布、Calibration
- `country_factor_score_reporting.py`: 国別FactorScore推移
- `factor_score_performance_reporting.py`: SubScore・FactorScore予測力、分位、除外分析
- `regularization.py`: OLS/Ridgeおよび学習窓内標準化

## テスト

- 収録テストファイル数: 18
- 確認済み: `21 passed`
