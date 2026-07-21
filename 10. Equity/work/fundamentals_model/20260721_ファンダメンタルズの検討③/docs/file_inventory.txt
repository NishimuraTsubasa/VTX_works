# 個別銘柄スコアリングモデル v0.12 ファイル一覧

## 実行に必要な全体構成

```text
stock_scoring_model_v12/
├─ README.md
├─ pyproject.toml
├─ requirements.txt
├─ run_scoring.bat
├─ run_binscatter.bat
├─ .vscode/launch.json
├─ config/model_config.py
├─ data/
│  ├─ input/
│  │  ├─ factors_and_returns.xlsx
│  │  └─ factor_master.xlsx
│  └─ templates/
│     ├─ factors_and_returns_template.xlsx
│     └─ factor_master_template.xlsx
├─ docs/
├─ scripts/
├─ src/stock_scoring_model/
├─ tests/
└─ outputs/
```

## コード

| File | 役割 |
|---|---|
| pipeline.py | 全体オーケストレーション |
| layer1_single_factor.py | 単一FAのLinear/Piecewise/Quadratic |
| layer1_model_selection.py | OOS RankIC・1-SE選択 |
| layer1_oof.py | Walk-forward OOF SubScore |
| layer2_factor_aggregation.py | FactorScore集約 |
| layer2_ic_weighting.py | 相関調整ICウェイト |
| nonlinear_basis.py | 第3層の非線形基底 |
| sector_grouping.py | 国-地域、セクターグループ対応 |
| interaction_features.py | セクター×FactorScore等 |
| layer3_design_matrix.py | 第3層説明変数行列 |
| layer3_country_model.py | 国別独立モデル |
| layer3_regional_model.py | 地域プールモデル |
| layer3_partial_pooling.py | 部分プーリングモデル |
| layer3_pooled.py | Rolling pooled Ridge |
| layer3_cross_sectional.py | 月次断面係数平均 |
| layer3_scope_selector.py | 推定範囲切替 |
| regularization.py | 列別罰則付きRidge |
| scenarios.py | S00-S07構築 |
| evaluation.py | RankIC・5分位評価 |
| reporting.py | Excel/PDF出力 |
| binscatter.py | Time-averaged binscatter |

## 入力

`country_sector_features.xlsx` はv0.12の本線では使用しません。国リターン予測を独立に行わず、国別・地域別の個別銘柄回帰を行うためです。

## 出力

| File | 内容 |
|---|---|
| analysis_summary.xlsx | S00-S07比較、第1層選択、第2層ウェイト |
| layer3_diagnostics.xlsx | 推定範囲別評価・予測・係数 |
| quintile_cumulative_returns.pdf | 各シナリオQ1-Q5 |
| stock_scoring_scenario_comparison.pdf | S00-S07横比較 |
| layer3_scope_comparison.pdf | 国別・地域・部分プーリング比較 |
| layer3_country_diagnostics.pdf | 国別RankIC |
| coefficient_stability.pdf | 第3層主要係数推移 |
| sector_factor_interactions.pdf | セクター×FactorScore係数 |
