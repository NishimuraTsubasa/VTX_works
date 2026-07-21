# 個別銘柄スコアリングモデル v0.12.2 ファイル一覧

## 全体構成

```text
stock_scoring_model_v12_2/
├─ README.md
├─ REPLACE_AND_RUN.txt
├─ FILE_LIST.txt
├─ pyproject.toml
├─ requirements.txt
├─ .vscode/launch.json
├─ config/
│  └─ model_config.py
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

## 主要コード

| File | 役割 |
|---|---|
| `pipeline.py` | 全体オーケストレーション |
| `scenarios.py` | S00～S06およびS07推定方式別シナリオ構築 |
| `evaluation.py` | 全利用可能期間・厳密な共通OOS期間のRankIC、5分位評価 |
| `layer1_single_factor.py` | 単一FAのLinear/Piecewise/Quadratic |
| `layer1_model_selection.py` | 過去検証期間による第1層モデル選択 |
| `layer1_oof.py` | Walk-forward OOF SubScore |
| `layer2_factor_aggregation.py` | FactorScore集約 |
| `layer2_ic_weighting.py` | 相関調整ICウェイト |
| `layer3_design_matrix.py` | 第3層の線形・非線形基底、ダミー、交差項 |
| `layer3_pooled.py` | Rolling pooled OLS/Ridge |
| `layer3_cross_sectional.py` | 月次断面係数平均のOLS/Ridge |
| `layer3_country_model.py` | 国別独立モデル |
| `layer3_regional_model.py` | 地域プールモデル |
| `layer3_partial_pooling.py` | 地域＋国部分プーリング |
| `layer3_scope_selector.py` | 推定範囲切替 |
| `regularization.py` | OLS、Ridge、係数出力 |
| `reporting.py` | Excel・PDFレポート |
| `master.py` | factor_master.xlsx設定読込 |
| `binscatter.py` | Time-averaged binscatter |

## v0.12.2で追加・更新したテスト

| File | 確認内容 |
|---|---|
| `test_common_oos.py` | 全シナリオで共通するDate×ISIN積集合と再順位化 |
| `test_layer3_estimators.py` | OLS/Ridgeの予測生成、同一OOS条件 |

## 入力

| File | 内容 |
|---|---|
| `factors_and_returns.xlsx` | date、ISIN、国、セクター、時価総額、リターン、FA列 |
| `factor_master.xlsx` | FA分類、集約、派生FA、地域、セクターグループ、第3層設定 |
| `factor_master_template.xlsx` | 別PC作成・入力開始用テンプレート |

`country_sector_features.xlsx` は本線では使用しません。国指数リターンを独立に予測するのではなく、国別・地域別の個別銘柄モデルを推定します。

## 主要出力

| File | 内容 |
|---|---|
| `analysis_summary.xlsx` | 全期間・共通OOSのシナリオ比較、RankIC差、5分位、第1・第2層履歴 |
| `stock_scoring_scenario_comparison.pdf` | 共通OOSを主軸にした全シナリオ比較 |
| `s07_ols_ridge_comparison.xlsx` | S07 OLS/Ridgeの指標、月次RankIC、係数、alpha |
| `s07_ols_ridge_comparison.pdf` | S07 OLS/Ridgeの共通OOS比較 |
| `layer3_diagnostics.xlsx` | 推定範囲別の予測・係数・モデル履歴 |
| `layer3_scope_comparison.pdf` | 国別・地域・部分プーリング比較 |
| `layer3_country_diagnostics.pdf` | 国別RankIC診断 |
| `coefficient_stability.pdf` | 第3層係数の時系列安定性 |
| `sector_factor_interactions.pdf` | セクター×FactorScore係数 |
| `stock_score_patterns/*.xlsx` | 各シナリオの銘柄別スコア |
| `history/*.xlsx` | モデル選択・係数・予測履歴 |

## ドキュメント

| File | 内容 |
|---|---|
| `config_guide.md` | 2018年開始データ用の期間・S07設定 |
| `common_oos_and_s07_estimators.md` | 共通OOSとOLS/Ridgeの数式・比較方法 |
| `factor_master_excel_creation_instructions_for_copilot.txt` | 別PCでCopilotにExcelを作成させる指示書 |
| `input_generation_readme.md` | factors_and_returns.xlsx作成仕様 |
| `three_layer_model_spec.md` | 3層モデル全体仕様 |
| `binscatter_time_averaged_spec.md` | Binscatter仕様 |
