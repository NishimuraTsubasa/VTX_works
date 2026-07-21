# 個別銘柄スコアリングモデル v0.12.6（統合完全版）

## 1. この版について

v0.12.6は、v0.12.2以降に追加した機能を**一つの整合したプロジェクトへ統合した完全版**です。差分パッチを順番に適用する必要はありません。本フォルダだけを新しいプロジェクトルートとして使用してください。

統合済みの主な機能：

- S00〜S07のシナリオ比較
- 全シナリオの共通 `Date × ISIN` OOS評価
- S07のOLS・Ridge公平比較
- 第3層の国別・地域別・部分プーリング
- 第3層連続変数の学習窓内標準化
- 国別係数・実効セクター傾き・国別パフォーマンス診断
- S06・S07のモデルフィット、係数、R²、予測分布、Calibration診断
- 国別FactorScoreの時系列・最新値・相対水準の可視化
- SubScore・FactorScoreごとの予測力、5分位、Rank IC、除外分析
- 日本語PDFフォントの自動検出・埋め込み

同梱入力は動作確認用の合成データです。実運用データへ置き換えてください。

## 2. モデル構造

1. **第1層**：グローバル単一FAモデルからWalk-forward OOF `SubScore` を作成
2. **第2層**：SubScoreをValue、Momentum、Quality等へ集約し `FactorScore` を作成
3. **第3層**：国別・地域別・部分プーリングのいずれかで、FactorScore、セクターグループ、交差項から翌月総リターンを予測

第3層の推定範囲：

- `country_independent`
- `regional_pooling`
- `hierarchical_partial_pooling`

## 3. 分析シナリオ

| ID | 内容 |
|---|---|
| S00 | 現行0-1順位・全FA直接等ウェイト |
| S01 | 欠損FAウェイト再調整 |
| S02 | Winsorize追加 |
| S03 | 国×セクター・サイズ中立化後の直接等ウェイト |
| S04 | Factor Group階層等ウェイト |
| S05 | 相関調整ICウェイト |
| S06 | 第1層選択モデル＋第2層FactorScore |
| S07_OLS_Linear | 第3層・線形基底・OLS |
| S07_Ridge_Linear | 第3層・線形基底・Ridge |
| S07_Ridge_Flexible | 第3層・3基底Ridge（初期設定は無効） |

## 4. 2018年開始データ向け初期値

```text
第1層最低学習期間       18か月
第1層モデル選択期間      6か月
第1層最大学習窓         36か月
第3層最低学習期間       12か月（第1層OOF開始後）
第3層最大学習窓         36か月
Ridge alpha検証期間      6か月
```

S06・S07は最低学習期間が必要なため、S00〜S05よりOOS開始が遅くなります。採用判断には、各モデルの全利用可能期間ではなく、`analysis_summary.xlsx` の共通OOS指標を使用してください。

## 5. 実行方法

プロジェクトルートで実行します。

```powershell
py -m pip install -e .
py scripts\validate_inputs.py --config config\model_config.py
py scripts\run_pipeline.py --config config\model_config.py
```

Binscatterだけを再作成する場合：

```powershell
py scripts\run_binscatter.py --config config\model_config.py
```

テスト：

```powershell
py -m pytest -q
```

VS Codeでは `.vscode/launch.json` の `Debug scoring pipeline` を使用します。`src/stock_scoring_model/pipeline.py` を直接実行しません。

## 6. 入力

```text
data/input/
├─ factors_and_returns.xlsx
└─ factor_master.xlsx
```

`factor_master.xlsx` はREADME以外の全シートで、1行目がカラムヘッダー、2行目以降がデータです。

## 7. 主な出力

```text
outputs/
├─ analysis_summary.xlsx
├─ quintile_cumulative_returns.pdf
├─ stock_scoring_scenario_comparison.pdf
├─ layer3_diagnostics.xlsx
├─ layer3_scope_comparison.pdf
├─ layer3_country_diagnostics.pdf
├─ coefficient_stability.pdf
├─ sector_factor_interactions.pdf
├─ s07_ols_ridge_comparison.xlsx
├─ s07_ols_ridge_comparison.pdf
├─ s07_country_diagnostics.xlsx
├─ s07_country_diagnostics.pdf
├─ s06_s07_model_fit_diagnostics.xlsx
├─ s06_s07_model_fit_diagnostics.pdf
├─ country_factor_score_trends.xlsx
├─ country_factor_score_trends.pdf
├─ factor_score_performance_diagnostics.xlsx
├─ factor_score_performance_diagnostics.pdf
├─ stock_score_patterns/
├─ history/
└─ diagnostics/
```

### 共通OOS・シナリオ比較

`analysis_summary.xlsx` と `stock_scoring_scenario_comparison.pdf`：

- 共通OOS Mean Rank IC、ICIR、正符号率
- Q1〜Q5、Q5−Q1
- S03対比差、勝率

### S07係数・国別診断

`s07_country_diagnostics.xlsx / PDF`：

- 国別OLS・Ridgeパフォーマンス
- 国別ローリングRank IC
- 国別Q5−Q1累積リターン
- 最新係数、係数安定性、Ridge alpha
- 主効果＋交差項によるセクター内実効傾き

### S06・S07モデルフィット

`s06_s07_model_fit_diagnostics.xlsx / PDF`：

- S06単一FA係数・選択モデル・実効ウェイト
- S07国別・時点別係数
- Train / Validation / OOS R²、Adjusted R²、RMSE、MAE
- Pearson / Spearman、予測分布、実現分布、Calibration

### FactorScore予測力

`factor_score_performance_diagnostics.xlsx / PDF`：

- SubScore・FactorScore別Mean Rank IC
- 5分位・Q5−Q1・累積リターン
- Raw Input ScoreとOOF SubScore比較
- 国別・国×セクター内の予測力
- FactorScore間相関
- Leave-One-Group-Out増分分析

### 国別FactorScore推移

`country_factor_score_trends.xlsx / PDF`：

- 国別Value・Momentum・Quality等の推移
- 最新値、過去36か月Zスコア、国間Zスコア
- 各国で最も高かったFactorの頻度

## 8. 日本語PDF

実行PCから以下の日本語フォントを順に探索します。

- Noto Sans CJK JP / Noto Sans JP
- Yu Gothic
- Meiryo
- MS Gothic
- IPAexGothic
- Hiragino Sans

PDFフォントはTrueTypeとして埋め込みます。フォントファイル自体は本プロジェクトへ同梱しません。

## 9. 設定の優先順位

1. `factor_master.xlsx` の設定
2. `config/model_config.py`
3. コード内の安全な既定値

通常変更する項目は、可能な限りExcelまたはConfigで管理してください。

## 10. バージョン整合

- プロジェクト版：`0.12.6`
- Pythonパッケージ版：`0.12.6`
- Config・README・実行手順：`v0.12.6`
- 差分パッチ適用：不要
