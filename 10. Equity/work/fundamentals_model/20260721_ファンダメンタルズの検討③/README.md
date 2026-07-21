# 個別銘柄スコアリングモデル v0.12.2

## 1. 今回の更新目的

グローバル個別銘柄ユニバースについて、S00以降の分析パターンを**同一の共通OOS期間・同一の銘柄集合**で公平に比較します。また、S07については、同じ線形説明変数を用いた次の2方式を並列評価します。

- `S07_OLS_Linear`：無正則化の線形回帰（OLS）
- `S07_Ridge_Linear`：L2正則化付き線形回帰（Ridge）

必要な場合だけ、従来の3基底Ridgeも補助比較として有効化できます。

- `S07_Ridge_Flexible`：Linear・Piecewise・Quadratic基底を同時投入するRidge

同梱サンプルは2018年1月開始・48か月の合成データです。実データの投資成果を示すものではありません。

## 2. 3層モデル

1. **第1層**：グローバル単一ファクター回帰から各FAコードのWalk-forward OOF `SubScore` を生成
2. **第2層**：SubScoreをValue・Momentum・Quality等へ集約し、`FactorScore` を生成
3. **第3層**：国別・地域別等の単位で、FactorScore、セクターグループ、交差項から翌月総リターンを予測

第3層の推定範囲は次から選択できます。

- `country_independent`：国別独立モデル
- `regional_pooling`：地域プールモデル
- `hierarchical_partial_pooling`：地域共通係数＋国別補正

## 3. 2018年開始データへの初期設定

月次データが2018年1月から始まる前提で、評価期間を過度に短くせず、かつ純粋なOOS性を保つため、初期値を次のようにしています。

```text
第1層最低学習期間       18か月
第1層モデル選択期間      6か月
第1層最大学習窓         36か月
第3層最低学習期間       12か月（第1層OOF FactorScoreが利用可能になった後）
第3層最大学習窓         36か月
Ridge alpha検証期間      6か月（第3層学習窓の末尾）
```

データ欠損がなく2018年1月から月次入力がある場合、S07の初回予測は概ね2020年半ば以降になります。実際の開始日は、利用可能なFA・国別銘柄数・目的変数の時点整合に基づき自動決定されます。

## 4. 共通OOS評価

主比較では、全シナリオについて次をすべて揃えます。

- 同じ予測月
- 同じISIN集合
- 同じ翌月リターン
- 各シナリオで有効な予測値

全シナリオの `Date × ISIN` の積集合を作り、その共通集合上でスコア順位と5分位を再計算します。

```text
Common OOS sample
= intersection of valid Date × ISIN keys across all enabled scenarios
```

したがって、`CommonMeanRankIC` 等は、S03とS06・S07を同じ期間・同じ銘柄で比較した値です。各モデルが利用できる全期間の結果も補助情報として残します。

主な共通OOS指標：

- `CommonMeanRankIC`
- `CommonMedianRankIC`
- `CommonRankICIR`
- `CommonRankICPositiveRate`
- `CommonQ5MinusQ1AnnualizedReturn`
- `CommonQ5MinusQ1Sharpe`
- `CommonQuintileMonotonicity`
- S03対比のMean RankIC差・勝率

## 5. S07 OLSとRidgeの公平な比較

OLSとRidgeでは、次の条件を同一にしています。

- FactorScore
- 線形基底
- セクターグループダミー
- 選択したセクター×FactorScore交差項
- 学習窓
- 最低学習期間
- OOS開始条件
- 共通OOS評価集合

異なるのは正則化だけです。

### OLS

```text
minimize Σ(y - Xβ)^2
```

### Ridge

```text
minimize Σ(y - Xβ)^2 + α||β||²
```

Ridgeの `alpha` は、各予測時点の過去学習窓の末尾6か月を検証期間として選択し、その後、選択したalphaで過去学習・検証データを合わせて再推定します。未来情報は使用しません。

## 6. 分析シナリオ

| ID | 内容 |
|---|---|
| S00 | 現行0-1順位・全FA直接等ウェイト |
| S01 | 欠損FAウェイト再調整 |
| S02 | Winsorize追加 |
| S03 | 国・セクター・サイズ中立化比較 |
| S04 | 階層等ウェイト |
| S05 | 相関調整IC |
| S06 | 第1層の選択モデル＋第2層FactorScore |
| S07_OLS_Linear | 第3層・線形基底・OLS |
| S07_Ridge_Linear | 第3層・線形基底・Ridge |
| S07_Ridge_Flexible | 第3層・3基底Ridge（初期設定は無効） |

## 7. 実行方法

### 初回のみ

```powershell
py -m pip install -e .
```

### 入力検証

```powershell
py scripts\validate_inputs.py --config config\model_config.py
```

### スコアリング全体

```powershell
py scripts\run_pipeline.py --config config\model_config.py
```

### Binscatterのみ

```powershell
py scripts\run_binscatter.py --config config\model_config.py
```

## 8. VS Codeデバッグ

`.vscode/launch.json` の `Debug scoring pipeline` から開始します。エントリーポイントは次です。

```text
scripts/run_pipeline.py
```

`src/stock_scoring_model/pipeline.py` を直接実行しません。

## 9. 入力

```text
data/input/
├─ factors_and_returns.xlsx
└─ factor_master.xlsx
```

`factor_master.xlsx` の `Layer3_Settings` で、以下を変更できます。

- `Lookback_Periods`
- `Minimum_Train_Periods`
- `Ridge_Validation_Periods`
- `S07_OLS_Linear_Enabled`
- `S07_Ridge_Linear_Enabled`
- `S07_Ridge_Flexible_Enabled`

`README`以外の全構造化シートは、1行目がカラムヘッダー、2行目以降が実データです。

## 10. 主な出力

```text
outputs/
├─ analysis_summary.xlsx
├─ stock_scoring_scenario_comparison.pdf
├─ s07_ols_ridge_comparison.xlsx
├─ s07_ols_ridge_comparison.pdf
├─ layer3_diagnostics.xlsx
├─ layer3_scope_comparison.pdf
├─ layer3_country_diagnostics.pdf
├─ coefficient_stability.pdf
├─ sector_factor_interactions.pdf
├─ stock_score_patterns/
└─ history/
```

### analysis_summary.xlsx

- 全利用可能期間の評価
- 厳密な共通OOS期間の評価
- 共通OOSの月次RankIC
- S03対比RankIC差・勝率
- 5分位リターン
- 第1層モデル選択履歴
- 第2層ウェイト履歴

### s07_ols_ridge_comparison.xlsx / PDF

- OLSとRidgeの共通OOS Q5-Q1累積リターン
- 共通OOS RankIC
- S03対比
- 係数・alpha履歴
- 評価期間と観測数

## 11. 検証

```powershell
py -m pytest -q
```

同梱版では13件のテストを収録しています。主な追加テストは、共通 `Date × ISIN` 積集合と、S07 OLS/Ridgeの同一OOS開始条件です。
