# 個別銘柄スコアリングモデル v0.12

## 1. 目的

グローバル個別銘柄ユニバースを対象に、現行の単純スコアから3層モデルまでを同じ評価条件で比較します。

1. **第1層**：グローバル単一ファクター回帰で、各FAコードのOOF `SubScore` を生成
2. **第2層**：SubScoreをValue・Momentum・Quality等へ集約し、`FactorScore` を生成
3. **第3層**：FactorScore、非線形基底、セクターグループ、交差項を使って翌月総リターンを予測

第3層の推定範囲は次から選択できます。

- `country_independent`：国別独立モデル
- `regional_pooling`：地域プールモデル
- `hierarchical_partial_pooling`：地域共通係数＋国別補正

## 2. 主要式

第1層では各ファクター `k` について、Linear・Piecewise・Quadraticを比較します。

```text
r(i,t+1) = alpha(k,t) + f(k,t)(z(i,k,t)) + error
SubScore(i,k,t) = f_hat(k,t)(z(i,k,t))
```

第2層では同一Factor_GroupのSubScoreを集約します。

```text
FactorScore(i,h,t) = sum_k w(k,h,t) * SubScore(i,k,t)
```

第3層では国別・地域別等の単位で、セクター別のファクター感応度を学習します。

```text
r(i,t+1)
 = month effect
 + FactorScore nonlinear bases
 + SectorGroup dummy
 + SectorGroup x FactorScore interactions
 + error
```

## 3. 実行方法

### 初回のみ

```powershell
py -m pip install -e .
```

### スコアリング全体

```powershell
py scripts\run_pipeline.py --config config\model_config.py
```

### Binscatterのみ

```powershell
py scripts\run_binscatter.py --config config\model_config.py
```

### 入力検証のみ

```powershell
py scripts\validate_inputs.py --config config\model_config.py
```

## 4. VS Codeデバッグ

`.vscode/launch.json` の次の設定から開始します。

- `Debug scoring pipeline`
- `Debug binscatter`
- `Validate inputs`

エントリーポイントは `scripts/run_pipeline.py` です。`pipeline.py` を直接実行しません。

## 5. 入力

```text
data/input/
├─ factors_and_returns.xlsx
└─ factor_master.xlsx
```

`factor_master.xlsx`には次の第3層設定シートを追加しています。

- `Country_Region_Map`
- `Sector_Group_Map`
- `Sector_Factor_Interaction`
- `Layer3_Settings`

詳細は `docs/input_generation_readme.md` と `docs/factor_master_reference.md` を参照してください。

## 6. 分析シナリオ

| ID | 内容 |
|---|---|
| S00 | 現行0-1順位・全FA直接等ウェイト |
| S01 | 欠損FAウェイト再調整 |
| S02 | Winsorize追加 |
| S03 | 国・セクター・サイズ中立化比較 |
| S04 | 階層等ウェイト |
| S05 | 相関調整IC |
| S06 | 第1層＋第2層 |
| S07 | 第3層最終モデル |

## 7. 主な出力

```text
outputs/
├─ analysis_summary.xlsx
├─ layer3_diagnostics.xlsx
├─ quintile_cumulative_returns.pdf
├─ stock_scoring_scenario_comparison.pdf
├─ layer3_scope_comparison.pdf
├─ layer3_country_diagnostics.pdf
├─ coefficient_stability.pdf
├─ sector_factor_interactions.pdf
├─ stock_score_patterns/
├─ history/
└─ diagnostics/
```

サンプル出力は同梱の合成データで生成したものであり、実データの投資結果を示すものではありません。

## v0.12.1 factor_master.xlsx 標準化

- `README` 以外の全構造化シートは、1行目がカラムヘッダーです。
- 2行目以降は実データのみです。
- データシート上部へタイトル・説明・空白行を追加しないでください。
- 別PCでCopilotに再作成させる場合は、`docs/factor_master_excel_creation_instructions_for_copilot.txt` を使用してください。
- Copilotへの短い依頼文は `docs/copilot_prompt_create_factor_master.txt` です。
