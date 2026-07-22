# 個別銘柄スコアリングモデル v0.13.1

## 1. 今回の設計変更

v0.13.1では、第1層で行っていた単一FAのLinear・Piecewise・Quadratic回帰によるSubScore生成を、本番スコアリングから削除しました。

理由は、実証結果として以下が確認されたためです。

- S03のRaw Factor直接スコアが最も高いMean Rank ICを示した
- 第1層のValidation／OOS決定係数が継続的にマイナスだった
- 回帰予測値がゼロ付近へ圧縮され、S06以降で銘柄間の予測差が小さくなった
- 一方、Raw Factorを集約したValue・Momentum等のFactorScoreには分位単調性が確認された

新しい本線は次のとおりです。

```text
Raw Factor
  -> Winsorize
  -> Direction統一
  -> 国×セクター・サイズ中立化
  -> Centered Percentile Score (-1～1)
  -> 各FAのQ5-Q1 Factor Return作成
  -> 同一Factor Group内のFactor Return相関で重複調整
  -> Equal Weightへ縮小
  -> Aggregate FactorScore
  -> 国別OLS / Ridge
```

## 2. シナリオ

| ID | 内容 | 確認目的 |
|---|---|---|
| N00 | Raw Factor Scoreを全FA直接等ウェイト | 基準モデル |
| N01 | 階層表示＋FA数比例ウェイト | N00を階層形式で再現できるか |
| N02 | グループ内・グループ間等ウェイト | S04型のグループ等ウェイト効果 |
| N03 | Q5-Q1 Factor Return相関ウェイト | 相関調整のみの効果 |
| N04 | 相関ウェイト＋Equal Weight縮小 | Layer2本線候補 |
| N05 | 国別OLS・Factor主効果のみ | 最も単純な第3層 |
| N06 | 国別Ridge・Factor主効果のみ | 正則化の追加効果 |
| N07 | 国別Ridge・選択Sector×Factor交差項 | セクター別傾きの追加効果 |

N07でもセクターダミー主効果は既定で使用しません。セクターごとの定数項が予測を支配し、予測値が階段状になることを避けるためです。

## 3. インストールと実行

Windows PowerShellでプロジェクトルートへ移動し、次を実行します。

```powershell
py -m pip install -r requirements.txt
py -m pip install -e . --no-build-isolation
py scripts\validate_inputs.py --config config\model_config.py
py -m pytest -q
py scripts\run_pipeline.py --config config\model_config.py
```

## 4. 入力ファイル

### `data/input/factors_and_returns.xlsx`

`data`シートに以下を置きます。

```text
date
ISIN
stock_return
market_cap
currency
country
sector
FA0101, FA0102, ...
```

### `data/input/factor_master.xlsx`

全データシートは1行目がヘッダー、2行目以降がデータです。

主要シートは以下です。

- `Factor_Master`
- `Group_Settings`
- `Layer2_Settings`
- `Country_Region_Map`
- `Sector_Group_Map`
- `Sector_Factor_Interaction`
- `Layer3_Settings`

## 5. 主要出力

| ファイル | 内容 |
|---|---|
| `analysis_summary.xlsx` | 全シナリオの全期間・共通OOS比較 |
| `scenario_comparison.pdf` | Mean Rank IC、Q5-Q1、N00との差の可視化 |
| `quintile_cumulative_returns.pdf` | 各シナリオのQ1～Q5累積リターン |
| `factor_return_weight_diagnostics.xlsx/pdf` | FA別Q5-Q1、相関、Layer2ウェイト |
| `aggregate_factor_diagnostics.xlsx/pdf` | Value・Momentum等の単体予測力 |
| `layer3_model_diagnostics.xlsx/pdf` | 国別係数、R2、Alpha、予測値分散 |
| `country_factor_score_trends.xlsx/pdf` | 国別Aggregate FactorScore推移 |
| `model_parameter_summary.xlsx` | Layer2ウェイトとLayer3係数の統合一覧 |
| `stock_score_patterns/*.xlsx` | 各シナリオの最新銘柄別スコア |

旧仕様の以下の出力は削除しました。

- Layer1 Model Selection
- Layer1 SubScore履歴
- S06単一FA回帰係数
- Raw vs OOF SubScore比較
- 第1層回帰フィット診断PDF

## 6. 時点整合

Date=tのFAスコアはt+1リターンを予測します。

Date=tのLayer2ウェイト計算には、Formation Dateがt未満のFactor Returnだけを使用します。

```text
score at t-1 -> realized return at t -> available for weight at t
score at t   -> predicts return at t+1
```

したがって、Date=tの未実現Q5-Q1リターンはDate=tのウェイトに使用しません。

## 7. 推奨する確認順

1. N00とN01が一致するか
2. N02で精度が落ちるか
3. `factor_return_weight_diagnostics`で相関ウェイトが極端でないか
4. `aggregate_factor_diagnostics`でValue・Momentum等の単調性を確認
5. N04とN05を比較し、第3層回帰に追加価値があるか確認
6. N05とN06でOLSとRidgeを比較
7. N06とN07で交差項の追加価値を確認

複雑なモデルがN04を共通OOSで上回らない場合、N04を最終スコアとして採用する判断も合理的です。
## 8. 日本語PDFフォント

PDF出力前に、日本語グリフを持つフォントファイルを実体パスから検証・登録します。
Matplotlibの古いフォントキャッシュだけに依存しないため、タイトルや凡例が英数字だけになる問題を防ぎます。

通常はWindowsのYu GothicまたはMeiryo、LinuxのNoto Sans CJK JP、macOSのHiragino Sansを自動検出します。

実行前の確認は次で行えます。

```powershell
py scripts\check_pdf_font.py --config config\model_config.py
```

`outputs/japanese_font_check.pdf`を開き、日本語が表示されることを確認してください。

自動検出に失敗する場合は、`config/model_config.py`の次を実行PC上のフォントパスへ変更します。

```python
"reporting": {
    "japanese_font_path": r"C:\Windows\Fonts\YuGothM.ttc",
},
```

または環境変数`STOCK_SCORING_JAPANESE_FONT`へフォントの絶対パスを設定できます。
フォントが見つからない場合は文字化けしたPDFを作成せず、明確なエラーで停止します。
フォントファイル自体はプロジェクトに同梱しません。

## 9. 別PCでfactor_master.xlsxを作成する

Copilotへ貼り付ける詳細な指示書は次です。

```text
docs/factor_master_copilot_instructions.txt
```

この指示書には、全10シートの構造、列、初期値、ドロップダウン、検証処理、よくあるミスを記載しています。

