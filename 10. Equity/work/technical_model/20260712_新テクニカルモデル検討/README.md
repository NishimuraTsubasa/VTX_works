# Global Equity Futures Evidence Ensemble v2

## 1. 目的

本プロジェクトは、12の主要株価指数先物を対象として、以下を一体化した研究・実装用デモです。

1. 市場データをExcelで受け取る
2. 価格・ボラティリティ・需給・クロスアセット特徴量を計算する
3. 各特徴量を時系列Percentileまたは12市場内Rankへ変換する
4. 特徴量をPersistence、Correction、Volatility、Flow、Relative Strength、Intermarket、Macro MarketのEvidenceへ集約する
5. Rule、Ridge、Random Forest、LightGBM、Temporal Modelで予測する
6. 直近OOS Rank IC、ICIR、Hit Ratioに応じてモデルの発言力を調整する
7. 12市場のスコアをボラティリティ調整し、先物ポートフォリオへ変換する
8. 重要な数値結果をExcel、可視化結果をPDFへ出力する

データは合成ダミーデータです。表示される収益率や予測精度には投資上の意味はありません。

---

## 2. 出力物

### 2.1 Summary Excel

`outputs/summary/Summary.xlsx`

重要な結果はExcelに集約しています。

- `Executive_Summary`: KPI、最新モデルウェイト、上位Long/Short
- `Portfolio_Latest`: 最新の12市場ポートフォリオ
- `Model_Performance`: 平均Rank IC、ICIR、最新モデルウェイト
- `Model_Weights_History`: 動的モデルウェイト履歴
- `Model_RankIC_History`: モデル別Rank IC履歴
- `Evidence_Latest`: 最新Evidenceスコア
- `Evidence_Weights`: Evidenceウェイト履歴
- `Transformer_Attention`: 時間帯別AttentionとOcclusion Impact
- `Transformer_Context`: 時間帯 × Momentum / Trend / Volatility状態
- `Backtest_Monthly`: 月次バックテスト
- `Performance_Metrics`: 年率収益、ボラ、Sharpe、最大DD
- `Feature_Dictionary`: 使用特徴量、計算式、必要BBGフィールド
- `Input_Specification`: インプット列・型・必須条件
- `Config`: 実行条件

### 2.2 Model Outputs Excel

`outputs/data/Model_Outputs.xlsx`

モデルが生成した中間・最終結果をシート別に保存します。

- Predictions
- Model_Weights
- Model_RankIC
- Evidence_Weights
- Backtest
- Performance
- Latest_Portfolio
- Evidence_Latest
- Transformer_Attention
- Transformer_Patches
- Model_Performance
- Run_Manifest

### 2.3 PDF report

`outputs/report/Model_Report.pdf`

同種の可視化を同じページにまとめています。

1. Portfolio & Performance
2. Evidence State & Attribution
3. Model Ensemble / Dynamic Voting Power
4. Transformer: Time Band × Evidence State
5. Implementation Flow & Data Requirements

質問にあった「Transformerがどの局面を見ているか」は、PDFの第4ページで確認します。 同じ内容の数値表はSummary.xlsxの`Transformer_Attention`および`Transformer_Context`にも出力します。時間帯別Attentionだけでなく、同じ時間帯のMomentum、Trend Structure、Volatility状態も同時に表示します。またAttentionだけを因果的な説明とみなさず、その時間帯をNeutral化したOcclusion Impactも併記します。

---

## 3. フォルダ構造

```text
Global_Equity_Futures_Evidence_Ensemble_v2/
├─ README.md
├─ PROJECT_STRUCTURE.md
├─ requirements.txt
├─ run_pipeline.py
├─ run_all.sh
├─ config/
│  ├─ config.yaml
│  └─ evidence.yaml
├─ data/
│  ├─ input/
│  │  ├─ Universe_Master.xlsx
│  │  ├─ Cross_Asset_Data.xlsx
│  │  └─ market/
│  │     ├─ Market_SPX.xlsx
│  │     ├─ Market_NDX.xlsx
│  │     └─ ... 12市場
│  └─ processed/
│     ├─ Monthly_Model_Dataset.xlsx
│     └─ Daily_Signal_Sample.xlsx
├─ outputs/
│  ├─ data/
│  │  └─ Model_Outputs.xlsx
│  ├─ report/
│  │  └─ Model_Report.pdf
│  └─ summary/
│     └─ Summary.xlsx
├─ src/
│  ├─ data_generation.py
│  ├─ excel_io.py
│  ├─ features.py
│  ├─ models.py
│  ├─ ensemble.py
│  ├─ portfolio.py
│  ├─ reporting.py
│  └─ build_excel_summary.py
└─ tests/
   └─ smoke_test.py
```

CSVは配布物および通常の出力に使用しません。

---

## 4. 実行方法

```bash
pip install -r requirements.txt
python run_pipeline.py
python src/build_excel_summary.py
```

一括実行:

```bash
./run_all.sh
```

ダミー入力を再生成する場合:

```bash
./run_all.sh --rebuild
```

---

## 4.1 Bloomberg取得・事前処理（Version 3）

Bloomberg取得設定は `bbg_preprocess/config/BBG_Config.xlsx` で管理します。
通常運用では、Raw履歴をTicker単位でローカルキャッシュし、キャッシュ最終日以降だけを差分取得します。データ訂正やVolume/Open Interestの遅延更新を吸収するため、直近10営業日は重複取得して新しい値で上書きします。

```bash
# Mockデータで事前処理のみ確認
python bbg_preprocess/run_bbg_preprocess.py

# Bloomberg接続環境では、BBG_Config.xlsx の provider を xbbg へ変更
python bbg_preprocess/run_bbg_preprocess.py --config bbg_preprocess/config/BBG_Config.xlsx
```

主な出力は以下です。

- `data/input/Universe_Master.xlsx`
- `data/input/market/Market_<asset_id>.xlsx`
- `data/input/Cross_Asset_Data.xlsx`
- `bbg_preprocess/logs/BBG_Query_Log.xlsx`
- `bbg_preprocess/logs/BBG_Data_Quality.xlsx`

人が編集・確認する設定、モデル入力、監査ログはExcelで管理します。一方、毎回の大量再取得を防ぐ機械用Raw cacheは、速度と容量を考慮してParquetを推奨します。詳細は `bbg_preprocess/README_BBG_PREPROCESS.md` を参照してください。

---

## 5. Bloomberg実データへ置き換える場合

### 5.1 Universe_Master.xlsx

Sheet: `Universe`

必須列:

| Column | 内容 |
|---|---|
| asset_id | モデル内部ID。SPX、NDX、NKYなど |
| region | US、Europe、Japan、Asia ex Japan、Oceania |
| currency | 先物の建値通貨 |
| bbg_ticker | Bloomberg Genericまたは明示ロール先物Ticker |
| active | 実行対象かどうか |

### 5.2 Market_*.xlsx

配置場所: `data/input/market/`

1市場につき1ファイル、Sheet名は `Market_Data` とします。

必須列:

| Column | 必須 | Bloomberg Field | 内容 |
|---|---:|---|---|
| date | Yes | Date | 観測日 |
| asset_id | Yes | - | Universe_Masterと一致するID |
| region | Yes | - | 地域 |
| currency | Yes | - | 建値通貨 |
| px_open | Yes | PX_OPEN | 始値 |
| px_high | Yes | PX_HIGH | 高値 |
| px_low | Yes | PX_LOW | 安値 |
| px_last | Yes | PX_LAST | ロール調整済み連続終値 |
| volume | Optional | PX_VOLUME | 出来高 |
| open_interest | Optional | OPEN_INT | 建玉 |

価格系は必須です。VolumeとOpen Interestは欠損しても実行可能ですが、Flow EvidenceのConfidenceを下げる設計が必要です。

### 5.3 Cross_Asset_Data.xlsx

Sheet: `Cross_Asset_Data`

| Column | 必須 | Bloomberg例 | 使用目的 |
|---|---:|---|---|
| date | Yes | Date | 観測日 |
| vix | Yes | VIX Index | 株式ボラ・Risk Appetite |
| move | Optional | MOVE Index | 金利ボラ |
| dxy | Yes | DXY Curncy | 米ドル環境 |
| us2y | Yes | USGG2YR Index | 政策金利期待 |
| us10y | Yes | USGG10YR Index | 長期金利 |
| hy_spread | Optional | HY OAS Index | Credit Risk Appetite |
| oil | Optional | CL1 Comdty | 原油 |
| copper | Optional | HG1 Comdty | 景気敏感商品 |
| gold | Optional | GC1 Comdty | Safe Haven / Real Rate |
| usdjpy | Japan | USDJPY Curncy | 日本株FX Evidence |
| eurusd | Europe | EURUSD Curncy | 欧州株FX Evidence |
| gbpusd | FTSE | GBPUSD Curncy | 英国株FX Evidence |
| audusd | ASX | AUDUSD Curncy | 豪州株FX Evidence |
| usdcnh | HSI/HSCEI | USDCNH Curncy | 中国・香港FX Evidence |
| usdkrw | KOSPI | USDKRW Curncy | 韓国株FX Evidence |

クロスアセット系列は時差リーケージ回避のため、Version 1.0では一律1営業日Lagしてモデルへ入力します。

---

## 6. Futuresデータに求められる条件

1. `PX_LAST`はロールジャンプを補正した連続価格系列であること
2. 取引判断時点で利用可能だったデータだけを使うこと
3. クロスアセットは原則1営業日Lagすること
4. Volume / Open Interestはロール前後の歪みを管理すること
5. 休日により更新されていない価格を新しい観測値として扱わないこと
6. 20営業日先ターゲットを毎日評価する場合は重複リターンに注意すること
7. 本デモでは月末サンプルを使用し、非重複に近い評価とすること

---

## 7. 使用特徴量と計算式

詳細版は`Summary.xlsx`の`Feature_Dictionary`を参照してください。主要特徴量は以下です。

### 7.1 Price / Momentum

#### L日リターン

```text
ret_L(t) = P(t) / P(t-L) - 1
```

使用期間: 5、20、60、120、252営業日。

#### クロスセクションRank

```text
CSRank_i(t) = percentile rank of ret_i(t) among the 12 markets
```

12市場内で0から1へ正規化します。

### 7.2 Trend

#### 移動平均乖離

```text
MA_Gap_L(t) = P(t) / MA_L(P)(t) - 1
```

L = 20、60、120。

#### 移動平均傾き

```text
MA_Slope_60(t) = MA_60(P)(t) / MA_60(P)(t-20) - 1
```

#### Breakout距離

```text
Breakout_120(t) = P(t) / max(P(t-119:t)) - 1
```

### 7.3 Correction

#### Drawdown

```text
Drawdown_L(t) = P(t) / max(P(t-L+1:t)) - 1
```

#### RSI

```text
RSI_14 = 100 - 100 / (1 + AverageGain_14 / AverageLoss_14)
```

#### Bollinger位置

```text
BB_Position_20(t) = (P(t) - MA_20(P)(t)) / (2 * SD_20(P)(t))
```

### 7.4 Volatility

#### 実現ボラティリティ

```text
RV_L(t) = sqrt(252) * SD(log(P(t)/P(t-1)), L)
```

#### Volatility Ratio

```text
Vol_Ratio(t) = RV_20(t) / RV_60(t)
```

#### True Range

```text
TR(t) = max(H(t)-L(t), abs(H(t)-C(t-1)), abs(L(t)-C(t-1)))
```

#### ATR比率

```text
ATR_Pct_20(t) = MA_20(TR)(t) / P(t)
```

### 7.5 Flow

#### 出来高比率

```text
Volume_Ratio_20(t) = Volume(t) / MA_20(Volume)(t)
```

#### 建玉変化率

```text
OI_Change_20(t) = OI(t) / OI(t-20) - 1
```

ロール前後はConfidenceを下げます。

### 7.6 Relative Strength

#### Global相対リターン

```text
Relative_Global_60_i(t) = ret_60_i(t) - mean_j(ret_60_j(t))
```

#### Region相対リターン

```text
Relative_Region_60_i(t) = ret_60_i(t) - mean_{j in same region}(ret_60_j(t))
```

### 7.7 Rates

```text
US10Y_Change_20(t) = US10Y(t) - US10Y(t-20)
US2Y_Change_20(t)  = US2Y(t) - US2Y(t-20)
Curve(t)           = US10Y(t) - US2Y(t)
Curve_Change_20(t) = Curve(t) - Curve(t-20)
```

### 7.8 FX

```text
FX_Return_20(t) = FX(t) / FX(t-20) - 1
Local_FX_Support_i(t) = MarketSpecificSign_i * FX_Return_20(t)
```

符号は市場別です。例として日本株はUSDJPY上昇を支援的、米国株は過度なDXY上昇を逆風として扱います。

### 7.9 Commodity

```text
Oil_Return_20(t)    = Oil(t) / Oil(t-20) - 1
Copper_Return_20(t) = Copper(t) / Copper(t-20) - 1
Gold_Return_20(t)   = Gold(t) / Gold(t-20) - 1
```

商品価格は単独で方向を決めず、Rates、VIX、株式モメンタムと組み合わせます。

---

## 8. Signal化

Raw特徴量をそのままモデルへ入れず、原則として以下へ変換します。

### 時系列Percentile

```text
Pct_i(t) = rank of x_i(t) within its own trailing 756-business-day history
```

### クロスセクションRank

```text
CSRank_i(t) = rank of x_i(t) among the 12 markets at date t
```

PercentileやRankを使用する目的:

- 市場間の価格・ボラ・出来高スケール差を吸収する
- 外れ値への依存を抑える
- 「過去3年の上位何%か」とPMへ説明できる
- 12市場間の相対比較を可能にする

---

## 9. Evidence

Version 1.0では以下の7Evidenceを使用します。

| Evidence | 主な入力 |
|---|---|
| Persistence | 20/60/120日リターン、MA乖離、MA傾き、Breakout |
| Correction | 5/20日リターン、RSI、Bollinger位置、Drawdown |
| Volatility Support | RV20、Vol Ratio、ATR、VIX、MOVE |
| Flow | Volume Ratio、OI Change、価格方向 |
| Relative Strength | 20/60/120日CS Rank、Global/Region相対リターン |
| Intermarket | Global Equity、VIX、HY Spread、DXY、Copper |
| Macro Market | US金利、FX、Oil、Copper、Gold |

Evidence内ウェイトは最終固定値ではありません。

```text
Final Evidence Weight
= 75% Economic Prior
+ 25% trailing OOS reliability adjustment
```

---

## 10. モデルと動的アンサンブル

使用モデル:

- Rule
- Ridge
- Random Forest
- LightGBM
- Temporal Patch Attention model

各モデルが12市場の翌月リターン順位を予測します。

モデル品質:

```text
Quality_m(t)
= 50% Scaled trailing Rank IC
+ 30% Scaled ICIR
+ 20% Positive-IC Hit Ratio
```

最終モデルウェイト:

```text
ModelWeight_m(t)
= 70% Equal-weight prior
+ 30% Softmax(Quality_m(t))
```

直近で外しているモデルの寄与度は下がりますが、ノイズへ過剰反応しないようEqual Weightへ縮小します。

---

## 11. Transformerの説明

Temporal Modelは過去252営業日のSignal / Evidence系列を入力します。

表示するもの:

1. どの時間帯を重視したか
2. その時間帯のMomentum状態
3. その時間帯のTrend Structure状態
4. その時間帯のVolatility状態
5. その時間帯をNeutral化した際の予測変化

```text
Temporal explanation
= Time-band importance
× Evidence state in that band
× Occlusion impact
```

Attentionだけを因果的説明とは扱いません。

---

## 12. 本番化前に必要な追加検証

- Bloomberg Genericのロールルール確認
- 実際の取引可能時刻でのPoint-in-Time検証
- 先物倍率、証拠金、通貨換算
- Volume / OIの限月別連結
- Bid-Ask / Market Impactを含む取引コスト
- Walk-forward再学習
- TransformerをTFTまたはPatchTSTへ交換
- Seed / Fold間のAttention安定性
- 非重複ホライズンでのRank IC評価
- モデル・Evidenceウェイトの上下限制約

