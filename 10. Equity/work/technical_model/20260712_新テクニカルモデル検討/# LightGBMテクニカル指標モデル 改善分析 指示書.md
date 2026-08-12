````markdown
# 共通VRPを用いた12カ国クロスセクショナルモデル 再設計・検証指示書

## 1. 背景と目的

現在、12カ国の株価指数を対象として、テクニカル指標を特徴量、将来リターンを目的変数とするLightGBMモデルを構築している。

現状の分析では以下が確認されている。

- 対象資産：12カ国の株価指数
- モデル：LightGBM
- 主な特徴量：テクニカル指標
- 現在の主目的変数：20営業日先リターン
- 個別特徴量のRank ICは総じて弱い
- LightGBMのRank ICも時点によって正負が不安定
- 20営業日より短期、特に1営業日付近に予測力が偏っている可能性がある
- 12資産のみのため、日次クロスセクショナルRank ICや五分位分析は不安定になりやすい
- IVデータを12カ国すべてについて取得することは困難
- そのためVRPについては、全12カ国に同一のグローバルVRP系列を使用している

重要な点として、時点 $t$ において全12カ国に同じVRPを付与している場合、

$$
VRP_{1,t}
=
VRP_{2,t}
=
\cdots
=
VRP_{12,t}
$$

となる。

したがって、VRP単体にはその時点におけるクロスセクショナルな差が存在しない。

そのため本分析では、VRPを

> 「各国を直接順位付けする特徴量」

として利用するのではなく、

> 「世界的なリスク環境・市場レジームを表す共通状態変数」

として再定義する。

さらに、

- Global VRP
- 各国固有のテクニカル指標
- 各国固有のRealized Volatility
- Global VRPとLocal特徴量のInteraction
- 各国のGlobal VRPへの感応度

を組み合わせることで、各国間のクロスセクショナル差を作成できるか検証する。

---

# 2. 今回検証する主要仮説

以下の仮説を順番に検証する。

## 仮説1

Global VRP単体にはクロスセクショナルな順位付け能力はないが、

**どの市場環境でテクニカルシグナルが有効になるか**

を説明するレジーム変数として利用できる。

---

## 仮説2

以下のInteractionを作ることで、同一VRP系列でも国別のクロスセクショナル差を作ることができる。

$$
GlobalVRP_t
\times
LocalFeature_{i,t}
$$

例：

$$
VRP_t \times RSI_{i,t}
$$

$$
VRP_t \times Momentum_{i,t}
$$

$$
VRP_t \times RV_{i,t}
$$

---

## 仮説3

IVが取得できない国については、Realized Volatilityを各国固有のリスク状態として利用できる。

Global VRPをGlobal情報、

Realized VolatilityをLocal情報

として組み合わせる。

---

## 仮説4

Global VRPへの感応度は国ごとに異なるため、

$$
\beta^{VRP}_{i,t}
$$

をRolling推定することでクロスセクショナルな特徴量を作成できる。

---

## 仮説5

現在の目的変数である絶対リターンより、

**各国の相対リターン**

を予測対象とした方が12カ国ランキングという最終目的に適している。

---

# 3. データ構造

データは原則として、

**1行 = 1日 × 1カ国**

とする。

最低限以下のカラムを用意する。

| カラム | 内容 |
|---|---|
| `date` | 基準日 |
| `country` | 国または指数コード |
| `price` | 株価指数 |
| `global_vrp` | 全12カ国共通のVRP |
| 各種テクニカル指標 | RSI、Momentum、MA乖離等 |
| `future_return_xd` | 各ホライズンの将来リターン |
| `prediction` | モデル予測値 |

---

# 4. Step 1：Global VRPの性質を確認

最初に、現在使用しているVRPについて時系列特性を確認する。

以下を算出・可視化する。

- VRP Level
- VRP Change
- VRP Rolling Mean
- VRP Rolling Standard Deviation
- VRP Rolling Z-score
- VRP Historical Percentile

## 4.1 VRP変化率または差分

```python
df["global_vrp_change"] = (
    df.groupby("country")["global_vrp"]
      .diff()
)
```

全カ国で同じVRPを使用している場合、実際には一つのユニークな時系列を抽出して計算してもよい。

---

## 4.2 Rolling Z-score

例えば252営業日を使用する。

$$
VRPZ_t
=
\frac{
VRP_t-\mu_{t,252}
}{
\sigma_{t,252}
}
$$

Python例：

```python
vrp_ts = (
    df[["date", "global_vrp"]]
    .drop_duplicates("date")
    .sort_values("date")
    .set_index("date")
)

window = 252

vrp_ts["vrp_mean_252"] = (
    vrp_ts["global_vrp"]
    .rolling(window)
    .mean()
)

vrp_ts["vrp_std_252"] = (
    vrp_ts["global_vrp"]
    .rolling(window)
    .std()
)

vrp_ts["global_vrp_z"] = (
    (
        vrp_ts["global_vrp"]
        - vrp_ts["vrp_mean_252"]
    )
    / vrp_ts["vrp_std_252"]
)
```

重要：

**将来データを使って全期間平均・標準偏差を計算しないこと。**

RollingまたはExpandingで過去情報のみを使用する。

---

# 5. Step 2：Global VRP Regimeを作成

Global VRPをクロスセクショナル特徴量としてではなく、市場レジームとして分類する。

まず以下の3分類を基本とする。

- Low VRP
- Normal VRP
- High VRP

例：

```python
def classify_vrp_regime(z):
    if pd.isna(z):
        return np.nan
    elif z <= -0.5:
        return "Low"
    elif z >= 0.5:
        return "High"
    else:
        return "Normal"

vrp_ts["vrp_regime"] = (
    vrp_ts["global_vrp_z"]
    .apply(classify_vrp_regime)
)
```

閾値は固定せず、

- ±0.5σ
- ±1.0σ
- 過去データの33% / 67% percentile

などについてRobustness Checkを行う。

---

# 6. Step 3：VRP Regime別に各特徴量のRank ICを計算

各時点について、

$$
IC_{j,t}
=
SpearmanCorr_i
(
X_{j,i,t},
R_{i,t+h}
)
$$

を計算する。

その後、ICをVRP Regime別に集計する。

目的は、

> VRPの水準によって各テクニカル指標の効き方が変化しているか

を検証すること。

例：

| VRP Regime | RSI IC | Momentum IC | RV IC |
|---|---:|---:|---:|
| Low | +0.03 | +0.05 | -0.01 |
| Normal | +0.00 | +0.01 | +0.00 |
| High | -0.06 | -0.04 | +0.05 |

このような結果が確認された場合、

> テクニカル特徴量の有効性がVRPレジームに依存する

と解釈する。

特に以下を確認する。

- ICの大きさ
- ICの符号
- Positive Ratio
- ICIR
- サンプル数

---

# 7. Step 4：各国固有のRealized Volatilityを作成

12カ国すべてについてIVを取得できないため、各国固有のリスク状態としてRealized Volatilityを作成する。

日次リターンを、

$$
r_{i,t}
=
\log
\left(
\frac{P_{i,t}}{P_{i,t-1}}
\right)
$$

とする。

Python例：

```python
import numpy as np

df = df.sort_values(
    ["country", "date"]
)

df["daily_return"] = (
    df.groupby("country")["price"]
    .transform(
        lambda x: np.log(x / x.shift(1))
    )
)
```

---

## 7.1 Local RV

以下を作成する。

- RV 5日
- RV 20日
- RV 60日

例：

```python
for h in [5, 20, 60]:

    df[f"rv_{h}d"] = (
        df.groupby("country")["daily_return"]
        .transform(
            lambda x:
            x.rolling(h).std()
            * np.sqrt(252)
        )
    )
```

---

# 8. Step 5：Relative RVを作成

単純な各国RVだけではなく、

> 世界全体と比較してその国のボラティリティが高いか

を測定する。

まず各日時点で12カ国の平均RVを作る。

$$
GlobalRV_t
=
\frac{1}{N}
\sum_i RV_{i,t}
$$

```python
df["global_rv_20d"] = (
    df.groupby("date")["rv_20d"]
    .transform("mean")
)
```

Relative RV：

$$
RelativeRV_{i,t}
=
\frac{
RV_{i,t}
}{
GlobalRV_t
}
$$

```python
df["relative_rv_20d"] = (
    df["rv_20d"]
    / df["global_rv_20d"]
)
```

または差分型も検討する。

$$
RelativeRVDiff_{i,t}
=
RV_{i,t}
-
GlobalRV_t
$$

---

# 9. Step 6：Local RV Shockを作成

各国自身の過去と比較して、現在のボラティリティが異常に高いかを測定する。

$$
RVShock_{i,t}
=
\frac{
RV_{i,t}
-
MA(RV_{i,t})
}{
SD(RV_{i,t})
}
$$

例：

```python
window = 252

rv_mean = (
    df.groupby("country")["rv_20d"]
    .transform(
        lambda x: x.rolling(window).mean()
    )
)

rv_std = (
    df.groupby("country")["rv_20d"]
    .transform(
        lambda x: x.rolling(window).std()
    )
)

df["rv_shock_20d"] = (
    (df["rv_20d"] - rv_mean)
    / rv_std
)
```

---

# 10. Step 7：Global VRP × Local特徴量のInteractionを作成

今回の分析で最も重要な部分の一つ。

Global VRPは同一時点では全12カ国で同じため、VRP単体にはクロスセクショナル分散がない。

一方、

$$
VRP_t
\times
X_{i,t}
$$

とすることで国ごとの差を作る。

以下を最低限作成する。

```python
interaction_features = [
    "RSI",
    "momentum_5d",
    "momentum_20d",
    "ma_deviation",
    "rv_5d",
    "rv_20d",
    "rv_60d",
    "relative_rv_20d",
    "rv_shock_20d",
]

for feature in interaction_features:

    df[f"vrp_x_{feature}"] = (
        df["global_vrp_z"]
        * df[feature]
    )
```

特に重要な候補：

$$
VRPZ_t\times RSI_{i,t}
$$

$$
VRPZ_t\times Momentum_{i,t}
$$

$$
VRPZ_t\times RV_{i,t}
$$

$$
VRPZ_t\times RelativeRV_{i,t}
$$

---

# 11. Step 8：特徴量をクロスセクショナル標準化する

12カ国の値を比較しやすくするため、Local特徴量について各日時点でCross-sectional RankまたはZ-scoreを作成する。

例えばRank：

```python
local_features = [
    "RSI",
    "momentum_5d",
    "momentum_20d",
    "ma_deviation",
    "rv_20d",
    "relative_rv_20d",
    "rv_shock_20d",
]

for feature in local_features:

    df[f"{feature}_rank"] = (
        df.groupby("date")[feature]
        .rank(
            pct=True,
            method="average"
        )
    )
```

Interactionについても、

$$
GlobalVRP_t
\times
Rank(X_{i,t})
$$

を検討する。

この方法は国ごとの絶対水準差の影響を抑える目的で使用する。

---

# 12. Step 9：各国のRolling VRP Betaを作成

Global VRPショックに対する各国株式の感応度を推定する。

概念的には、

$$
r_{i,t}
=
\alpha_i
+
\beta^{VRP}_{i,t}\Delta VRP_t
+
\epsilon_{i,t}
$$

をRolling Regressionする。

例えば252営業日のRolling Windowを使用する。

最終的な特徴量候補：

- `vrp_beta_126d`
- `vrp_beta_252d`

さらに、

$$
VRPExposure_{i,t}
=
\beta^{VRP}_{i,t}
\times VRPZ_t
$$

を作成する。

```text
vrp_beta
vrp_beta × global_vrp_z
```

をモデルに投入する。

重要：

Rolling Betaの推定には必ず時点 $t$ 以前のデータのみを使用する。

---

# 13. Step 10：目的変数を相対リターン化する

現在の絶対リターン、

$$
R_{i,t}^{h}
$$

だけでなく、クロスセクショナル相対リターンを作成する。

各時点の12カ国平均リターン：

$$
\bar R_t^h
=
\frac{1}{N}
\sum_i R_{i,t}^h
$$

相対リターン：

$$
RelativeReturn_{i,t}^{h}
=
R_{i,t}^{h}
-
\bar R_t^h
$$

Python例：

```python
for h in [1, 5, 10, 20, 40]:

    target = f"future_return_{h}d"

    cs_mean = (
        df.groupby("date")[target]
        .transform("mean")
    )

    df[f"relative_return_{h}d"] = (
        df[target] - cs_mean
    )
```

主要な比較対象：

```text
Absolute Return Target
vs
Cross-sectional Relative Return Target
```

---

# 14. Step 11：Cross-sectional Rank Targetも検証する

最終目的が12カ国を順位付けすることであるため、実現リターンの順位そのものもターゲット候補とする。

```python
for h in [1, 5, 10, 20, 40]:

    target = f"future_return_{h}d"

    df[f"target_rank_{h}d"] = (
        df.groupby("date")[target]
        .rank(
            pct=True,
            method="average"
        )
    )
```

以下の3種類を比較する。

1. Absolute Return
2. Relative Return
3. Cross-sectional Rank

---

# 15. Step 12：ホライズンを再検証する

現状の分析結果から20営業日より短期に予測力が存在する可能性があるため、以下を必ず比較する。

```text
1D
5D
10D
20D
40D
```

各ホライズンについて、

- 単体特徴量Rank IC
- VRP Interaction Rank IC
- Ridge Rank IC
- LightGBM Rank IC
- Long-Short Return

を計算する。

最終的に以下の表を作る。

| Model / Feature | 1D | 5D | 10D | 20D | 40D |
|---|---:|---:|---:|---:|---:|
| Technical only | | | | | |
| Technical + RV | | | | | |
| Technical + VRP | | | | | |
| Technical + VRP Interaction | | | | | |
| Full Model | | | | | |

---

# 16. Step 13：モデルセットを段階的に比較する

いきなり全特徴量を投入せず、以下のNested Modelを比較する。

## Model A：Technical Only

```text
RSI
Momentum
MA deviation
その他既存テクニカル
```

---

## Model B：Technical + Local RV

```text
Technical
+
RV 5D
RV 20D
RV 60D
Relative RV
RV Shock
```

---

## Model C：Technical + Global VRP

```text
Technical
+
Global VRP Level
Global VRP Z-score
Global VRP Change
VRP Regime
```

---

## Model D：Technical + VRP Interaction

```text
Technical
+
Global VRP
+
VRP × RSI
VRP × Momentum
VRP × RV
VRP × Relative RV
```

---

## Model E：Full Model

```text
Technical
+
Local RV
+
Relative RV
+
RV Shock
+
Global VRP
+
VRP Regime
+
VRP Interaction
+
Rolling VRP Beta
+
VRP Beta × VRP
```

---

# 17. Step 14：Ridgeを必ずベンチマークにする

12資産しかないため、複雑なLightGBMが必要とは限らない。

以下を必ず比較する。

1. Equal Weight
2. IC Weighted Score
3. Ridge
4. LightGBM

特にInteractionを明示的に作った場合、

**Ridgeでも十分な予測力が出るか**

を確認する。

もし、

```text
Ridge OOS Rank IC ≒ LightGBM OOS Rank IC
```

であれば、解釈性・安定性を優先してRidgeを採用する余地がある。

---

# 18. Step 15：LightGBMではInteractionを自動的に拾えるか確認

LightGBMは理論上、

```text
Global VRP
+
RSI
```

からVRP × RSIの非線形Interactionを学習可能である。

しかし今回、

- 資産数が12
- ICが弱い
- サンプルノイズが大きい

ため、Interactionを明示的に追加したモデルも比較する。

以下を比較する。

```text
LightGBM without explicit interaction
vs
LightGBM with explicit interaction
```

---

# 19. Step 16：VRP Regime別のモデル性能を評価する

モデル予測値について、

- Low VRP
- Normal VRP
- High VRP

それぞれでRank ICを計算する。

例：

| Model | Low VRP | Normal | High VRP |
|---|---:|---:|---:|
| Technical Only | +0.03 | 0.00 | -0.04 |
| + Local RV | +0.04 | +0.01 | +0.01 |
| + VRP Interaction | +0.05 | +0.02 | +0.07 |

このように、

> VRP Interactionを入れることでHigh VRP環境の性能が改善するか

を重点的に確認する。

---

# 20. Step 17：12資産という制約に合わせて評価方法を変更する

12資産しかない場合、Daily Rank ICは非常に不安定になりやすい。

そのためRank ICだけでモデルを判断しない。

最低限以下を併用する。

## 20.1 Mean Rank IC

従来通り計算。

---

## 20.2 Positive IC Ratio

$$
P(IC_t>0)
$$

---

## 20.3 Top 3 vs Bottom 3

予測上位3カ国と下位3カ国のリターン差を計算する。

$$
Spread_t
=
Mean(Return_{Top3})
-
Mean(Return_{Bottom3})
$$

---

## 20.4 Pairwise Ranking Accuracy

12カ国から2カ国を取り出したとき、

> モデルがどちらのリターンが高いか正しく予測した割合

を計算する。

12カ国の場合、1時点あたり、

$$
{12 \choose 2}
=
66
$$

ペア存在する。

Rank ICより細かな順位精度評価として使用する。

---

## 20.5 三分位分析

12資産で五分位にすると、

1グループあたり約2～3資産

しか存在しない。

そのため五分位だけでなく三分位を実施する。

```text
Top 4
Middle 4
Bottom 4
```

を基本候補とする。

---

# 21. Step 18：五分位分析と三分位分析を比較

以下を比較する。

```text
Quintile
vs
Tertile
vs
Top3 / Bottom3
```

確認する指標：

- 単調性
- Spread
- 勝率
- Turnover
- 年別安定性
- Transaction Cost控除後リターン

12資産の場合、最終判断では

**Top3 - Bottom3**

または

**Tertile**

を重視する。

---

# 22. Step 19：時系列Validationを実施

モデル学習は必ず時系列順に行う。

例：

```text
Train       : 過去年
Validation  : その次の期間
Test        : 完全Holdout
```

可能であればWalk-forward方式を使用する。

例：

```text
Train 1 → Test 1
Train 2 → Test 2
Train 3 → Test 3
...
```

20営業日ターゲットを使用する場合はPurging / Embargoを行う。

---

# 23. Step 20：VRP Interactionの追加価値を定量化する

以下のモデル比較を行う。

$$
\Delta IC
=
IC_{WithVRPInteraction}
-
IC_{WithoutVRPInteraction}
$$

同様に、

$$
\Delta Spread
=
Spread_{WithVRPInteraction}
-
Spread_{WithoutVRPInteraction}
$$

を計算する。

以下の表を作成する。

| 指標 | Base | + VRP | + Interaction | Improvement |
|---|---:|---:|---:|---:|
| OOS Rank IC | | | | |
| Positive Ratio | | | | |
| Top3-Bottom3 | | | | |
| Tertile Spread | | | | |
| Net Return | | | | |

VRPを追加したことで改善していない場合は、

> VRPは現在のモデルに追加価値を提供していない

と判断する。

無理に残さないこと。

---

# 24. Step 21：Rolling VRP Betaの追加価値を確認

以下を比較する。

```text
Global VRPのみ
vs
Global VRP + VRP Beta
vs
Global VRP + VRP Beta × VRP
```

特に、

$$
\beta^{VRP}_{i,t}\times VRP_t
$$

にクロスセクショナルな予測力が存在するかRank ICで確認する。

VRP Beta自体が不安定な場合は、

- 126D
- 252D
- 504D

などのWindowを比較する。

---

# 25. Step 22：Global VRPの利用に関する注意事項

今回使用しているVRPは、

**各国固有のVRPではない。**

したがって分析結果を説明する際には、

> Country VRP

とは呼ばず、

> Global VRP
> Global volatility risk premium proxy
> Global risk regime variable

などとして扱う。

また、

$$
GlobalVRP_t
$$

単体に各国のクロスセクショナル順位付け能力があるとは解釈しない。

---

# 26. 最終的に推奨する特徴量構成

## Local Technical

```text
RSI
Momentum 5D
Momentum 20D
MA deviation
その他既存テクニカル
```

## Local Risk

```text
RV 5D
RV 20D
RV 60D
Relative RV
RV Shock
```

## Global Regime

```text
Global VRP Level
Global VRP Change
Global VRP Z-score
Global VRP Percentile
Global VRP Regime
```

## Global × Local Interaction

```text
VRP × RSI
VRP × Momentum
VRP × MA deviation
VRP × RV
VRP × Relative RV
VRP × RV Shock
```

## Country-specific Global Risk Exposure

```text
Rolling VRP Beta
VRP Beta × Global VRP
```

---

# 27. 最終的なモデル設計候補

最終的には以下の構造を第一候補として検証する。

$$
\boxed{
LocalTechnical_{i,t}
+
LocalRisk_{i,t}
+
GlobalVRP_t
+
GlobalVRP_t
\times
LocalFeatures_{i,t}
+
VRPExposure_{i,t}
}
$$

目的変数は、

$$
\boxed{
RelativeReturn_{i,t+h}
}
$$

を第一候補とし、

Absolute ReturnおよびRank Targetと比較する。

---

# 28. 分析優先順位

以下の順番で実施する。

```text
1. Global VRPの時系列Z-score・Regime作成
        ↓
2. VRP Regime別の既存特徴量Rank IC
        ↓
3. Local RV作成
        ↓
4. Relative RV / RV Shock作成
        ↓
5. Global VRP × Local Feature作成
        ↓
6. Absolute Return → Relative Return比較
        ↓
7. 1D / 5D / 10D / 20D / 40D比較
        ↓
8. Technical Onlyモデル
        ↓
9. Technical + Local RV
        ↓
10. Technical + Global VRP
        ↓
11. Technical + VRP Interaction
        ↓
12. Rolling VRP Beta追加
        ↓
13. Ridge vs LightGBM
        ↓
14. VRP Regime別OOS性能
        ↓
15. Top3-Bottom3 / Tertile評価
        ↓
16. 完全Holdoutで最終検証
```

---

# 29. 最終成果物

最低限、以下を出力する。

## ① VRP Regime別特徴量Rank IC

| Feature | Low VRP | Normal VRP | High VRP |
|---|---:|---:|---:|

---

## ② ホライズン別モデルRank IC

| Model | 1D | 5D | 10D | 20D | 40D |
|---|---:|---:|---:|---:|---:|

---

## ③ Target比較

| Target | Ridge IC | LightGBM IC | Top3-Bottom3 |
|---|---:|---:|---:|
| Absolute Return | | | |
| Relative Return | | | |
| Rank Target | | | |

---

## ④ 特徴量セット別モデル比較

| Model | OOS IC | ICIR | Positive % | Top3-Bottom3 |
|---|---:|---:|---:|---:|
| Technical | | | | |
| + Local RV | | | | |
| + Global VRP | | | | |
| + VRP Interaction | | | | |
| + VRP Beta | | | | |

---

## ⑤ VRP Regime別モデル性能

| Model | Low | Normal | High |
|---|---:|---:|---:|

---

## ⑥ Portfolio評価

| Model | Tertile Spread | Top3-Bottom3 | Turnover | Net Return |
|---|---:|---:|---:|---:|

---

# 30. 最終レポートで回答すべき事項

以下について必ず明確な結論を出すこと。

1. Global VRPによってテクニカル指標の有効性が変化するか
2. VRPをRegime Variableとして利用する価値があるか
3. VRP × Local Featureに追加予測力があるか
4. Local RVを追加するとOOS性能が改善するか
5. Relative RVにクロスセクショナル予測力があるか
6. Rolling VRP Betaに追加予測力があるか
7. Absolute ReturnよりRelative Returnを目的変数にした方が良いか
8. 最適な予測ホライズンは1D / 5D / 10D / 20D / 40Dのどれか
9. RidgeとLightGBMのどちらがOOSで安定しているか
10. 12資産ではRank ICよりTop3-Bottom3等を重視すべきか
11. VRP追加によるIncremental Valueが存在するか
12. 最終モデルにVRPを残す合理性があるか

---

# 31. 判断ルール

## Case 1：VRP Regime別でICが明確に異なる

VRPをGlobal Regime Variableとして残す。

---

## Case 2：VRP単体は無効だがInteractionが有効

Global VRP単体ではなく、

```text
VRP × Local Feature
```

を中心に使用する。

---

## Case 3：Local RV追加で改善

IV不足を無理に補完せず、

```text
Global VRP + Local RV
```

という構造を採用する。

---

## Case 4：VRP Betaで改善

各国のGlobal VRP感応度を使用する。

---

## Case 5：Relative Return Targetで改善

12カ国ランキングモデルとしてRelative Returnを正式ターゲット候補とする。

---

## Case 6：すべて改善なし

Global VRPはモデルから除外する。

「理論的に使えそうだから」という理由だけで特徴量を残さない。

---

# 32. 最重要ポイント

今回、全12カ国について同じVRPを使用していること自体を直ちに問題とはしない。

ただし、そのVRPを

> 各国固有のボラティリティ・リスク・プレミアム

として扱ってはいけない。

Global VRPの役割は、

$$
\boxed{
Country\ Ranking\ Signal
}
$$

ではなく、

$$
\boxed{
Global\ Market\ State
}
$$

とする。

そのうえで、

$$
\boxed{
Global\ State
\times
Local\ Characteristics
}
$$

という設計によって国別のクロスセクショナル予測力を作る。

今回の最優先検証は、

> **VRPが高い局面・低い局面で、既存テクニカル特徴量のRank ICがどのように変化するか**

である。

ここで明確なレジーム差が確認された場合、Global VRPは直接的なAlpha Factorではなく、

> **Alpha Signalの有効性を切り替えるRegime Variable**

として機能している可能性が高い。

その結果を確認した後に、

```text
VRP × Technical
VRP × Local RV
VRP Beta × VRP
```

を順次追加し、必ずOut-of-SampleでIncremental Valueを検証すること。
````
