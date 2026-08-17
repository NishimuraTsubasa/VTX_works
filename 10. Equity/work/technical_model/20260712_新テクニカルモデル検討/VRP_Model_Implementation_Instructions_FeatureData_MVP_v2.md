# VRP条件付き20営業日リターン予測モデル：実装指示書【Featureデータ作成済み前提・MVP版】

## 0. この指示書の前提

本分析では、**特徴量データはすでに作成済み**とする。

したがって、本指示書では以下は対象外とする。

- Raw price data の取得
- 各特徴量の算出ロジック
- RSI / Momentum / Volatility / Trend 等の生成
- 特徴量ファイルの構築
- 元データのETL

本分析の開始点は、以下のような「分析可能な日次Feature Data」が既に存在する状態とする。

### 入力データのイメージ

| date | country | feature_1 | feature_2 | ... | vrp | ret_fwd_20bd |
|---|---|---:|---:|---|---:|---:|
| 2020-01-02 | US | ... | ... | ... | ... | ... |
| 2020-01-02 | JP | ... | ... | ... | ... | ... |
| 2020-01-02 | DE | ... | ... | ... | ... | ... |

最低限必要な列：

```text
date
country / index_id
各特徴量
VRP
20営業日先リターンLabel
```

可能であれば以下も保持する。

```text
label_end_date
```

`label_end_date` が存在しない場合は、営業日カレンダーを用いて20BD先の日付を別途作成する。

---

# 1. 分析目的

日次特徴量データを学習データとして利用し、**月1回、各国・指数の20営業日先リターンを予測する**。

構造は以下。

```text
作成済み日次Feature Data
        ↓
20BD Feature Validity
        ↓
MAIN / LINEAR / QUADRATIC のRole判定
        ↓
半年または年次でRoleを固定
        ↓
月次で回帰モデル再学習
        ↓
月1回20BDリターン予測
        ↓
Monthly OOS評価
```

---

# 2. 予測対象

主目的変数は20営業日先リターンのみとする。

\[
y_{i,t}=R_{i,t\rightarrow t+20}
\]

相対リターンを既にLabelとして作成している場合は、そのLabelをそのまま利用する。

5BD / 10BD / 40BD等を保有している場合も、Feature Roleの判定には使用しない。

用途は参考出力のみ。

```text
Primary Target  : 20BD
Other Horizons : Diagnostic only
```

---

# 3. 分析頻度

```text
Input Data Frequency      : Daily
Target Horizon            : 20BD
Feature Validity Update   : Semiannual を第一候補
Model Refit               : Monthly
Prediction Frequency      : Monthly
```

Feature Validityの半年更新と、回帰係数の月次更新を分ける。

毎月Feature Roleを変更しない。

---

# 4. 日次20BD LabelのOverlapping

日次データでは、

\[
R_{t:t+20}
\]

と、

\[
R_{t+1:t+21}
\]

が大きく重複する。

このこと自体は問題とせず、**日次Labelをすべて学習に利用する**。

ただし、Feature ValidityでRank IC等の統計量を評価するときは、通常のiid標準誤差を使用しない。

最低限、

```text
Newey-West / HAC lag = 20
```

を使用する。

初期実装ではHAC lagを20で固定してよい。

30 / 40 lagやBlock Bootstrapは、MVP完成後のRobustnessとする。

---

# 5. Label Availability

月次Prediction Dateを \(T\) とする。

Trainに利用できる行は、

\[
label\_end\_date \le T
\]

を満たすものだけとする。

20BD Forward Returnがまだ確定していない直近日次データをTrainに入れてはいけない。

`label_end_date` がある場合は必ずこの列を使って判定する。

---

# 6. Feature Universe

既に特徴量データが完成しているため、新規のFeature Engineeringは行わない。

ただし、以下の最低限のHard Filterのみ実施してよい。

### 6.1 欠損

Coverageが極端に低い特徴量は除外。

初期値：

\[
Coverage \ge 80\%
\]

### 6.2 定数・ほぼ定数

クロスセクションでほぼ変化しない特徴量は除外。

### 6.3 明らかな重複

特徴量同士の相関が、

\[
|\rho|>0.95
\]

程度でほぼ同一の場合のみ整理を検討する。

### 6.4 単独Rank ICによる大量削減は禁止

Ridgeを使用するため、

> 単独Rank ICが弱いから除外

という事前削減は原則行わない。

目的は、Feature Validityで「使う / 使わない」を決めることではなく、**VRPに対してどのような形で使うかを決めること**である。

---

# 7. Cross-sectional Rank

特徴量がRaw値の場合、各dateでCross-sectional percentile rankへ変換する。

\[
Rank_{i,j,t}=PctRank(X_{i,j,t})
\]

Centering：

\[
X_{i,j,t}^{rank}=Rank_{i,j,t}-0.5
\]

すでにRank Featureを入力データとして保有している場合は再計算しない。

---

# 8. VRP Regime

VRPからLow / Normal / Highを作る。

第一候補：

```text
Low    : bottom 20%
Normal : 20% - 80%
High   : top 20%
```

Percentile thresholdは、その時点までの過去データだけを使って計算する。

未来期間を含む全期間percentileは禁止。

---

# 9. Continuous VRP

Continuous Interaction用にVRP z-scoreを作る。

\[
z_t
=
\frac{VRP_t-\mu_t}{\sigma_t}
\]

第一候補：

```text
rolling window = 252BD
shift = 1
```

未来データを使用しない。

---

# 10. Feature Validityの目的

Feature Validityでは、各特徴量について、

```text
MAIN
LINEAR
QUADRATIC
```

のRoleを決定する。

重要：

**MAIN = Rank ICが高い特徴量、ではない。**

MAINは、

> VRP Low / Normal / Highでも20BD Rank ICがほぼ変わらない特徴量

とする。

---

# 11. 20BD Rank IC

特徴量 \(j\)、date \(t\) について、

\[
IC_{j,t}
=
\rho_S(X_{i,j,t},y_{i,t}^{20})
\]

を計算する。

VRP Regime別平均：

\[
\mu_L
=
E(IC_t\mid Low)
\]

\[
\mu_N
=
E(IC_t\mid Normal)
\]

\[
\mu_H
=
E(IC_t\mid High)
\]

この3値でRoleを判定する。

---

# 12. MAIN判定

Regime Range：

\[
Range_j
=
\max(\mu_L,\mu_N,\mu_H)
-
\min(\mu_L,\mu_N,\mu_H)
\]

第一候補：

\[
Range_j\le0.02
\]

ならMAIN候補。

加えて可能なら、

\[
\mu_L-\mu_N
\]

\[
\mu_N-\mu_H
\]

\[
\mu_L-\mu_H
\]

が実務的許容範囲内か確認する。

MVPでは、まず `Range <= 0.02` を主ルールとしてよい。

HACによる信頼区間・Equivalence Testは出力するが、実装を過度に複雑化する場合は後付けでもよい。

---

# 13. LINEAR判定

MAINではなく、

\[
Range_j>0.02
\]

かつ、

\[
\mu_L<\mu_N<\mu_H
\]

または、

\[
\mu_L>\mu_N>\mu_H
\]

のように概ね単調ならLINEAR候補。

Low–High差：

\[
\Delta_j=\mu_H-\mu_L
\]

も保存する。

---

# 14. QUADRATIC判定

Curvature：

\[
C_j
=
\mu_L+\mu_H-2\mu_N
\]

を計算する。

第一候補：

\[
|C_j|\ge0.03
\]

かつ、HACを考慮したCurvatureの有意性が一定以上であればQUADRATIC候補。

初期候補：

```text
abs(t_curve) >= 2
```

特徴量が非常に多い場合はBenjamini-Hochberg FDRを適用する。

第一候補：

```text
q-value < 0.10
```

Sign Flipも保存する。

\[
SignFlip
=
I[
\min(\mu_L,\mu_N,\mu_H)<0<
\max(\mu_L,\mu_N,\mu_H)
]
\]

ただしSign Flipは必須条件にしない。

---

# 15. Role判定の簡易フロー

```text
Feature
   ↓
Regime Range <= 0.02 ?
   ├─ Yes → MAIN
   │
   └─ No
        ↓
   Low → Normal → High が単調？
        ├─ Yes
        │    ↓
        │ Curvatureが強い？
        │    ├─ No  → LINEAR
        │    └─ Yes → QUADRATIC候補
        │
        └─ No
             ↓
        Curvature有意？
             ├─ Yes → QUADRATIC
             └─ No  → MAINまたはLINEARの低次側
```

判定が曖昧な場合は、より単純なRoleに倒す。

---

# 16. Expanding Feature Validity

更新日 \(T_k\) までの利用可能な全過去データを用いる。

```text
FV_EXP
```

として保存する。

これは、

> 長期的なFeature構造

を表すモデルとする。

---

# 17. Rolling Feature Validity

更新日 \(T_k\) の直近5年を第一候補として用いる。

```text
FV_ROLL
rolling_years = 5
```

これは、

> 最近の市場構造

を表すモデルとする。

---

# 18. EXPとROLLを共通化しない

Primary Analysisでは、

```text
EXP Model
ROLL Model
```

を別々に分析する。

共通Featureのみを採用する必要はない。

EXP / ROLLでRoleが異なること自体を分析対象とする。

Common Modelは、必要になった場合のみ後からReferenceとして追加する。

MVPでは実装不要。

---

# 19. Feature Validity更新

半年更新例：

```text
2020/06 FV update
2020/12 FV update
2021/06 FV update
2021/12 FV update
...
```

各更新日でRoleを決め、そのRoleを次回更新日まで固定する。

毎月Roleを再計算しない。

---

# 20. 初期比較モデル

MVPではモデルを増やしすぎない。

以下の4モデルをPrimary Comparisonとする。

```text
M0 : Ridge Main Effect
MD : VRP Dummy Interaction
M1 : Continuous VRP Linear Interaction
M2 : Selective Quadratic VRP
```

補助としてOLSを1本出してもよい。

Elastic NetはPhase 2へ回す。

---

# 21. M0：Ridge Main Effect

VRP Interactionなし。

\[
y
=
\alpha
+
\sum_j\beta_jX_j
+
\epsilon
\]

全Eligible Featureの主効果をRidgeで推定する。

目的：

> VRPを使わない場合の基準性能。

---

# 22. OLS Benchmark

既存分析ですでに単純線形モデルを計算している場合、そのモデルは残す。

\[
y
=
\alpha+X\beta+\epsilon
\]

ただしFeature数が多くOLSが不安定な場合は、無理に本番候補とせずBenchmark扱いとする。

比較：

\[
M0_{Ridge}
\quad vs\quad
OLS
\]

---

# 23. MD：VRP Dummy Interaction

NormalをReferenceとする。

\[
D_L=I(Regime=Low)
\]

\[
D_H=I(Regime=High)
\]

モデル：

\[
y
=
\alpha
+
X\beta
+
D_LX_{S_{VRP}}\gamma_L
+
D_HX_{S_{VRP}}\gamma_H
+
\epsilon
\]

ここで、

\[
S_{VRP}
=
LINEAR\cup QUADRATIC
\]

とする。

重要：

VRP Dummy単体だけでは同一日の全国に同じ値が入るため、Cross-sectional Rankをほぼ変化させない。

そのため、比較モデルとしては、

```text
VRP Dummy
```

だけではなく、

```text
VRP Dummy × Feature
```

をPrimary Dummy Modelとする。

---

# 24. M1：Continuous VRP Linear Interaction

\[
y
=
\alpha
+
X\beta
+
zX_{S_1}\gamma
+
\epsilon
\]

ここで、

\[
S_1
=
LINEAR\cup QUADRATIC
\]

とする。

QUADRATIC Featureも一次項は持つ。

---

# 25. M2：Selective Quadratic VRP

\[
y
=
\alpha
+
X\beta
+
zX_{S_1}\gamma
+
z^2X_{S_2}\delta
+
\epsilon
\]

ここで、

\[
S_2
=
QUADRATIC
\]

のみ。

全特徴量に二次項を付けない。

---

# 26. RidgeをPrimaryとする

M0 / MD / M1 / M2は原則Ridgeで比較する。

同じEstimatorを使うことで、

> Feature構造の違い

を比較しやすくする。

Ridge alphaはTrain / Validation内で選択する。

初期Grid：

```python
RIDGE_ALPHA_GRID = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]
```

---

# 27. Elastic Netは初期実装では必須にしない

MVPではElastic Netを外してよい。

M2がM1を改善し、二次項の価値が確認された後に、

> 不要な二次係数を0にした方が良いか

を検証するPhase 2として追加する。

したがって、初期分析で実装する必要はない。

---

# 28. 月次Walk-Forward

各月のPrediction Date \(T_m\) で以下を実行する。

```text
1. 最新のFeature Roleを取得
2. label_end_date <= train_cutoff の行だけ取得
3. Train Windowを作成
4. Validationを末尾から分割
5. StandardScalerをTrainだけでfit
6. Ridge alpha選択
7. M0 / MD / M1 / M2をfit
8. 当月断面を予測
9. predictionを保存
10. 20BD後にactualと比較
```

---

# 29. EXP ModelのTrain

MVPでは単純化のため、

```text
FV = EXP
Fit = EXP
```

とする。

---

# 30. ROLL ModelのTrain

MVPでは、

```text
FV = ROLL
Fit = ROLL
```

とする。

第一候補：

```text
rolling_years = 5
```

これにより初期実装では、

```text
EXP Structural Model
ROLL Adaptive Model
```

の2本だけを比較する。

`FV_EXP × Fit_ROLL` 等の2×2分析は後から追加する。

---

# 31. Validation

Random K-Foldは禁止。

Train末尾の時系列期間をValidationとして使用する。

例：

```text
Train -------------------- Validation ---- Test Month
```

20BD Label availabilityを考慮する。

---

# 32. StandardScaler

ScalerはTrainのみでfit。

```python
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_valid = scaler.transform(X_valid)
X_test  = scaler.transform(X_test)
```

---

# 33. Monthly Primary Evaluation

実際のPrediction Frequencyが月1回なので、Primary OOSは月次のみ。

各Prediction Dateで、

\[
IC_t
=
\rho_S(\hat y_{i,t},y_{i,t})
\]

を計算する。

---

# 34. 必須評価指標

MVPでは以下で十分。

```text
Mean Monthly Rank IC
Median Monthly Rank IC
ICIR
Positive IC Ratio
VRP Regime別 Rank IC
Top3 - Bottom3 Return
モデル別Rank IC時系列
```

可能なら、

```text
Transaction Cost控除後
Turnover
```

も追加する。

---

# 35. モデル比較

最重要比較：

### VRP効果があるか

\[
MD-M0
\]

### Continuous VRPがDummyより良いか

\[
M1-MD
\]

### 二次項に追加価値があるか

\[
M2-M1
\]

### EXP / ROLL差

```text
M0_EXP vs M0_ROLL
MD_EXP vs MD_ROLL
M1_EXP vs M1_ROLL
M2_EXP vs M2_ROLL
```

---

# 36. M2採用条件

M2を最初から本命にしない。

以下が確認できた場合のみM2を採用候補とする。

1. Mean Monthly Rank ICがM1より改善
2. ICIRが悪化しない
3. Regime別でも改善理由が説明可能
4. 特定数か月だけに依存しない
5. Top-Bottomでも改善
6. EXPまたはROLLの少なくとも一方で安定

M2が改善しなければM1を採用する。

---

# 37. Feature Validity出力

最低限以下をCSVまたはExcelで保存する。

```text
feature
fv_update_date
fv_type
ic20_all
ic20_low
ic20_normal
ic20_high
regime_range
linear_diff
linear_t_hac
curvature
curvature_t_hac
curvature_q
sign_flip
role
```

`fv_type`：

```text
EXP
ROLL
```

---

# 38. Prediction出力

```text
prediction_date
country
actual_20bd
pred_M0
pred_MD
pred_M1
pred_M2
rank_M0
rank_MD
rank_M1
rank_M2
vrp
vrp_z
vrp_regime
fv_type
```

---

# 39. Summary出力

推奨Sheet：

```text
Model_Summary
Monthly_RankIC
RankIC_by_Regime
EXP_vs_ROLL
M0_vs_MD
MD_vs_M1
M1_vs_M2
TopBottom
FeatureValidity_EXP
FeatureValidity_ROLL
FeatureRole_History
```

MVPではこれ以上増やさなくてよい。

---

# 40. 実装ファイル構成

特徴量データが既にあるため、多数の `.py` ファイルは不要。

推奨は2～3ファイル。

```text
project/
├── config.py
├── model_analysis.py
└── run_analysis.py
```

既存の1200行程度の分析 `.py` が動いている場合、無理に分割する必要はない。

---

# 41. config.py

パラメータのみ。

例：

```python
TARGET_HORIZON = 20

FV_UPDATE_FREQ = "6M"

FV_ROLLING_YEARS = 5
FIT_ROLLING_YEARS = 5

VRP_Z_WINDOW = 252
VRP_LOW_Q = 0.20
VRP_HIGH_Q = 0.80

MAIN_RANGE_THRESHOLD = 0.02
CURVATURE_THRESHOLD = 0.03
CURVATURE_T_THRESHOLD = 2.0

HAC_LAG = 20
FDR_Q = 0.10

RIDGE_ALPHA_GRID = [
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
]
```

---

# 42. model_analysis.py

既存Feature Dataを受け取って以下を実行する。

```python
load_feature_data()

prepare_vrp_state()

calc_rank_ic()

run_feature_validity_exp()
run_feature_validity_roll()

classify_feature_role()

build_m0_matrix()
build_dummy_matrix()
build_m1_matrix()
build_m2_matrix()

fit_ridge()

run_monthly_walk_forward()

evaluate_monthly_predictions()

save_results()
```

既存コードが1ファイルで管理しやすい場合、この処理を既存 `.py` に関数追加してもよい。

---

# 43. run_analysis.py

処理順を記載するだけにする。

```python
df = load_feature_data(PATH)

fv_exp = run_feature_validity_exp(df)
fv_roll = run_feature_validity_roll(df)

result_exp = run_monthly_walk_forward(
    df=df,
    feature_roles=fv_exp,
    mode="expanding",
)

result_roll = run_monthly_walk_forward(
    df=df,
    feature_roles=fv_roll,
    mode="rolling",
)

save_results(
    fv_exp=fv_exp,
    fv_roll=fv_roll,
    result_exp=result_exp,
    result_roll=result_roll,
)
```

---

# 44. 既存1200行コードを利用する場合

最初から全面リファクタリングしない。

まず既存コードを以下のセクションに整理するだけでよい。

```text
SECTION 1 : Input / Config
SECTION 2 : VRP State
SECTION 3 : Feature Validity
SECTION 4 : Model Matrix
SECTION 5 : Monthly Walk-Forward
SECTION 6 : Evaluation
SECTION 7 : Output
```

必要に応じて関数化する。

目的は「ファイルを細かく分けること」ではなく、

> 変更箇所と分析ロジックが追える状態にすること

である。

---

# 45. 初期実装でやらないこと

MVP段階では以下は実装不要。

```text
Elastic Net
Full Quadratic
Full Dummy Interaction
Common Model
FV × Fit の4通りすべて
Block Bootstrap
20 offset non-overlap
Rolling 3Y / 7Y
VRP 25/75, 30/70
複雑なHysteresis
Group Lasso
Hierarchical Lasso
```

まず中心仮説を検証する。

---

# 46. Phase 2に回す項目

MVPでM2に価値が確認できた後のみ追加。

```text
Elastic Net
Full Quadratic
Selection Frequency
Common Model
FV_EXP × Fit_ROLL
FV_ROLL × Fit_EXP
Non-overlap Robustness
Block Bootstrap
Feature Validity Annual vs Semiannual
```

---

# 47. 実装上の禁止事項

以下は禁止。

1. Feature Data自体を再生成し始める。
2. Feature Roleを毎月変更する。
3. 20BD以外のReturnでRoleを決める。
4. Future VRP percentileを使用する。
5. Future LabelをTrainに入れる。
6. Testを含めてScalerをfitする。
7. Random K-Fold。
8. OOS結果を見てRole thresholdをその場で変更する。
9. 全Featureに無条件で二次項を付ける。
10. M2が複雑だからという理由だけで採用する。
11. Daily Rank ICをMonthly Production OOSと混同する。
12. 既存コードを無理に多数のファイルへ分割する。

---

# 48. 最終的な実験フロー

```text
【作成済みFeature Data】
            ↓
        Data Check
            ↓
     VRP z / Regime
            ↓
        20BD Rank IC
            ↓
   ┌────────────────┐
   │ FV_EXP         │
   │ FV_ROLL        │
   └────────────────┘
            ↓
 MAIN / LINEAR / QUADRATIC
            ↓
       Roleを半年固定
            ↓
 ┌───────────────────────────┐
 │ M0 : Ridge               │
 │ MD : VRP Dummy × Feature │
 │ M1 : z × Feature         │
 │ M2 : z² × Feature        │
 └───────────────────────────┘
            ↓
      月次Walk-Forward
            ↓
      Monthly 20BD OOS
            ↓
 Rank IC / ICIR / Regime / Top-Bottom
            ↓
       モデル比較
```

---

# 49. 最終判断の順番

最初に、

\[
M0
\]

を基準とする。

次に、

\[
MD
\]

がM0を改善するか確認。

改善すれば、

> VRPレジームによるFeature slope変化に価値がある。

次に、

\[
M1
\]

がMDを改善するか確認。

改善すれば、

> DummyよりContinuous VRPの方が有効。

最後に、

\[
M2
\]

がM1を改善するか確認。

改善すれば、

> 非単調なVRP state dependenceに追加価値がある。

---

# 50. 本分析の最終思想

本分析では、既に作成済みのFeature Dataを前提として、

\[
\boxed{
Feature\ Data
\rightarrow
Feature\ Validity
\rightarrow
Role
\rightarrow
Ridge
\rightarrow
Monthly\ OOS
}
\]

だけに集中する。

Role：

\[
\boxed{
MAIN
=
VRPに対して20BD Rank ICが安定
}
\]

\[
\boxed{
LINEAR
=
VRPに対して20BD Rank ICが単調変化
}
\]

\[
\boxed{
QUADRATIC
=
VRPに対して20BD Rank ICが非単調変化
}
\]

比較モデル：

\[
\boxed{
M0
\rightarrow
VRP\ Dummy
\rightarrow
Continuous\ Linear
\rightarrow
Selective\ Quadratic
}
\]

ExpandingとRollingは別モデルとして分析する。

初期実装ではRidgeを中心とし、Elastic Net等は後から追加する。

最も重要なのは、複雑な実験環境を先に作ることではなく、

> **VRPによるFeature効果の変化を取り込むことで、月次20BD OOS予測が本当に改善するか**

をシンプルかつリークなく確認することである。

---

# 51. 半年・年次レビュー時の「モデル変更妥当性」判定

Feature Validityを半年または年1回レビューする場合でも、

\[
\boxed{
Review\ Date = Model\ Change\ Date
}
\]

とはしない。

レビューはあくまで、

> 「現行モデルを維持するか、Feature Roleを変更するか、より複雑 / 単純なモデルへ変更するか」

を定量的に判定する日とする。

したがって、

```text
半年ごとに必ずモデル変更
```

ではなく、

```text
半年ごとに定量レビュー
        ↓
変更条件を満たす場合のみ変更
        ↓
満たさなければ現行モデル維持
```

とする。

このルールを事前固定しておくことで、

> 「なぜこのタイミングでモデルを変えたのか」

を後から定量的に説明できる状態にする。

---

# 52. モデル変更は3階層に分けて管理する

モデル変更を以下の3種類に分ける。

## Level 1：Feature Role変更

例：

```text
MAIN → LINEAR
LINEAR → QUADRATIC
QUADRATIC → LINEAR
LINEAR → MAIN
```

## Level 2：Feature Validity Windowの更新

例：

```text
FV_EXP のRole更新
FV_ROLL のRole更新
```

EXP / ROLL自体は別モデルとして維持する。

## Level 3：採用モデルFamilyの変更

例：

```text
M0 → MD
MD → M1
M1 → M2
M2 → M1
```

Level 1とLevel 3を混同しない。

---

# 53. Feature Role変更の定量条件

Feature Roleは各更新日に再判定するが、閾値境界付近のノイズだけでRoleを変更しない。

Role変更には、

1. 新Roleの条件を満たすこと
2. 現Roleの条件から十分離れていること
3. 必要に応じて連続レビューで再現すること

を要求する。

---

# 54. MAIN → LINEAR の格上げ

MAIN Feature \(j\) について、

\[
Range_j
=
\max(\mu_L,\mu_N,\mu_H)
-
\min(\mu_L,\mu_N,\mu_H)
\]

とする。

MAIN判定基準が、

\[
\delta_M=0.02
\]

の場合、Role変更用に少し厳しいEntry Thresholdを設定する。

例：

\[
Range_j>\delta_M^{enter}
\]

\[
\delta_M^{enter}=0.025
\]

かつ、

\[
\mu_L<\mu_N<\mu_H
\]

または、

\[
\mu_L>\mu_N>\mu_H
\]

を満たす。

さらにLow–High差について、

\[
q_j^{VRP}<0.10
\]

を補助条件とする。

初期仕様例：

```text
MAIN → LINEAR

Range > 0.025
AND
Monotonic = True
AND
VRP difference q-value < 0.10
```

---

# 55. LINEAR → MAIN の格下げ

格下げ側は別閾値を使い、Hysteresisを持たせる。

例：

\[
Range_j<\delta_M^{exit}
\]

\[
\delta_M^{exit}=0.015
\]

かつEquivalenceが成立する場合、

```text
LINEAR → MAIN
```

候補とする。

つまり、

\[
\delta_M^{exit}
<
\delta_M
<
\delta_M^{enter}
\]

とする。

例：

\[
0.015<0.020<0.025
\]

これにより0.02近辺を行き来するFeatureが半年ごとにRole変更されることを防ぐ。

---

# 56. LINEAR → QUADRATIC の格上げ

Curvature：

\[
C_j
=
\mu_L+\mu_H-2\mu_N
\]

について、通常のQuadratic Thresholdより少し強いEntry Thresholdを使用する。

例：

\[
|C_j|\ge\delta_C^{enter}
\]

\[
\delta_C^{enter}=0.035
\]

かつ、

\[
q_j^{Curve}<0.10
\]

とする。

補助条件：

\[
|t_j^{Curve}|\ge2
\]

初期仕様例：

```text
LINEAR → QUADRATIC

abs(Curvature) >= 0.035
AND
Curvature q-value < 0.10
```

Sign Flipがあれば追加Evidenceとして保存する。

---

# 57. QUADRATIC → LINEAR の格下げ

格下げ側：

\[
|C_j|<\delta_C^{exit}
\]

例：

\[
\delta_C^{exit}=0.020
\]

または、

\[
q_j^{Curve}\ge0.20
\]

が継続する場合に格下げ候補とする。

例：

```text
QUADRATIC → LINEAR

abs(Curvature) < 0.020
OR
Curvature q-value >= 0.20
```

ただし一度の判定で変更するか、2回連続Failを要求するかはSensitivityとして確認する。

---

# 58. Role変更のHysteresis

Role変更用に、

\[
Entry\ Threshold > Exit\ Threshold
\]

とする。

例：

```yaml
role_change:
  main_to_linear_range_enter: 0.025
  linear_to_main_range_exit: 0.015

  linear_to_quad_curve_enter: 0.035
  quad_to_linear_curve_exit: 0.020

  upgrade_q: 0.10
  downgrade_q: 0.20
```

この仕組みにより、

> Feature Validityが更新されるたびにFeature Roleが頻繁に入れ替わる

問題を抑制する。

---

# 59. Feature Role変更量のモニタリング

レビュー日 \(T_k\) ごとに、前回からRoleが変更されたFeature割合を計算する。

\[
RoleChangeRate_k
=
\frac{
\#\{j:Role_{j,k}\neq Role_{j,k-1}\}
}{
\# EligibleFeatures_k
}
\]

さらにRoleを、

```text
DROP = 0
MAIN = 1
LINEAR = 2
QUADRATIC = 3
```

と数値化し、

\[
RoleDistance_k
=
\frac1P
\sum_j
|Role_{j,k}-Role_{j,k-1}|
\]

も保存する。

これはモデル構造の変化量を説明する指標とする。

---

# 60. Role変更が大きすぎる場合

半年レビューで、

\[
RoleChangeRate
\]

が極端に高い場合、それを「市場構造が変わった」と即断しない。

まず、

- Regime Sample Size
- VRP分布
- データ欠損
- Rolling Window長
- Threshold境界
- 特定期間への依存

を確認する。

初期Warning Threshold例：

```text
RoleChangeRate > 30%
```

の場合、

```text
STRUCTURAL_CHANGE_WARNING = True
```

として、人間による確認対象とする。

30%は固定的な真値ではなく、初期ガバナンス値として使用する。

---

# 61. モデルFamily変更の考え方

Feature Roleが更新されたからといって、

```text
M1 → M2
```

等のモデルFamilyを自動変更しない。

モデルFamily変更は、

\[
\boxed{
Historical\ Walk\text{-}Forward\ OOS
}
\]

によって定量評価する。

例：

現行モデル：

```text
Incumbent = M1
```

候補：

```text
Challenger = M2
```

とする。

---

# 62. Model Change Evaluation Window

半年レビューだからといって、直近6回の月次予測だけで判断しない。

月次OOS 6点ではサンプル数が少なすぎるため、

第一候補として、

```text
直近24か月のMonthly Walk-Forward OOS
```

を使用する。

最低でも、

```text
12か月
```

を確保する。

十分なLive OOSが存在しない初期期間は、レビュー日時点より前のデータだけを利用してHistorical Walk-Forward Replayを実施する。

重要：

各Historical Prediction Dateでも、その時点で利用可能だった情報のみを使用する。

---

# 63. Candidate vs Incumbent のPaired Comparison

月次Prediction Date \(t\) について、

\[
d_t^{IC}
=
IC_t^{Candidate}
-
IC_t^{Incumbent}
\]

を作る。

平均改善：

\[
\Delta IC
=
\frac1T
\sum_t d_t^{IC}
\]

を算出する。

同じ月の同じ断面を比較するため、単純な別々の平均ではなくPaired Differenceを主に使う。

---

# 64. Rank IC改善の統計量

\[
t_{\Delta IC}^{HAC}
=
\frac{
\overline{d^{IC}}
}{
SE_{HAC}(\overline{d^{IC}})
}
\]

を算出する。

月次系列なので、HAC lagは日次20BD Feature Validityとは別に設定する。

初期候補：

```text
Monthly OOS HAC lag = 1 ～ 3
```

とする。

---

# 65. Monthly Win Rate

\[
WinRate^{IC}
=
\frac{
\#\{t:IC_t^{Candidate}>IC_t^{Incumbent}\}
}{
T
}
\]

を保存する。

平均ICだけでなく、

> 何か月でCandidateが勝ったか

を説明できるようにする。

---

# 66. Portfolio Improvement

Top3-Bottom3 Net Returnについて、

\[
d_t^{LS}
=
R_{t,Net}^{Candidate}
-
R_{t,Net}^{Incumbent}
\]

を計算する。

平均改善：

\[
\Delta LS_{Net}
=
\frac1T\sum_td_t^{LS}
\]

を保存する。

CandidateがRank ICを改善していても、Cost控除後Portfolioが悪化する場合は変更理由として弱い。

---

# 67. Regime別改善

VRP Regime \(r\) ごとに、

\[
\Delta IC_r
=
IC_r^{Candidate}
-
IC_r^{Incumbent}
\]

を計算する。

特にM2へ変更する場合、

> Quadraticを追加したにもかかわらず、Low / High VRPで全く改善していない

場合は変更理由として弱い。

---

# 68. Complexity Guardrail

より複雑なモデルへ変更する場合は、単純モデルへ変更する場合より強いEvidenceを要求する。

複雑度：

```text
M0 < MD < M1 < M2
```

例えば、

```text
M1 → M2
```

では、

- \(\Delta IC > 0\)
- ICIR悪化なし
- Net Top-Bottom悪化なし
- Regime別に説明可能

を要求する。

一方、

```text
M2 → M1
```

では、

> M2の追加価値が確認できなくなった

こと自体を格下げ理由としてよい。

---

# 69. モデル変更の推奨Gate Rule

初期仕様として、Candidateへ変更する条件を以下とする。

## Gate A：Primary Performance

\[
\Delta IC>0
\]

かつ、

\[
t_{\Delta IC}^{HAC}\ge1.0
\]

または、

\[
WinRate^{IC}\ge60\%
\]

を満たす。

## Gate B：Risk-adjusted Stability

\[
ICIR_{Candidate}
\ge
0.9\times ICIR_{Incumbent}
\]

つまりICIRが10%以上悪化しない。

## Gate C：Portfolio

\[
\Delta LS_{Net}\ge0
\]

## Gate D：Regime Explanation

Candidate導入理由と整合するRegimeで、

\[
\Delta IC_r>0
\]

が確認できる。

例：

M2ならLow / Highの少なくとも一方で改善が確認されること。

---

# 70. より厳格なStrong Change Rule

十分なサンプルがある場合は、

\[
t_{\Delta IC}^{HAC}\ge1.64
\]

またはBootstrapで、

\[
P(\Delta IC>0)\ge90\%
\]

をStrong Evidenceとする。

ただし月次OOS数が少ない初期段階で5%有意差のみを絶対条件にすると変更がほぼ不可能になるため、

```text
Performance
+
Stability
+
Economic Value
+
Regime Consistency
```

のGate方式をPrimaryとする。

---

# 71. Model Change Decision

最終Decision：

```text
CHANGE
KEEP
REVIEW
```

とする。

### CHANGE

全Primary Gateを満たす。

### KEEP

Primary Performance Gateを満たさない。

### REVIEW

性能は改善しているが、

- ICIR悪化
- Net Portfolio悪化
- Regime整合性なし
- Role Change Rate異常

等が存在する。

---

# 72. 半年 / 年次レビューの説明テンプレート

モデル変更時は以下を自動出力する。

```text
Review Date:
Incumbent Model:
Candidate Model:

Evaluation Window:
Number of Monthly OOS Observations:

Mean Rank IC
  Incumbent:
  Candidate:
  Difference:

HAC t-stat of IC Difference:

IC Win Rate:

ICIR
  Incumbent:
  Candidate:

Top3-Bottom3 Net
  Incumbent:
  Candidate:
  Difference:

Regime Delta IC
  Low:
  Normal:
  High:

Feature Role Change Rate:

Number of
  MAIN → LINEAR:
  LINEAR → QUADRATIC:
  QUADRATIC → LINEAR:
  LINEAR → MAIN:

Decision:
Reason Code:
```

---

# 73. Reason Code

変更理由を標準化する。

例：

```text
MC01 = Rank IC improvement
MC02 = ICIR improvement
MC03 = Net portfolio improvement
MC04 = Recent regime adaptation
MC05 = Quadratic effect newly supported
MC06 = Quadratic effect no longer supported
MC07 = Rolling structure changed
MC08 = Model simplification without performance loss
MC09 = Change rejected due to instability
MC10 = Change rejected due to insufficient sample
```

---

# 74. Feature変更理由も保存する

各Featureについて、

```text
feature
old_role
new_role
range_old
range_new
curvature_old
curvature_new
q_old
q_new
reason
```

を保存する。

例：

```text
rv_5d
LINEAR → QUADRATIC
Curvature = 0.051
q = 0.04
Reason = RC02_QUADRATIC_ENTER
```

---

# 75. Role Reason Code

```text
RC01_MAIN_STABLE
RC02_QUADRATIC_ENTER
RC03_QUADRATIC_EXIT
RC04_LINEAR_ENTER
RC05_LINEAR_EXIT
RC06_EQUIVALENCE_RESTORED
RC07_INSUFFICIENT_REGIME_SAMPLE
RC08_FDR_FAIL
RC09_THRESHOLD_BUFFER
```

これにより、

> 「なぜ半年後にこのFeatureだけ二次項になったのか」

を後から機械的に説明できる。

---

# 76. Review Report

半年または年次レビューごとに、

```text
model_review_YYYYMM.xlsx
```

を出力する。

推奨Sheet：

```text
Decision_Summary
Candidate_vs_Incumbent
RankIC_Difference
Regime_Comparison
Portfolio_Comparison
FeatureRole_Changes
FeatureValidity_EXP
FeatureValidity_ROLL
Threshold_Config
```

---

# 77. モデル変更時の重要原則

半年または年1回のレビューで最も重要なのは、

\[
\boxed{
Schedule\ Driven\ Review
\neq
Schedule\ Driven\ Change
}
\]

である。

つまり、

> 半年経ったから変える

のではなく、

> 半年経ったので過去情報だけを使って再評価し、定量的な改善が確認された場合のみ変える

とする。

---

# 78. 更新頻度そのものの妥当性

半年更新と年次更新のどちらが良いかもOOSで比較可能とする。

例えば、

```text
FV_UPDATE = 6M
FV_UPDATE = 12M
```

について同じHistorical Walk-Forwardを実施し、

- Monthly Rank IC
- ICIR
- Top-Bottom Net
- Role Change Rate
- Model Change Count

を比較する。

頻繁な更新が性能改善につながらず、

\[
RoleChangeRate
\]

だけ高くなる場合、年次更新を優先する。

逆に半年更新でRecent Regimeへの適応が明確に改善する場合、半年更新を採用する。

---

# 79. 更新頻度比較の指標

半年更新 \(6M\) と年次更新 \(12M\) について、

\[
\Delta IC_{6M-12M}
=
IC_{6M}-IC_{12M}
\]

\[
\Delta ICIR_{6M-12M}
=
ICIR_{6M}-ICIR_{12M}
\]

\[
\Delta Net_{6M-12M}
=
NetReturn_{6M}-NetReturn_{12M}
\]

を比較する。

さらに、

\[
ComplexityCost
=
ModelChangeCount
+
\lambda_{role}RoleChangeRate
\]

のような管理指標も参考表示する。

Primary判断は予測性能と安定性で行い、Change Countはガバナンス上の補助指標とする。

---

# 80. 更新頻度の最終決定ルール

初期案：

### Semiannualを採用

以下が確認される場合。

```text
6MのMonthly Rank IC > 12M
AND
6MのICIRが悪化しない
AND
6MのNet Top-Bottomが改善
AND
Role Change Rateが過度に高くない
```

### Annualを採用

以下の場合。

```text
6Mと12Mの性能差が小さい
OR
6MでRole変更が多いだけ
OR
6MでCoefficient / Roleが不安定
```

この場合はより単純な12M更新を選択する。

---

# 81. MVPでの実装優先度

モデル変更ガバナンスについても、最初から全項目を複雑に実装しなくてよい。

MVPでは最低限以下を実装する。

```text
1. 前回Roleとの比較
2. Role Change Rate
3. Candidate vs Incumbent のMonthly ΔRankIC
4. HAC t-stat of ΔRankIC
5. IC Win Rate
6. ICIR
7. Top-Bottom Net
8. Regime別ΔIC
9. CHANGE / KEEP / REVIEW
10. Reason Code
```

Bootstrapや高度なModel Confidence Set等はPhase 2とする。

---

# 82. 最終的な半年 / 年次レビューの流れ

```text
Scheduled Review Date
        ↓
過去情報だけでFeature Validity更新
        ↓
EXP / ROLL Role Candidate作成
        ↓
前回Roleとの変更量を計測
        ↓
Candidate ModelをHistorical Walk-Forwardで再評価
        ↓
Incumbent vs Candidate
        ↓
ΔRankIC
HAC t-stat
Win Rate
ICIR
Net Top-Bottom
Regime ΔIC
        ↓
Gate Rule
   ├─ PASS   → CHANGE
   ├─ FAIL   → KEEP
   └─ MIXED  → REVIEW
        ↓
Decision Report保存
        ↓
次回レビューまでモデル構造Freeze
```

---

# 83. 半年・年次変更を説明する最終ロジック

モデル変更は、

\[
\boxed{
\text{Feature Validityの統計的変化}
}
\]

だけでも、

\[
\boxed{
\text{OOS性能改善}
}
\]

だけでも決めない。

以下の双方を満たすことを原則とする。

\[
\boxed{
Structural\ Evidence
+
Predictive\ Evidence
}
\]

Structural Evidence：

- Regime Range
- Monotonicity
- Curvature
- FDR
- Role Stability

Predictive Evidence：

- Monthly OOS Rank IC
- Paired \(\Delta IC\)
- HAC t-stat
- IC Win Rate
- ICIR
- Net Top-Bottom
- Regime別改善

これにより、

> 「半年経過したからモデルを変更した」

ではなく、

> 「半年レビュー時点でFeature効果の構造変化が定量的に確認され、かつ候補モデルが過去情報のみを用いたWalk-Forward OOSで現行モデルを上回ったため変更した」

と説明可能な状態を作る。


以上。
