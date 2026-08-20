# ValueDate基準 20営業日予測 × 4モデル・アンサンブル型 Long/Short ポートフォリオ構築 実装指示書
## 日次学習データを使用し、月次の指定ValueDateでのみ20BD予測・ポートフォリオ構築を行う

---

# 0. 最重要前提

本分析は、**日次リターンを毎日予測するモデルではない**。

使用する学習データは日次であるが、実際の予測は、月ごとにあらかじめ指定された **ValueDate** においてのみ実施する。

各ValueDate \(T_m\) について、

\[
\boxed{
T_m \rightarrow T_m+20営業日
}
\]

の20営業日先リターンを予測する。

したがって、本分析の頻度は以下のように明確に分ける。

```text
学習データ頻度        : Daily
学習Label             : 各日から20営業日先Forward Return
モデル再学習          : 各ValueDateごと
予測実行日            : 月次の指定ValueDateのみ
予測ホライズン        : ValueDateから20営業日
ポートフォリオ構築    : 各ValueDateのみ
OOS評価               : 各ValueDateの20BD実現リターンのみ
```

日次学習データを利用する理由は、20BD予測関係の推定に利用できる過去情報を最大限活用するためである。

一方、日々ポジションを更新することはしない。

---

# 1. ValueDate

ValueDateは外部から指定する。

例：

```text
2024-01-31
2024-02-29
2024-03-28
...
```

必ずしも月末である必要はない。

入力例：

```python
VALUE_DATES = [
    "2024-01-31",
    "2024-02-29",
    "2024-03-28",
]
```

モデルの予測、Net Long選択、最終ウェイト計算はこのValueDateでのみ実施する。

---

# 2. 学習データ

特徴量データは既に作成済みであることを前提とする。

最低限以下を保持する。

```text
date
asset_id
feature_1
feature_2
...
vrp
target_20bd
label_end_date
```

各日 \(t\) のLabelは、

\[
y_{i,t}^{20}
=
R_{i,t\rightarrow t+20BD}
\]

とする。

つまり、日次学習データの各行は、

```text
その日の日次特徴量
    ↓
その日から20営業日先のリターン
```

という対応を持つ。

---

# 3. Prediction DateとTraining Dataの違い

重要：

日次データすべてについてPredictionを出す必要はない。

日次データはTraining Sampleとしてのみ使用する。

実際にPredictionを保存するのは、

\[
t\in VALUE\_DATES
\]

の場合だけ。

擬似コード：

```python
for value_date in VALUE_DATES:

    train_df = make_training_data(
        daily_feature_df,
        value_date=value_date,
    )

    model.fit(train_df)

    x_test = daily_feature_df[
        daily_feature_df["date"] == value_date
    ]

    prediction = model.predict(x_test)

    save_prediction(value_date, prediction)
```

---

# 4. Label Availability

ValueDateを \(T_m\) とする。

Trainに利用可能な日次行は、

\[
label\_end\_date \le TrainCutoff_m
\]

を満たすものだけとする。

ValueDate直前の日次データであっても、その20BD先リターンがValueDate時点で未確定なら学習には使用しない。

したがって、単純に、

```python
date < value_date
```

だけでTrainを作ってはいけない。

必ず、

```python
label_end_date <= train_cutoff
```

を確認する。

---

# 5. Purge

20BD Forward Return Labelを利用するため、Training / Validationの境界でもLabel期間の重複を避ける。

最低限、

\[
Purge \ge 20BD
\]

を使用する。

初期候補：

```python
PURGE_BD = 25
```

ただし、ValueDate Test自体は月次1回のみ。

---

# 6. Overlapping Label

日次Training Dataでは、

\[
R_{t:t+20}
\]

と、

\[
R_{t+1:t+21}
\]

が大きくoverlapする。

これはTraining Sampleとして利用してよい。

ただし、

- Feature Validity
- Rank IC
- HAC t-stat
- Curvature
- Equivalence

等の統計推論ではserial dependenceを考慮する。

初期実装：

```python
HAC_LAG = 20
```

とする。

---

# 7. 4モデル

各ValueDateで以下の4モデルを再学習し、ValueDate断面の12アセットについて20BD予測を作成する。

## M0

線形モデル。

Target：

```text
20BD Relative Return
```

## M1

VRP Linear Interactionを含む線形モデル。

Target：

```text
20BD Relative Return
```

## LGB1

LightGBM。

Target：

```text
20BD Absolute Return
```

## LGB2

別仕様のLightGBM。

Target：

```text
20BD Absolute Return
```

全モデルのForecast Horizonは同一で、

\[
\boxed{
ValueDate \rightarrow ValueDate+20BD
}
\]

とする。

---

# 8. ValueDateごとの予測データ

各ValueDateで以下を作る。

```text
value_date
asset_id
pred_m0_rel_20bd
pred_m1_rel_20bd
pred_lgb1_abs_20bd
pred_lgb2_abs_20bd
residual_vol
```

このValueDate断面のみを使って最終ポートフォリオを構築する。

---

# 9. Linear Core Score

M0 / M1は相対リターン予測。

ValueDate断面の12アセットでPercentile Rankへ変換する。

\[
r_{i,M0}
=
PctRank(\hat r_{i,M0})-0.5
\]

\[
r_{i,M1}
=
PctRank(\hat r_{i,M1})-0.5
\]

Linear Core：

\[
S_i^{LIN}
=
\frac{
r_{i,M0}+r_{i,M1}
}{2}
\]

Linearを最終ポジション方向のCoreとする。

---

# 10. LightGBM Relative Score

LGB1 / LGB2はAbsolute Return Forecastである。

Asset Selection用にはValueDate断面でRank化する。

\[
r_{i,LGB1}
=
PctRank(\hat r_{i,LGB1})-0.5
\]

\[
r_{i,LGB2}
=
PctRank(\hat r_{i,LGB2})-0.5
\]

\[
S_i^{ML}
=
\frac{
r_{i,LGB1}+r_{i,LGB2}
}{2}
\]

---

# 11. Primary Alpha Logic

説明可能性を優先する。

LightGBMにLong / Short方向を反転させない。

Linear CoreとMLの方向が一致：

\[
S_i^{FINAL}
=
1.2S_i^{LIN}
\]

方向が不一致：

\[
S_i^{FINAL}
=
0.8S_i^{LIN}
\]

したがって、

\[
\boxed{
方向決定 = M0/M1
}
\]

\[
\boxed{
LightGBM = Linear Signalの確信度調整
}
\]

とする。

Config：

```python
ML_CONFIRM_UP = 1.20
ML_CONFIRM_DOWN = 0.80
```

---

# 12. 比較用Alpha Logic

Secondaryとして、

\[
S_i^{FINAL}
=
0.8S_i^{LIN}
+
0.2S_i^{ML}
\]

も比較可能とする。

ただしPrimaryはLinear Core / ML Confirmation。

---

# 13. Residual Vol Weighting

現在の方針を維持する。

\[
Q_i
=
\frac{
S_i^{FINAL}
}{
\sigma_{i,res}
}
\]

残差ボラが小さいAssetほど同じScoreでも大きいPositionになる。

Residual Volの推定ロジック自体は変更しない。

---

# 14. Long Gross

Long Grossはユーザー設定値。

例：

```python
LONG_GROSS = 0.95
```

モデルが自動的にLong Grossを変更しない。

---

# 15. Net Long候補

Config：

```python
NET_LONG_CANDIDATES = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
]
```

各Net Long \(N\) について、

\[
ShortGross(N)
=
LongGross-N
\]

とする。

---

# 16. Position Cap

各Asset：

\[
|w_i|<33\%
\]

実装値：

```python
POSITION_CAP = 0.325
```

---

# 17. 各Net候補のPortfolio作成

各ValueDateについて、

```text
Net = 0%
Net = 5%
Net = 10%
Net = 15%
Net = 20%
```

の5つのPortfolioを作る。

各候補で、

\[
L=LongGross
\]

\[
S=LongGross-Net
\]

とする。

Long / Short内の相対配分は、

\[
Q_i
=
S_i^{FINAL}/ResidualVol_i
\]

で決める。

---

# 18. Long Side Allocation

\[
q_i^+
=
\max(Q_i,0)
\]

Raw：

\[
w_{i,L}^{raw}
=
L
\frac{q_i^+}{\sum_jq_j^+}
\]

Position Cap適用後、余剰Grossは同じLong Sideの未Cap Assetへ再配分する。

---

# 19. Short Side Allocation

\[
q_i^-
=
\max(-Q_i,0)
\]

Raw：

\[
w_{i,S}^{raw}
=
-
S
\frac{q_i^-}{\sum_jq_j^-}
\]

Position Cap適用後、余剰GrossはShort Side内で再配分。

---

# 20. Capacity Check

Long Gross Targetを達成可能か確認。

\[
N_{Long}\times PositionCap
\ge LongGross
\]

Short：

\[
N_{Short}\times PositionCap
\ge ShortGross
\]

満たさない場合、

```text
CAPACITY_WARNING = True
```

を出力する。

---

# 21. Net Long選択用Expected Return

Net候補の比較は20BD Expected Absolute Portfolio Returnで行う。

---

# 22. Linear Relative Alpha

ValueDateで、

\[
\alpha_i^{LIN}
=
\frac{
\hat r_{i,M0}^{rel}
+
\hat r_{i,M1}^{rel}
}{2}
\]

とする。

断面平均が0でない場合はdemeanする。

---

# 23. LightGBM Market Component

ValueDate断面のLGB Absolute Return Forecastから、

\[
m_t^{ML}
=
\frac12
\left[
\frac1{12}\sum_i\hat r_{i,LGB1}^{abs}
+
\frac1{12}\sum_i\hat r_{i,LGB2}^{abs}
\right]
\]

を計算する。

これは、

\[
\boxed{
ValueDateから20BD先までの市場共通方向予測
}
\]

として使用する。

---

# 24. LightGBM Relative Alpha

\[
\alpha_i^{ML}
=
\frac12
\left[
(\hat r_{i,LGB1}^{abs}-\overline{\hat r}_{LGB1})
+
(\hat r_{i,LGB2}^{abs}-\overline{\hat r}_{LGB2})
\right]
\]

---

# 25. Expected Return Vector

Linearを厚めに使用する。

第一候補：

\[
\hat\mu_i
=
m_t^{ML}
+
0.8\alpha_i^{LIN}
+
0.2\alpha_i^{ML}
\]

Sensitivity：

```text
80 / 20
90 / 10
```

Primaryは80 / 20固定。

毎月最適化してWeightを変えない。

---

# 26. Net候補のExpected Portfolio Return

各Net候補 \(N\) のPortfolio Weightを、

\[
w(N)
\]

とする。

\[
\hat R_p(N)
=
w(N)^\top\hat\mu
\]

これは、

\[
\boxed{
ValueDateから20BD先のExpected Portfolio Return
}
\]

である。

---

# 27. Covariance Matrix

Net候補のRisk比較にはTotal Return Covarianceを使用する。

Residual Vol：

```text
Asset Sizing用
```

Total Covariance：

```text
Portfolio Risk評価用
```

と役割分担する。

---

# 28. Covariance推定時点

各ValueDate \(T_m\) ごとに、

\[
t\le T_m
\]

の過去Returnのみを使う。

未来Returnは禁止。

第一候補：

```text
lookback = 252BD
Ledoit-Wolf Shrinkage
```

---

# 29. 20BD Covariance

日次Covariance：

\[
\Sigma_{daily,T_m}
\]

から、

\[
\Sigma_{20,T_m}
=
20\Sigma_{daily,T_m}
\]

とする。

これは各ValueDateで更新する。

---

# 30. Candidate Portfolio Risk

各Net候補について、

\[
\sigma_p(N)
=
\sqrt{
w(N)^\top
\Sigma_{20,T_m}
w(N)
}
\]

を計算する。

これはValueDateから20BD Horizonに対応するPortfolio Volatility。

---

# 31. Primary Net Selection

第一候補：

\[
\boxed{
Risk Limit内でExpected 20BD Return最大
}
\]

Risk Limit：

\[
\sigma_p(N)\le\sigma_{max}
\]

Feasible Candidate中、

\[
N^*
=
\arg\max_N
\hat R_p(N)
\]

を選択する。

---

# 32. 比較Net Selection

MVPでは以下3つを比較する。

### N0：固定

```text
Net Long = 10%
```

### N1：Expected Sharpe最大

\[
N^*
=
\arg\max_N
\frac{\hat R_p(N)}
{\sigma_p(N)}
\]

### N2：Risk Limit内Expected Return最大

Primary Candidate。

---

# 33. 重要：Realized ReturnでNetを選ばない

各ValueDateで、

> 20BD後に実際に最も高いリターンになったNet

を選ぶことは禁止。

Net選択はValueDate時点で利用可能な、

- 4モデルForecast
- Residual Vol
- Total Covariance
- User Long Gross
- Risk Limit

だけで決める。

20BD後のRealized ReturnはOOS評価専用。

---

# 34. 最終ウェイト

選択Net \(N^*\) に対応する、

\[
w(N^*)
\]

を最終Portfolio Weightとする。

---

# 35. 必須Constraint Check

各ValueDateで、

\[
LongGross
=
\sum_{w_i>0}w_i
\]

\[
ShortGross
=
\sum_{w_i<0}|w_i|
\]

\[
NetLong
=
\sum_iw_i
\]

を計算。

必須：

```python
assert abs(long_gross - USER_LONG_GROSS) <= tol
assert short_gross < 1.0
assert 0.0 <= net_long <= 0.20
assert max(abs(weights)) <= POSITION_CAP + tol
```

---

# 36. OOS評価

OOS評価もValueDate単位のみ。

各ValueDate \(T_m\) の最終Portfolioについて、

\[
R_{p,T_m}^{realized}
=
w_{T_m}^\top
R_{T_m\rightarrow T_m+20BD}
\]

を計算する。

日次でPortfolio Returnを毎日予測するわけではない。

---

# 37. ValueDate OOS出力

```text
value_date
selected_net_long
long_gross
short_gross
expected_return_20bd
expected_vol_20bd
expected_sharpe
realized_return_20bd
risk_limit_pass
capacity_warning
```

---

# 38. Asset Weight出力

```text
value_date
asset_id
pred_m0_rel_20bd
pred_m1_rel_20bd
pred_lgb1_abs_20bd
pred_lgb2_abs_20bd
score_linear
score_ml
score_final
residual_vol
risk_adjusted_score
final_weight
```

---

# 39. Net Candidate比較出力

```text
value_date
net_candidate
long_gross
short_gross
expected_return_20bd
expected_vol_20bd
expected_sharpe
risk_limit_pass
selected_flag
```

---

# 40. 推奨Config

```yaml
prediction:
  frequency: value_date_only
  horizon_bd: 20

portfolio:
  long_gross: 0.95

  net_long_candidates:
    - 0.00
    - 0.05
    - 0.10
    - 0.15
    - 0.20

  position_cap: 0.325

  alpha_method: linear_core_ml_confirmation

  ml_confirmation_up: 1.20
  ml_confirmation_down: 0.80

  expected_return_linear_weight: 0.80
  expected_return_ml_relative_weight: 0.20

  net_selection_method: return_max_under_risk_limit

  portfolio_vol_limit_20bd: null

covariance:
  lookback_bd: 252
  estimator: ledoit_wolf
  horizon_bd: 20

training:
  data_frequency: daily
  label_horizon_bd: 20
  label_availability_check: true
  purge_bd: 25
```

---

# 41. 実装フロー

```text
【Daily Feature Data】
        ↓
各ValueDateをLoop
        ↓
label_end_date <= train_cutoff
の日次Training Dataを抽出
        ↓
M0 / M1 / LGB1 / LGB2を再学習
        ↓
そのValueDate断面のみ20BD Forecast
        ↓
Linear Core / ML Confirmation
        ↓
Residual Vol Adjustment
        ↓
User Long Gross読込
        ↓
Net = 0/5/10/15/20% の5候補
        ↓
各候補のPortfolio Weight作成
        ↓
Position Cap適用
        ↓
ValueDate時点までのReturnでCovariance推定
        ↓
各候補のExpected 20BD Return / Risk算出
        ↓
Net Long選択
        ↓
Final Weight保存
        ↓
20BD経過後にRealized Return評価
```

---

# 42. 推奨コード構成

既存コードへ追加する場合、大規模分割は不要。

```text
portfolio_value_date.py
run_portfolio_value_dates.py
```

程度でよい。

既存1ファイルに統合可能なら、無理に分割しない。

---

# 43. 主要関数

```python
def get_training_data_for_value_date(
    df,
    value_date,
):
    pass


def fit_four_models(
    train_df,
):
    pass


def predict_at_value_date(
    models,
    value_date_df,
):
    pass


def make_linear_core_score(
    pred_m0,
    pred_m1,
):
    pass


def make_ml_score(
    pred_lgb1,
    pred_lgb2,
):
    pass


def apply_ml_confirmation(
    score_linear,
    score_ml,
):
    pass


def residual_vol_adjust(
    score,
    residual_vol,
):
    pass


def make_portfolio_for_net(
    score,
    residual_vol,
    long_gross,
    net_long,
    position_cap,
):
    pass


def build_expected_return_vector(
    predictions,
):
    pass


def estimate_covariance_asof_value_date(
    return_df,
    value_date,
):
    pass


def evaluate_net_candidates(
    portfolios,
    expected_return,
    covariance,
):
    pass


def select_net_long(
    candidate_table,
    method,
    risk_limit,
):
    pass
```

---

# 44. 初期比較

まず以下を比較する。

### P0

```text
Linear Core / ML Confirmation
+
Net 10%固定
```

### P1

```text
Linear Core / ML Confirmation
+
Expected Sharpe最大Net
```

### P2

```text
Linear Core / ML Confirmation
+
Risk Limit内Expected 20BD Return最大Net
```

第一候補：

\[
\boxed{P2}
\]

---

# 45. 初期実装で行わないこと

```text
毎日Forecast
毎日Portfolio Rebalance
12Asset Weightのフル平均分散最適化
Dynamic Model Weight
Dynamic Linear/ML Weight
Net候補1%刻み
Turnover Penalty
Black-Litterman
```

まずValueDate単位のシンプルなルールを完成させる。

---

# 46. 最終的な考え方

本モデルは、

\[
\boxed{
Daily\ Data
\neq
Daily\ Forecast
}
\]

である。

正しくは、

\[
\boxed{
Daily\ Training\ Data
\rightarrow
Monthly\ ValueDate
\rightarrow
20BD\ Forecast
}
\]

である。

Portfolio Constructionも各ValueDateで一度だけ行う。

最終構造：

\[
\boxed{
M0/M1 = Relative\ Linear\ Core
}
\]

\[
\boxed{
LGB1/LGB2 = Absolute\ ML\ Overlay
}
\]

\[
\boxed{
Residual\ Vol = Asset\ Weighting
}
\]

\[
\boxed{
Total\ Covariance = Net\ Long\ Candidate\ Risk
}
\]

\[
\boxed{
LongGross = User\ Setting
}
\]

\[
\boxed{
NetLong = 0\%,5\%,10\%,15\%,20\%
}
\]

そして、

\[
\boxed{
ValueDate時点の情報だけを用い、
Risk Limit内でExpected 20BD Returnが最大となるNet Longを選択
}
\]

する。

以上。
