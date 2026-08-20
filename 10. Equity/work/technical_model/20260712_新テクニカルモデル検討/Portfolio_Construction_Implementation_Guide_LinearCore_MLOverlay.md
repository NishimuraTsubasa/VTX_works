# 4モデル・アンサンブル型 Long/Short ポートフォリオ構築 実装指示書
## Linear Core + LightGBM Overlay + Residual Vol Weighting + Dynamic Net Long

---

# 0. 目的

本書は、既に作成済みの4モデルの予測値を用いて、12アセットのLong/Shortポートフォリオの最終ウェイトを決定するための実装指示書である。

対象モデルは以下。

- M0：線形モデル、20BD相対リターン予測
- M1：線形モデル、20BD相対リターン予測
- LGB1：LightGBM、20BD絶対リターン予測
- LGB2：LightGBM、20BD絶対リターン予測

本実装では、説明可能性を重視して **M0 / M1をCore** とし、LightGBMは補助的に利用する。

また、現行方針である、

\[
\boxed{
\text{予測Scoreが高く、残差ボラが低いアセットほど大きく持つ}
}
\]

という資産別ウェイト決定ロジックは維持する。

フルの平均分散最適化で12アセットのウェイトを直接最適化するのではなく、まず既存のResidual Vol Weightingで各Net Long候補の完成ポートフォリオを作り、その中からExpected ReturnとCovariance Riskを用いてNet Longを選択する。

---

# 1. 最終ポートフォリオ制約

以下を必須とする。

## 1.1 Long Gross

Long Grossはモデルが自動決定するのではなく、ユーザー設定値とする。

例：

```python
LONG_GROSS = 0.95
```

想定範囲：

\[
0.90\le LongGross<1.00
\]

ただし実装上はユーザーConfig値として持たせ、ハードコードしない。

---

## 1.2 Net Long

Net Long候補を離散的なリストとして保持する。

第一候補：

```python
NET_LONG_CANDIDATES = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
]
```

制約：

\[
0\le NetLong\le0.20
\]

---

## 1.3 Short Gross

各Net Long候補 \(N\) に対して、

\[
ShortGross(N)=LongGross-N
\]

とする。

例：Long Gross = 95%

| Net Long | Long Gross | Short Gross |
|---:|---:|---:|
| 0% | 95% | 95% |
| 5% | 95% | 90% |
| 10% | 95% | 85% |
| 15% | 95% | 80% |
| 20% | 95% | 75% |

必ず、

\[
ShortGross<100\%
\]

を満たすことを確認する。

---

## 1.4 個別ポジション上限

各アセットの最終ウェイトは、

\[
|w_i|<33\%
\]

とする。

実装では安全余裕を持たせ、

```python
POSITION_CAP = 0.325
```

を第一候補とする。

---

# 2. 全体処理フロー

```text
M0 relative prediction ─┐
M1 relative prediction ─┼─→ Linear Core
                        │
LGB1 absolute prediction ─┐
LGB2 absolute prediction ─┴─→ ML Confirmation / Market Component
                              ↓
                    Final Cross-sectional Score
                              ↓
                    Residual Vol Adjustment
                              ↓
          Net Long候補ごとに完成Portfolioを作成
                              ↓
               Total CovarianceでRisk計測
                              ↓
       Expected Return / Risk基準でNet Longを選択
                              ↓
                     Final Portfolio
```

---

# 3. モデル予測値の入力

月次Prediction Dateごとに以下を受け取る。

```text
date
asset_id
pred_m0
pred_m1
pred_lgb1
pred_lgb2
residual_vol
```

別途、共分散推定用に各アセットの日次実現リターンデータを保持する。

---

# 4. Linear Core Score

M0 / M1は20BD相対リターン予測。

まず各Prediction Date断面で、それぞれの予測値をPercentile Rankへ変換する。

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

Linearモデルが最終Long / Short方向の中心となる。

---

# 5. LightGBM Relative Score

LightGBMは絶対リターン予測である。

Cross-sectional Asset Selection用には、その絶対水準ではなく断面順位を利用する。

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

# 6. Primary Alpha Logic：Linear Core / ML Confirmation

説明可能性を優先し、PrimaryではLightGBMにLong / Short方向を反転させない。

Linear CoreとMLの方向が一致する場合：

\[
S_i^{FINAL}
=
1.2S_i^{LIN}
\]

方向が異なる場合：

\[
S_i^{FINAL}
=
0.8S_i^{LIN}
\]

したがって、

\[
\boxed{
\text{ポジション方向はLinearが決定し、MLは確信度のみ調整}
}
\]

とする。

実装：

```python
same_direction = np.sign(score_linear) == np.sign(score_ml)

score_final = np.where(
    same_direction,
    1.20 * score_linear,
    0.80 * score_linear,
)
```

`score_linear == 0`の場合は0のままとする。

---

# 7. Secondary Alpha Logic：Weighted Ensemble

比較用として以下も保持可能。

\[
S_i^{FINAL}
=
0.8S_i^{LIN}
+
0.2S_i^{ML}
\]

ただしPrimaryはSection 6のLinear Core / ML Confirmationとする。

Config化：

```python
ALPHA_METHOD = "linear_core_ml_confirmation"
```

候補：

```text
linear_core_ml_confirmation
weighted_80_20
```

---

# 8. Residual Vol Adjustment

現行方針を維持する。

\[
Q_i
=
\frac{
S_i^{FINAL}
}{
\sigma_{i,res}
}
\]

ここで、

\[
\sigma_{i,res}>0
\]

を必須とする。

Residual Volが欠損・0以下の場合は当該アセットをその月の配分対象外とするか、既存のデータ補完ルールを適用する。

Residual Volの算出方法自体は本書では変更しない。

---

# 9. Long / Short候補集合

Long / Short候補の選択ルールは、現行ロジックが存在する場合は原則それを維持する。

最低限、

```text
Long candidate  : Qが相対的に高いAsset
Short candidate : Qが相対的に低いAsset
```

とする。

Long / Short Gross TargetをPOSITION_CAPの下で満たせるだけの候補数が存在することを確認する。

Capacity条件：

\[
N_{Long}\times PositionCap
\ge
LongGross
\]

\[
N_{Short}\times PositionCap
\ge
ShortGross
\]

満たさない場合は、その月をエラーにせず、

```text
CAPACITY_WARNING
```

を出力し、既存の候補拡張ルールを利用する。

勝手に最適化によって反対方向のポジションを作らない。

---

# 10. Residual VolベースのSide Weight

Long側について、

\[
q_i^+
=
\max(Q_i,0)
\]

Short側：

\[
q_i^-
=
\max(-Q_i,0)
\]

とする。

現行のLong / Short候補集合が別途存在する場合は、その集合内だけで計算する。

Raw Weight：

\[
w_{i,L}^{raw}
=
LongGross
\frac{
q_i^+
}{
\sum_j q_j^+
}
\]

\[
w_{i,S}^{raw}
=
-
ShortGross
\frac{
q_i^-
}{
\sum_j q_j^-
}
\]

---

# 11. Position Cap付き再配分

Raw Weightが、

\[
|w_i|>POSITION\_CAP
\]

となった場合はCapする。

Cap超過分は同一Sideの未Cap Assetへ、元の \(q_i\) 比率で再配分する。

LongとShortを別々に処理する。

擬似コード：

```python
def capped_proportional_allocation(
    raw_score,
    gross_target,
    position_cap,
):
    # raw_score >= 0 を想定
    # 1. 比例配分
    # 2. cap超過Assetをcapに固定
    # 3. 未配分Grossを残りAssetへ再配分
    # 4. 全Assetがcap以下になるまで反復
    # 5. gross_targetを達成できない場合はwarning
    ...
```

---

# 12. Net Longごとの完成Portfolio

各候補 \(N\) について、

\[
L=LongGross
\]

\[
S=L-N
\]

を設定する。

そのLong / Short Grossを用いてSection 10～11を実行し、

\[
w(N)
\]

を完成させる。

つまり、

```text
w(0%)
w(5%)
w(10%)
w(15%)
w(20%)
```

という5つの実行可能Portfolioを作る。

ここまでの12アセット相対配分はResidual Volロジックで決定する。

---

# 13. Net Long選択用Expected Return Vector

Net Long候補の比較では、各AssetのExpected Absolute Returnが必要になる。

M0 / M1はRelative、LGB1 / LGB2はAbsoluteなので役割を分ける。

---

## 13.1 Linear Relative Alpha

\[
\alpha_i^{LIN}
=
\frac{
\hat r_{i,M0}+\hat r_{i,M1}
}{2}
\]

断面平均を引く。

\[
\alpha_i^{LIN}
\leftarrow
\alpha_i^{LIN}
-
\frac1{12}
\sum_j\alpha_j^{LIN}
\]

---

## 13.2 LightGBM Market Component

\[
m_t^{ML}
=
\frac12
\left(
\frac1{12}\sum_i\hat r_{i,LGB1}
+
\frac1{12}\sum_i\hat r_{i,LGB2}
\right)
\]

これは12アセット全体の20BD Market Direction Componentとして扱う。

---

## 13.3 LightGBM Relative Alpha

\[
\alpha_i^{ML}
=
\frac12
\left[
(\hat r_{i,LGB1}-\bar r_{LGB1})
+
(\hat r_{i,LGB2}-\bar r_{LGB2})
\right]
\]

---

## 13.4 Expected Return Vector

Linearを厚く利用する。

第一候補：

\[
\boxed{
\hat\mu_i
=
m_t^{ML}
+
0.8\alpha_i^{LIN}
+
0.2\alpha_i^{ML}
}
\]

Config：

```python
LINEAR_EXPECTED_RETURN_WEIGHT = 0.80
ML_RELATIVE_RETURN_WEIGHT = 0.20
```

LightGBMの寄与をさらに下げたい場合は、

```text
90 / 10
```

をSensitivityとして比較可能とする。

Primaryは80 / 20固定とし、毎月動的にウェイト変更しない。

---

# 14. Candidate Portfolio Expected Return

Net候補 \(N\) ごとに、

\[
\hat R_p(N)
=
w(N)^\top\hat\mu
\]

を計算する。

これは20BD Expected Portfolio Return。

---

# 15. Covariance Matrix

Net Longを増加させた場合の市場共通リスクまで評価する必要があるため、Net選択用RiskにはResidual Covarianceではなく **Total Return Covariance** を使用する。

一方、個別Asset Weightingは従来通りResidual Volを使用する。

役割：

\[
\boxed{
\text{Asset Weighting}
=
Residual\ Vol
}
\]

\[
\boxed{
\text{Portfolio Risk Evaluation}
=
Total\ Covariance
}
\]

---

# 16. CovarianceのPrimary推定方法

複雑化を避けるため、第一候補は直近252BDの日次Total Returnを用いる。

安定化のため、可能であればLedoit-Wolf Shrinkageを使用する。

```python
from sklearn.covariance import LedoitWolf

cov_daily = LedoitWolf().fit(asset_daily_returns).covariance_
```

20BD Horizonへ変換：

\[
\Sigma_{20}
=
20\Sigma_{daily}
\]

Primary：

```text
lookback = 252BD
estimator = Ledoit-Wolf
```

比較用としてSample Covarianceも保存可能だが、初期本番ロジックは1種類に固定する。

---

# 17. Candidate Portfolio Risk

各Net候補について、

\[
\sigma_p(N)
=
\sqrt{
w(N)^\top
\Sigma_{20}
w(N)
}
\]

を計算する。

20BD Expected Volatilityとして扱う。

---

# 18. Primary Net Selection：Risk Limit内でExpected Return最大

本番第一候補。

ユーザーまたはConfigで、

```python
PORTFOLIO_VOL_LIMIT_20BD = ...
```

を設定する。

Feasible条件：

\[
\sigma_p(N)
\le
\sigma_{max}
\]

FeasibleなNet候補の中から、

\[
\boxed{
N^*
=
\arg\max_N
\hat R_p(N)
}
\]

を選択する。

説明：

> ユーザー設定Long Grossを維持し、0～20%のNet Long候補ごとに既存のResidual Vol Weightingでポートフォリオを構築する。その後、Total CovarianceでPortfolio Riskを評価し、許容リスク内でExpected Returnが最大となるNet Longを採用する。

---

# 19. Feasible Candidateが存在しない場合

すべての候補がRisk Limitを超える場合、

\[
N^*
=
\arg\min_N\sigma_p(N)
\]

とする。

この場合、

```text
RISK_LIMIT_BREACH = True
```

を出力する。

Long Gross自体はユーザー設定値なので、モデル側で勝手に変更しない。

---

# 20. Tie-break Rule

Expected Returnがほぼ同じ候補が複数ある場合は、より単純・低リスク側を採用する。

第一候補：

1. Expected Return最大
2. 差がTolerance以内ならPortfolio Volが低い方
3. それでも同じならNet Longが小さい方

例：

```python
RETURN_TIE_TOL = 1e-4
```

単位に応じて調整する。

---

# 21. Net Selection比較パターン

Primary以外もChallengerとして実装可能。

## N0：固定Net

```python
NET_LONG = 0.10
```

最も単純なBenchmark。

---

## N1：Expected Return最大

\[
N^*
=
\arg\max_N
\hat R_p(N)
\]

Risk制約なし。

Stress Test / Benchmark。

---

## N2：Expected Sharpe最大

\[
SR(N)
=
\frac{
\hat R_p(N)
}{
\sigma_p(N)
}
\]

\[
N^*
=
\arg\max_N SR(N)
\]

---

## N3：Risk Limit内でExpected Return最大

Primary。

\[
\max_N\hat R_p(N)
\]

subject to

\[
\sigma_p(N)\le\sigma_{max}
\]

---

## N4：Expected Utility

\[
U(N)
=
\hat R_p(N)
-
\lambda\sigma_p^2(N)
\]

\[
N^*
=
\arg\max_N U(N)
\]

\(\lambda\)の説明が必要なため、初期本番候補にはしない。

---

# 22. 推奨比較

最初は以下3つで十分。

```text
N0 : Net 10% Fixed
N2 : Expected Sharpe Max
N3 : Return Max under Risk Limit
```

N3をPrimary Candidateとする。

---

# 23. 最終ウェイト

選択された、

\[
N^*
\]

に対応する、

\[
w(N^*)
\]

を最終ウェイトとする。

必須チェック：

\[
\sum_{i:w_i>0}w_i
=
LongGross
\]

\[
\sum_{i:w_i<0}|w_i|
=
LongGross-N^*
\]

\[
\sum_iw_i
=
N^*
\]

\[
|w_i|\le PositionCap
\]

---

# 24. Final Validation

毎月以下をAssertする。

```python
assert long_gross < 1.0
assert short_gross < 1.0
assert 0.0 <= net_long <= 0.20
assert np.max(np.abs(weights)) <= POSITION_CAP + tolerance
```

Long Gross TargetがCap Capacity上達成不可能な場合はWarningを出す。

---

# 25. 出力テーブル：Asset Weight

```text
prediction_date
asset_id
pred_m0
pred_m1
pred_lgb1
pred_lgb2
score_m0_rank
score_m1_rank
score_lgb1_rank
score_lgb2_rank
score_linear
score_ml
ml_confirmation
score_final
residual_vol
risk_adjusted_score
final_weight
side
```

---

# 26. 出力テーブル：Net Candidate

各Prediction Dateで以下を保存する。

```text
prediction_date
net_long_candidate
long_gross
short_gross
expected_return_20bd
expected_vol_20bd
expected_sharpe
risk_limit_pass
selected_flag
```

これにより、

> なぜその月にNet Long 15%を選んだのか

を後から説明可能にする。

---

# 27. 月次Decision Summary

```text
Prediction Date:
Long Gross User Setting:
Net Selection Method:
Net Candidates:
Selected Net Long:
Selected Short Gross:

Expected Return:
Expected Vol:
Expected Sharpe:
Risk Limit:

Linear Weight in Expected Return:
ML Relative Weight:

Position Cap:
Capacity Warning:
Risk Limit Breach:
```

---

# 28. 推奨Config

```yaml
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

validation:
  return_tie_tolerance: 0.0001
```

`portfolio_vol_limit_20bd` はユーザー設定または既存Risk Budgetから入力する。

---

# 29. 推奨コード構成

大規模なファイル分割は不要。

既存分析コードへ追加する場合は以下程度でよい。

```text
portfolio.py
run_portfolio.py
```

既存ファイルに統合可能なら1ファイルでもよい。

---

# 30. portfolio.py の主要関数

```python
def make_cross_sectional_rank_score(pred):
    pass


def make_linear_core_score(pred_m0, pred_m1):
    pass


def make_ml_score(pred_lgb1, pred_lgb2):
    pass


def apply_ml_confirmation(score_linear, score_ml):
    pass


def residual_vol_adjust(score, residual_vol):
    pass


def capped_side_allocation(
    side_score,
    gross_target,
    position_cap,
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
    pred_m0,
    pred_m1,
    pred_lgb1,
    pred_lgb2,
    linear_weight=0.80,
    ml_weight=0.20,
):
    pass


def estimate_total_covariance(
    daily_returns,
    lookback=252,
):
    pass


def evaluate_net_candidate(
    weights,
    expected_return,
    covariance_20bd,
):
    pass


def select_net_long(
    candidate_table,
    method,
    risk_limit=None,
):
    pass
```

---

# 31. 実装順序

## Step 1
4モデル予測値を読込。

## Step 2
Linear / MLのRank Score作成。

## Step 3
Linear Core / ML ConfirmationでFinal Score作成。

## Step 4
Residual VolでRisk-adjusted Score作成。

## Step 5
ユーザー設定Long Grossを読込。

## Step 6
Net候補ごとにShort Gross算出。

## Step 7
Net候補ごとにResidual Volベースの完成Portfolioを作成。

## Step 8
Position Cap適用。

## Step 9
Expected Return Vector作成。

## Step 10
Total Covariance推定。

## Step 11
各候補PortfolioのExpected Return / Vol / Sharpe算出。

## Step 12
Net Selection Ruleで1候補を選択。

## Step 13
最終ウェイト・Decision Summary保存。

---

# 32. 初期実験パターン

以下を比較する。

### P0
Linear Core / ML Confirmation  
+ Net 10%固定

### P1
Linear Core / ML Confirmation  
+ Expected Sharpe最大Net

### P2
Linear Core / ML Confirmation  
+ Risk Limit内Expected Return最大Net

Primary Candidate：

\[
\boxed{P2}
\]

とする。

---

# 33. 評価指標

Portfolio Construction比較は最低限以下。

```text
20BD Realized Return
Annualized Return
Volatility
Sharpe
Max Drawdown
Turnover
Average Net Long
Net Long Distribution
Average Long Gross
Average Short Gross
Position Cap Hit Rate
Risk Limit Breach Rate
```

さらに、

```text
Net 0%
Net 5%
Net 10%
Net 15%
Net 20%
```

の選択頻度を出す。

---

# 34. Look-ahead禁止

Net Longを、

> その後実際に最もリターンが高かった候補

から選んではならない。

Candidate Selectionは必ずPrediction Date時点で利用可能な、

- モデル予測値
- Residual Vol
- 過去Return Covariance
- Risk Limit

だけで行う。

Realized ReturnはOOS評価にのみ使用する。

---

# 35. 初期実装でやらないこと

以下はPhase 2。

```text
12Asset Weightのフル平均分散最適化
Turnover Penalty
Dynamic Model Weight
Dynamic Linear / ML Weight
Black-Litterman
Risk Parityへの全面変更
Residual Covarianceによる直接最適化
Net候補を1%刻みにする
```

まず5つのNet候補だけで十分。

---

# 36. 最終思想

本実装では、

\[
\boxed{
\text{Asset Selection / Sizing}
=
Linear\ Core
+
ML\ Confirmation
+
Residual\ Vol
}
\]

と、

\[
\boxed{
\text{Net Exposure Selection}
=
Expected\ Return
+
Total\ Covariance\ Risk
}
\]

を分離する。

つまり、既存のResidual Volベースの資産別ウェイト設計を維持しながら、上位レイヤーでのみ分散共分散を使用する。

フルOptimizerではなく、

\[
\boxed{
NetLong\in\{0,5,10,15,20\%\}
}
\]

という少数の完成Portfolioから最適候補を選択する。

Primaryロジックは、

\[
\boxed{
\text{Risk Limit内でExpected Return最大}
}
\]

とする。

これにより、

- Linearモデル中心の説明可能性
- LightGBMの影響限定
- 現行Residual Vol Weightingの維持
- Long Grossのユーザー設定
- Net Longの動的調整
- Total CovarianceによるPortfolio Risk考慮
- ±33%未満の個別Position Cap

を同時に満たす。

以上。
