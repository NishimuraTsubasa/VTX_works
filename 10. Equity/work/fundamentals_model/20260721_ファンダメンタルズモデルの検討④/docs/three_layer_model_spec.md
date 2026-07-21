# 3層個別銘柄スコアリングモデル仕様

## 第1層：グローバル単一ファクター

時点 `t` のファクター `k` の前処理値を `z(i,k,t)`、翌月総リターンを `r(i,t+1)` とします。

### Linear

$$
r_{i,t+1}=\alpha_k+\beta_k z_{i,k,t}+\varepsilon_{i,t+1}
$$

### Piecewise

$$
r_{i,t+1}=\alpha_k+\beta_{1,k}z_{i,k,t}+\beta_{2,k}(z_{i,k,t}-\kappa)_++\varepsilon_{i,t+1}
$$

### Quadratic

$$
r_{i,t+1}=\alpha_k+\beta_{1,k}z_{i,k,t}+\beta_{2,k}z_{i,k,t}^2+\varepsilon_{i,t+1}
$$

候補は過去学習期間と検証期間で比較し、平均OOS RankICと1-SEルールで選びます。

$$
SubScore_{i,k,t}=\widehat f_{k,t}(z_{i,k,t})
$$

過去データへ付与するSubScoreは必ずWalk-forward OOFで生成します。

## 第2層：FactorScore集約

Factor_Group `h` に属するFA集合を `K_h` とします。

$$
FactorScore_{i,h,t}=\sum_{k\in K_h}w_{k,h,t}SubScore_{i,k,t}
$$

選択可能な集約方式：

- `equal_weight`
- `manual`
- `ic_adjusted`
- `pca`

## 第3層：国別・地域別の最終予測

FactorScore `F(i,h,t)` から次の基底を作ります。

$$
B(F)=\left[F,\ (F-\kappa)_+,\ F^2\right]
$$

セクターグループダミーを `D(i,q)` とすると、基本式は次です。

$$
\begin{aligned}
r_{i,t+1}={}&\alpha_t+\sum_h\boldsymbol\beta_h^\top B(F_{i,h,t})
+\sum_q\delta_qD_{i,q}\\
&+\sum_h\sum_q\boldsymbol\theta_{h,q}^\top\left[D_{i,q}B(F_{i,h,t})\right]+\varepsilon_{i,t+1}
\end{aligned}
$$

Ridge回帰で係数を縮小し、Walk-forwardで翌月予測を作ります。

## 最終スコア

第3層予測値を国別またはグローバルでPercentile Rankへ変換します。

$$
TotalScore_{i,t}=PercentileRank\left(\widehat r_{i,t+1}\right)
$$

既定は国別順位です。これにより国別モデルの切片差による比較困難を避けます。
