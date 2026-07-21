# Time-Averaged Binscatter仕様

## 1. 目的

単一FAと翌期個別銘柄リターンの関係を、外れ値に左右されにくいビン平均で可視化します。

分析スコープ:

- 全銘柄
- 国別
- 国 x セクター別

## 2. 時点ごとのビン化

時点`t`・スコープ内でFAスコアを等頻度ビンへ分割します。

$$
b_{i,t}\in\{1,\ldots,Q\}
$$

各時点・各ビンで平均FAスコアと翌期平均リターンを計算します。

$$
\bar{x}_{b,t}=\frac{1}{N_{b,t}}\sum_{i\in b}x_{i,t}
$$

$$
\bar{r}_{b,t+1}=\frac{1}{N_{b,t}}\sum_{i\in b}r_{i,t+1}
$$

## 3. Time-average

$$
\bar{x}_b=\frac{1}{T_b}\sum_t\bar{x}_{b,t}
$$

$$
\bar{r}_b=\frac{1}{T_b}\sum_t\bar{r}_{b,t+1}
$$

## 4. エラーバー

標準誤差:

$$
SE(\bar{r}_b)=\frac{Std_t(\bar{r}_{b,t+1})}{\sqrt{T_b}}
$$

95%信頼区間を選ぶ場合:

$$
CI95_b=1.96\times SE(\bar{r}_b)
$$

## 5. 回帰

### Linear

$$
\bar{r}_b=\alpha+\beta\bar{x}_b+\varepsilon_b
$$

### Quadratic

$$
\bar{r}_b=\alpha+\beta_1\bar{x}_b+\beta_2\bar{x}_b^2+\varepsilon_b
$$

### Broken-stick

$$
\bar{r}_b=\alpha+\beta_1\bar{x}_b+\beta_2(\bar{x}_b-\kappa)_++\varepsilon_b
$$

## 6. 図中表示

- FactorCode
- Scope
- 期間数
- ビン数
- 総観測数
- Pearson(bin)
- Spearman(bin)
- Linear R2
- Quadratic R2
- Broken-stick R2
- Broken-stick knot
- Top-Bottomビン差

## 7. R2の解釈

R2はtime-average後のビン点への適合度です。個別銘柄レベルの予測力やOOS性能を直接表すものではありません。

したがって、FA採用では次を併用します。

- R2
- Spearman(bin)
- Top-Bottom差
- 分位単調性
- 月次RankIC
- 5分位ポートフォリオ
- 国・セクター間の安定性

## 8. Broken-stickの注意

`auto` knotは同じビン点でR2最大となる点を選ぶため、Linearより有利です。診断図として利用し、最終モデル採用はOOS検証で判断します。
