# 第3層の推定範囲仕様

## 1. country_independent

各国を完全に分け、国 `c` ごとに独立したRidgeを推定します。

$$
\widehat\beta_c=\arg\min_\beta\sum_{i,t\in c}(r_{i,t+1}-X_{i,t}\beta)^2+\lambda\|\beta\|_2^2
$$

国別構造を最も直接反映できますが、特徴量数を抑える必要があります。

## 2. regional_pooling

地域内の国をまとめて推定します。国ダミーを統制変数として入れます。

$$
r_{i,t+1}=X_{i,t}\beta_{region}+CountryDummy_i\gamma+\varepsilon_{i,t+1}
$$

国別係数を共通化するため、推定が安定しやすくなります。

## 3. hierarchical_partial_pooling

国別係数を地域係数と国固有補正に分解します。

$$
\beta_c=\beta_{region(c)}+\Delta\beta_c
$$

国固有補正には強いRidge罰則を付けます。

$$
\min \sum(r-\widehat r)^2+\lambda_R\|\beta_{region}\|_2^2+\lambda_C\sum_c\|\Delta\beta_c\|_2^2
$$

通常は `lambda_C > lambda_R` とし、国固有情報が弱い場合は地域係数へ縮小します。

## 4. 学習方式

### rolling_pooled

過去L期間を縦に積んで一つの係数を推定します。本番候補です。

### cross_sectional_coefficient_average

各月の断面回帰係数を推定し、過去係数を平均します。係数の時系列安定性を診断する用途です。
