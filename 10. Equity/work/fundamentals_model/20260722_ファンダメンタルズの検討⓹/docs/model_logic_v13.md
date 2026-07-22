# v0.13.1 モデルロジック

## 1. Direct Raw Factor Score

FA kの前処理後値をxとします。

初期本線では、月次横断面のCentered Percentileを使用します。

$$
s_{i,k,t}=2\frac{\operatorname{rank}_{i,k,t}-0.5}{N_t}-1
$$

Direction=-1の場合は順位方向を反転します。

## 2. Q5-Q1 Factor Return

FA kについて、各国内でQ5とQ1を形成します。

$$
FR_{k,t+1}=R_{k,Q5,t+1}-R_{k,Q1,t+1}
$$

各国のFactor Returnを等ウェイト平均してグローバル系列を作ります。

## 3. 相関調整ウェイト

同一Factor Group hに属するFAの過去Factor Return相関行列をCとします。

$$
\widetilde C=(1-\delta)C+\delta I
$$

相関最小分散型のRaw Weightは概念的に、

$$
w^{Corr}\propto \widetilde C^{-1}\mathbf 1
$$

とし、負値を0へ制限し、最大ウェイトを適用します。

最終ウェイトはEqual Weightへ縮小します。

$$
w^{Final}=\eta w^{EW}+(1-\eta)w^{Corr}
$$

さらに前月ウェイトとの平滑化を行います。

## 4. Aggregate FactorScore

$$
F_{i,h,t}=\sum_{k\in h}w_{k,h,t}s_{i,k,t}
$$

## 5. 第3層

### N05

$$
r_{i,t+1}=\alpha_c+\sum_h\beta_{c,h}F_{i,h,t}+\varepsilon_{i,t+1}
$$

### N06

N05と同じ説明変数にRidge正則化を加えます。

### N07

$$
r_{i,t+1}=\alpha_c+\sum_h\beta_{c,h}F_{i,h,t}+\sum_{q,h}\theta_{c,q,h}D_{i,q}F_{i,h,t}+\varepsilon_{i,t+1}
$$

セクター主効果D単体は既定で含めません。
