# モデルロジックと分析フロー

## 1. 分析目的

現行の個別銘柄スコアから、前処理・階層化・予測モデル・OOF統合を順番に追加し、どの要素が個別銘柄順位予測の改善に寄与したかを確認します。

すべてのパターンで、翌期リターン、5分位評価、RankICの計算方法は固定します。

## 2. 基準となるファクター順位

各FAを時点内で0-1順位化します。

$$
p_{i,k,t}=\frac{\operatorname{rank}(x_{i,k,t})-0.5}{N_t}
$$

Directionにより、高い値が望ましい方向へ統一します。

## 3. 8つの分析パターン

### S00: Current Direct Equal Weight

欠損を中立値0.5で補完し、全ファクター数を固定して平均します。

$$
S_{i,t}^{S00}=\frac{1}{K}\sum_{k=1}^{K}\widetilde p_{i,k,t}
$$

### S01: Missing Adjusted Equal Weight

欠損FAを除き、銘柄ごとに利用可能FAだけで再正規化します。

$$
S_{i,t}^{S01}=\frac{\sum_k A_{i,k,t}p_{i,k,t}}{\sum_k A_{i,k,t}}
$$

### S02: Winsorized Direct Equal Weight

生FAを1%-99%点でクリップした後に順位化し、直接等ウェイトします。

$$
x_{i,k,t}^{win}=\min\{\max(x_{i,k,t},q_{0.01}),q_{0.99}\}
$$

### S03: Neutralized Direct Equal Weight

国・セクター・対数時価総額を回帰で除去し、残差を順位化します。

$$
x_{i,k,t}^{win}=a_{k,t}+Country_i+Sector_i+\beta_{k,t}\log(MarketCap_{i,t})+\varepsilon_{i,k,t}
$$

### S04: Hierarchical Equal Weight

FAをValue、Momentum等のグループへ分け、グループ内とグループ間をそれぞれ等ウェイトします。

$$
G_{i,h,t}=\frac{1}{|K_h|}\sum_{k\in K_h}z_{i,k,t}
$$

$$
S_{i,t}^{S04}=\frac{1}{H}\sum_{h=1}^{H}G_{i,h,t}
$$

### S05: Correlation Adjusted IC

過去OOS RankICが高く、他FAと重複しにくいFAを重視します。履歴不足時は等ウェイトへフォールバックします。

$$
\widehat{\boldsymbol w}_{h,t}\propto \widetilde\Sigma_{h,t}^{-1}\boldsymbol\mu_{h,t}
$$

### S06: Selected Factor Models

元FAと設定された派生FAについて、次の4候補を時間順OOSで比較します。

- Linear
- Piecewise
- Quadratic
- Combined Ridge

最良平均OOS RankICから1標準誤差以内の候補のうち、最も単純なモデルを選択します。

### S07: Full OOF Ridge

S06で作成した各グループ予測を説明変数とし、過去のOOF予測だけを使うRidgeで最終予測を作成します。

$$
r_{i,t+1}=\gamma_0+\sum_h\gamma_h\widehat r_{i,h,t+1}^{OOF}+\varepsilon_{i,t+1}
$$

## 4. 派生特徴量

### 差分

$$
\Delta x_{i,k,t}^{(P,L)}=x_{i,k,t-L}-x_{i,k,t-L-P}
$$

### 移動平均乖離

$$
MADev_{i,k,t}^{(W,L)}=x_{i,k,t-L}-\frac{1}{W}\sum_{j=1}^{W}x_{i,k,t-L-j}
$$

### 移動平均比率

$$
MARatio_{i,k,t}^{(W,L)}=\frac{x_{i,k,t-L}}{MA_{i,k,t}^{(W,L)}}-1
$$

### 過去平均乖離

$$
ExpDev_{i,k,t}^{(L)}=x_{i,k,t-L}-\operatorname{Mean}_{\tau<t-L}(x_{i,k,\tau})
$$

## 5. 5分位評価

各月にTotalScoreで5分位へ分割します。

$$
R_{q,t+1}=\frac{1}{N_{q,t}}\sum_{i\in q}r_{i,t+1}
$$

ロング・ショートは次です。

$$
R_{LS,t+1}=R_{Q5,t+1}-R_{Q1,t+1}
$$

## 6. RankIC

$$
RankIC_t=Corr^{Spearman}(TotalScore_{i,t},r_{i,t+1})
$$

## 7. 比較する指標

- Q1-Q5累積リターン
- Q5-Q1累積リターン
- 平均RankIC
- RankIC IR
- RankIC正符号率
- 分位単調性
- Q5-Q1 Sharpe
- 最大ドローダウン
- S05のFAウェイト安定性
- S06/S07の採用モデルと1-SE選択理由

## 8. 分析順序

```text
S00 現行モデル
 -> S01 欠損ウェイト再調整
 -> S02 外れ値処理
 -> S03 中立化
 -> S04 階層等ウェイト
 -> S05 相関調整IC
 -> S06 単一FA回帰モデル選択
 -> S07 OOF Ridge統合
```

前段階との差分と、S00からの累積改善の両方を確認します。
