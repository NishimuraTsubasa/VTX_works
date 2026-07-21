# 共通OOS評価とS07推定方式

## 1. 共通OOSの定義

シナリオ集合を \(\mathcal M\) とし、シナリオ \(m\) が有効な予測を持つ `Date × ISIN` 集合を \(K_m\) とします。

$$
K_{\mathrm{common}}
=
\bigcap_{m\in\mathcal M}K_m
$$

共通OOS評価では、\(K_{\mathrm{common}}\) に含まれる行だけを使用します。そのうえで、各月・各順位付け単位についてスコアを再順位化します。

$$
S_{i,t,m}^{\mathrm{common}}
=
\operatorname{PercentileRank}_{K_{\mathrm{common}},t}
\left(\widehat r_{i,t+1,m}\right)
$$

これにより、S03、S06、S07間で、期間差とユニバース差を除いた比較が可能です。

## 2. 2018年開始時の時系列

初期設定では、第1層は過去18か月以上でOOF予測を開始します。第3層は、そのOOF FactorScoreが12か月蓄積した後に予測を開始します。

```text
2018-01                 元データ開始
       ├─ Layer1最低18か月
2019年後半頃            Layer1 OOF SubScore開始
       ├─ Layer3最低12か月
2020年半ば頃            S07 OOS予測開始の目安
```

入力欠損、FAカバレッジ、国別銘柄数により実際の日付は変化します。日付はコードが自動判定し、`analysis_summary.xlsx` に出力します。

## 3. S07 OLS

$$
\widehat{\boldsymbol\beta}^{\mathrm{OLS}}
=
\arg\min_{\boldsymbol\beta}
\sum_{i,t}
\left(
 r_{i,t+1}-\boldsymbol x_{i,t}^{\top}\boldsymbol\beta
\right)^2
$$

第3層の説明変数は、FactorScoreの線形項、セクターグループダミー、選択した交差項です。

## 4. S07 Ridge

$$
\widehat{\boldsymbol\beta}^{\mathrm{Ridge}}
=
\arg\min_{\boldsymbol\beta}
\left[
\sum_{i,t}
\left(
 r_{i,t+1}-\boldsymbol x_{i,t}^{\top}\boldsymbol\beta
\right)^2
+
\alpha\lVert\boldsymbol\beta\rVert_2^2
\right]
$$

OLSと同じ説明変数を使用するため、結果差は主として正則化によるものです。

## 5. 比較項目

- 共通OOS Mean RankIC
- 共通OOS ICIR
- S03対比RankIC差
- S03を上回った月割合
- Q5-Q1累積リターン
- 係数の時系列標準偏差
- 係数符号反転率
- Ridge alphaの選択履歴

RidgeのMean RankICがOLSを上回らなくても、係数安定性やドローダウンが改善する可能性があります。逆にOLSが安定して優位なら、現状の説明変数数・相関構造では正則化が不要という示唆になります。
