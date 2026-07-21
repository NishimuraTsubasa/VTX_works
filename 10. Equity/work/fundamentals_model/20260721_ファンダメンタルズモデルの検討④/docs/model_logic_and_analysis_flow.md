# モデルロジックと分析フロー

```text
入力データ
  ├─ Raw FA
  ├─ 派生FA（差分・移動平均乖離等）
  ├─ 翌月総リターン
  ├─ 国・セクター・時価総額
  └─ factor_master設定
        ↓
前処理
  ├─ 方向統一
  ├─ Winsorize
  └─ クロスセクション順位正規化
        ↓
第1層：グローバル単一FA
  ├─ Linear
  ├─ Piecewise
  └─ Quadratic
        ↓ Walk-forward OOF
SubScore（FA単位）
        ↓
第2層：Factor_Group集約
  ├─ Equal Weight
  ├─ Manual
  ├─ Correlation-adjusted IC
  └─ PCA
        ↓
FactorScore（Value / Momentum / Quality等）
        ↓
第3層
  ├─ 国別独立
  ├─ 地域プール
  └─ 地域＋国部分プーリング
        + 非線形基底
        + Sector Group Dummy
        + Sector Group × FactorScore
        ↓
翌月総リターン予測
        ↓
TotalScore
        ↓
RankIC・5分位・Q5-Q1・係数安定性で評価
```
