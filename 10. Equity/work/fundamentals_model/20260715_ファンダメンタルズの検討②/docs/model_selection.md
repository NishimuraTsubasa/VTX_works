# 単一ファクターモデル選択

## 候補モデル

- Linear
- Piecewise
- Quadratic
- Combined Ridge

## 主指標

各テスト時点のクロスセクション予測値と翌期個別銘柄リターンのSpearman RankICを計算し、その時系列平均を使用します。

## 1-SE rule

1. 平均OOS RankICが最大の候補を `best_raw_model` とする。
2. `one_se_threshold = best_primary_metric - one_se_multiplier × best_standard_error` を計算する。
3. 閾値以上の候補を `within_one_se=True` とする。
4. eligible候補の中で最小複雑度を選ぶ。
5. 同じ複雑度なら平均OOS RankICが高い候補を選ぶ。

## Linearが選ばれる場合

- Linearが最良指標そのものだった場合：`BEST_OOS_METRIC`
- 非線形候補が最良でもLinearが1-SE閾値以上の場合：`ONE_SE_SIMPLER_MODEL`

## 主要列

| 列 | 内容 |
|---|---|
| `best_raw_model` | 平均OOS RankICが最大の候補 |
| `selected_model` | 1-SE ruleと複雑度を考慮した採用モデル |
| `best_primary_metric` | 最良候補の平均OOS RankIC |
| `selected_primary_metric` | 採用候補の平均OOS RankIC |
| `best_standard_error` | 最良候補のRankIC標準誤差 |
| `one_se_threshold` | ほぼ同等候補の下限 |
| `selected_delta_from_best` | 最良候補との差 |
| `selection_reason_jp` | 採用理由 |
| `adopted` | 採用ゲートを通過したか |
