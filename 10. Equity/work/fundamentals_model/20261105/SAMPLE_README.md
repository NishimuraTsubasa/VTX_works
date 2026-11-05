# 単一Pythonファイルの実行・接続ガイド

対象：`value_gm_research.py` v1.0.0。全処理はこの1ファイルにまとまっています。元の `factor_tilt_research.py` をimportする必要はありません。Python 3.10以降、依存パッケージはNumPy・pandas・SciPyです。コードはネット接続・API送信・自動発注を行いません。

## 1. 最初に動かす

```bash
python -m pip install numpy pandas scipy
python value_gm_research.py self-test
python value_gm_research.py demo --out demo_results
```

デモは**合成の13ファクター・161か月**です。Value 7本／Growth Momentum 6本という割り当てはテスト用で、ユーザーの実際の13本の内訳を推定したものではありません。デフォルトでは、12か月のトレンドを作るウォームアップ11か月と学習ペア60か月を確保し、**89か月のOOS、7配分系列**を比較します。入力末尾までの161行のうち、最後の行は予測対象リターンで、そこから先の未知リターンは生成しません。

`sample_demo/` は同梱したデモの実行記録です。実際の運用成績ではありません。既存フォルダが空でない場合は停止します。再実行は新しい出力先を指定してください。

## 2. 比較する7系列

| ID | 実装 | 注意点 |
|---|---|---|
| B0 | ValueとGMの固定目標配分 | デフォルト各50%。毎月戻すのでbuy-and-holdではない |
| B1 | 共通学習窓の過去平均スプレッドを配分へ入力 | 特性なしの比較対象 |
| A1 | 個別ファクター予測を固定グループで集約して引き算 | 実データでは保存済み予測を使う |
| A2 | A1の過去平均からの差を半分にする | 水準全体を半分にはしない |
| C1 | Value−GMを、相対特徴量から直接Ridge予測 | 少数の固定特徴量・固定lambda |
| C2 | 各グループを自身の特性でRidge予測して引き算 | C1とは入力設計も違う |
| D1 | 割安度／ファンダメンタルズ／トレンド各モデルの予測平均 | 全成分は同じスプレッドを予測 |

**デモのA1は、固定lambdaのサンプルlocal Ridgeです。** 現行ユーザーモデルや、旧版のPLSモデルを再現したという意味ではありません。実データで現行A1を維持するには `core_mode="saved"` として、実際に生成済みの `predictions.csv` を渡します。

D1の3成分の予測は保存しますが、別々の売買戦略には自動展開しません。P3の状態空間モデル・校正係数推定・残差補正・相関クラスタリング／木による選択器は、このサンプルでは実装していません。

## 3. 実データ用のひな形

```bash
python value_gm_research.py templates --out my_inputs
```

出力は `runtime_config.json`、`returns.csv`、`features.csv`、`group_membership.csv` です。CSVはヘッダーだけで、架空の13本を実データとして埋めてはいません。

### 3.1 returns.csv

```csv
date,VALUE_FACTOR_1,GM_FACTOR_1
2013-06-30,0.01,-0.005
```

例の1行は形式説明です。各列は1ファクターの実現月次リターンで、1%は0.01。月末日、連続月次、重複なし。構成因子のNaN/infを拒否します。構成外の列を入力しても売買対象に加えません。原因子のリターン列を渡し、符号調整が必要ならmembershipのorientationに一度だけ記録します。

### 3.2 group_membership.csv

```csv
group,factor_id,weight,return_orientation,membership_version,definition_decided_at
VALUE,VALUE_FACTOR_1,1.0,1,my_v1,2026-11-05
GROWTH_MOMENTUM,GM_FACTOR_1,1.0,1,my_v1,2026-11-05
```

これは2因子の**形式例**です。実際の固定構成・ウェイトへ置き換えてください。groupは `VALUE` または `GROWTH_MOMENTUM`、各グループのweight合計1、factor_idの重複は禁止。所属・本数はコードで決めつけません。定義日がOOSより後でも処理できますが、過去OOSに対する事前登録とは呼べません。

`return_orientation=-1` は `factor_pnl` だけで利用できます。`sleeve_total`で商品リターンに−1を掛けて逆向き商品を作ることは禁止します。前処理済みで符号を反転済みならorientationは+1にしてください。

### 3.3 features.csv

```csv
date,factor,available_at,x_value_spread,x_profitability_change
2013-06-30,VALUE_FACTOR_1,2013-06-28,0.8,0.02
2013-06-30,GM_FACTOR_1,2013-06-28,0.4,0.01
```

全月×グループ構成因子の行が必要です。dateは意思決定スナップショット月末であり、財務決算期ではありません。列の経済的意味・単位・向きをそろえた**未標準化の特性**を渡します。

各グループ内で特性を固定ウェイト集約し、次に学習内で欠損補完・winsorize・標準化します。ある構成因子の特性が欠けたとき、残りのウェイトを再正規化しません。グループ特性を欠損として保持し、学習内中央値補完に回します。リターンの符号を特性にも無条件で掛ける処理はありません。

`available_at`はその行の情報が使えた日時。月末終日より未来なら拒否します。不明日時の許可は `allow_unverified_pit=true` を明示した場合だけですが、許可後もPIT未確認と記録します。日時チェックは改訂履歴を保証しません。グループ内の財務指標の意味が異なるとき、列名だけそろえて平均しないでください。

### 3.4 保存済み predictions.csv

必要列は `decision_date, target_date, model, factor, prediction`。旧コードのCSVをそのまま読み込める形です。予測対象の実現リターンではなく、**その時点で作った月次リターン予測**を渡します。

`target_date`はdecisionの翌月末。設定の `core_source_model` で1モデルを選びます。選んだモデルについて、共通OOSの全構成因子の予測が必要です。欠けた因子だけ平均から外すことはしません。`train_end`列がある場合は意思決定時点を超えていないかも検査します。

外部予測が、全期間で選んだ特徴量・モデルを使っていないかまでは、このファイル単独では確認できません。選択履歴を別途監査してください。共通の `historical_mean` は生のスプレッドリターンから計算し、保存CSVのモデルごとに異なる過去平均を混ぜません。

## 4. runtime_config.json の設定

`templates/config_spec.json` は人間向けの全体仕様であり、**実行には使いません**。CLIに渡すのは、`templates`コマンドまたは同梱の `sample_runtime_templates/runtime_config.json` で生成したruntime形式です。

```json
{
  "dataset_kind": "real",
  "settings": {
    "core_scope": "prediction_targets",
    "core_factor_ids": ["実際のID1", "実際のID2"],
    "core_mode": "saved",
    "core_source_model": "実際の予測CSVのmodel名",
    "feature_orientation_confirmed": true,
    "return_kind": "factor_pnl",
    "feature_columns": {
      "VALUATION": ["x_value_spread"],
      "FUNDAMENTAL": ["x_profitability_change"]
    }
  },
  "inputs": {
    "returns": "returns.csv",
    "features": "features.csv",
    "membership": "group_membership.csv",
    "predictions": "predictions.csv"
  }
}
```

未指定の設定はサンプルのデフォルト値になります。相対パスはJSONの場所から解決します。core_scopeは `prediction_targets / information_sources / tradable_constituents / mixed` のいずれかで、13本の役割を実際に確認して指定します。上の2要素のIDは形式例であり、本数制約ではありません。`unresolved`や空のcore_factor_idsでは実データ実行を止めます。

`feature_columns`は実在する同一定義の列を、各ブロック最大2つ指定します。TRENDは既知の直近12か月グループリターンの算術合計として自動生成するので列指定は不要です。C1の説明変数はグループ差、C2は各グループの水準。最初はnew Ridge lambda=1.0固定、グリッド選択なしです。

```bash
python value_gm_research.py run --config my_inputs/runtime_config.json --out actual_results_001
```

`core_mode="sample_local_ridge"`を明示すれば、保存済み予測なしでサンプルの個別Ridgeを動かせます。ただし、これを現行13本モデルの再現と報告しないでください。`core_mode="none"`はA1/A2なしの部分比較です。

### 入力が不足する場合

- features全体がない場合、C1/C2/D1はnot_runになります。A1/A2/B0/B1は保存済み予測とリターン・所属表があれば実行できます。
- 片方の財務情報ブロックだけ欠ける場合、C1/C2は利用可能列＋トレンドで実行し、D1だけnot_runになります。使用列をログで確認してください。
- 保存済み予測がない場合、saved設定のA1/A2はnot_runです。別モデルへ自動代替しません。
- 辞書の誤対応、未来の利用可能日時、月抜け、構成因子の欠落等はエラーです。`FAILED.json`を残します。

実行可能なモデル間で共通の連続OOSを使います。モデルがそろわないランと全7系列がそろうランで期間が異なり得るため、違う期間の指標を直接順位比較しないでください。

## 5. 予測を再学習せず、配分だけ再計算

```bash
python value_gm_research.py reallocate --config my_inputs/runtime_config.json --spread-predictions actual_results_001/spread_predictions.csv --out allocation_results_002
```

この入口は配分・費用の比較用です。membership、対象リターン、予測期間を維持してください。配分設定だけを変えます。学習設定を変えても予測は再fitされません。予測CSVのtargetと生リターンから作るスプレッドの一致を確認しますが、元予測の完全な生成履歴までは検証しません。

初期比較で配分の設定まで大量に探索しないでください。費用感応度は主設定ポジションを固定して0/10/25bpsに差し替えるもので、費用ごとの再最適化ではありません。

## 6. 数式と実装の対応

共通のリターン差は `D = R_VALUE - R_GROWTH_MOMENTUM`。学習は月末tまでのラベル、予測はt+1です。

```text
A2: mean + 0.5 * (A1_after_common_clip - mean)
C1: Ridge(group_feature_difference -> next_month_D)
C2: Ridge(value_features -> next_value_return)
    - Ridge(gm_features -> next_gm_return)
D1: mean(raw forecast from VALUATION, FUNDAMENTAL, TREND)
```

D1は月次リターン単位へ戻したraw予測を平均し、その後一度だけ共通クリップをします。A1は保存済みのモデル固有の処理を含み得るため、その上流処理は別に記録します。A2はA1の共通クリップ後に縮小し、再び半分を掛けません。旧版の `prediction_shrinkage=0.5` はこの専用配分に持ち込みません。

配分は `q = (0.5+b, 0.5-b)` を基本に、固定構成行列でファクター配分へ変換します。目的関数は指示書第9節の通り。二次項＋L1の1次元凸問題の全折点と区間内停留点を調べる解法を実装しており、最適化失敗時に等ウェイトへ黙って戻す設計ではありません。

`factor_pnl`では配分変更の費用proxy、`sleeve_total`では費用後NAVを考慮する自己資金制約と保有比率ドリフトを使います。後者でも、原資産内部回転・借株等を再現するものではありません。

## 7. 出力の読み順

まず `experiment_registry.csv` で実行／not_runを確認し、`audit_manifest.json` で入力・設定・履歴を確認します。その後 `prediction_metrics.csv` のALL行、`strategy_comparison.csv` のbaseline=B1を読みます。

| 出力 | 用途 |
|---|---|
| spread_predictions.csv | 6予測系列、共通過去平均、raw/処理後、実現値、学習時点 |
| component_predictions.csv | D1を構成する3モデルと使用列 |
| strategy_returns.csv / weights.csv | 7配分系列のgross/net/費用/配分/寄与 |
| prediction_metrics.csv | 全期間・年別・前後半・24か月窓 |
| strategy_comparison.csv | B1/B0との差を直接使うHAC・IR |
| multiple_testing.csv | 主比較5件のHolm。過去の全探索を補正したものではない |
| bootstrap_comparison.csv | 同じ月ブロックでのペア再標本化区間 |
| cost_sensitivity.csv | 同一ポジションで費用率だけ変更 |
| nav_paths.csv | 各NAVとB1への相対NAV |
| diagnostics.csv | 欠損・非活性列・外れ値等 |
| REPORT.md | 概要と重要な制約 |

2グループの横断順位ICは主指標にしません。Dの予測改善と同じ配分ルールでのB1比追加損益を別々に判断します。`mean_vol_ratio_rf0`は無リスク金利ゼロの比率です。期間別の成績や費用まで見ず、最良値だけでモデルを採用しないでください。

## 8. テストと未実装の区別

`self-test`には39の自動チェックがあります。未来変更・prefix不変性、縮小の端点、Ridgeの線形性、配分恒等式・制約、初期費用、total-returnドリフト、HAC・Holm・ペアbootstrap、保存予測の欠落、上書き防止などを検査します。

別紙 `ACCEPTANCE_TESTS.md` は**実プロジェクトの検収要件**です。その全項目について実データ接続を検証済みという意味ではありません。特に、現行13本の意味・選定履歴、古いコードの動作保存、実際の公表日時、P3のQ=0等は、実装担当者側で別途確認する必要があります。

不足しているのはコードが実行できるためのデータではなく、**ユーザーの実際の13本・分類・特性定義・既存予測への対応付け**です。合成データのCSVで代用して、実データの結果を作らないでください。

## 9. 分割する際の境界

SECTION 00→config、01→入力／グループ、02→前処理／モデル、03→予測、04→配分／費用、05→評価、06→レポート、07→demo／tests、08→CLI。

プログラムからも `build_group_panel()` → `run_experiment()` → `export_results()` の順で利用できます。`Result.tables`にDataFrameを保持します。予測と配分を分けているため、次の担当者が比較処理を追加する際も、予測結果を不用意に再生成する必要はありません。
