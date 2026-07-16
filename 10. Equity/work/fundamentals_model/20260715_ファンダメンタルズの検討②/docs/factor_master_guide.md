# Factor Master運用ガイド

## 1. 目的

`factor_master.xlsx`は、ファクターコード、名称、分類、方向、個別例外、グループ統合方法を、Pythonコードを変更せず管理するための入力ファイルです。

## 2. 通常の更新手順

1. Factor_MasterへFAコードと名称を追加
2. Factor_Groupをプルダウンから選択
3. Enabled、Direction、Min_Coverageを設定
4. 通常はTransform、Winsorize、Neutralize、Rank_Normalizeをdefaultのままにする
5. Group_Settingsで統合方法を確認
6. 分析実行後、Config_ValidationとResolved_Factor_Settingsを確認

## 3. Direction

- `1`：高いほど望ましい
- `-1`：低いほど望ましい

例：Earnings Yieldは1、PERは-1、ROEは1、Accrualsは-1です。

## 4. 個別例外

大半のファクターは共通デフォルトを使用します。以下のような場合だけ個別設定を変更します。

- 分布が非常に裾の厚い指標：Winsorizeをmad_3
- 金額規模の指標：Transformをlogまたはlog1p
- すでに外部で中立化済み：Neutralizeを0
- 絶対値の差を残したい：Rank_Normalizeを0

## 5. グループ統合方法

### equal_weight

最も説明しやすく頑健な基準モデルです。

### manual

Base_Weightを使用します。投資仮説に基づく固定配分を使う場合に選びます。

### ic_adjusted

過去のOOS RankICとファクター間相関を使います。予測力は高められる可能性がありますが、推定誤差とウェイト変動を確認してください。

### pca

グループ内の共通変動を抽出します。予測力を直接最大化しないため、比較・診断モデルとしての利用を推奨します。

## 6. 入力検証

以下はエラーまたは警告になります。

- Factor_Codeの重複
- Directionが1または-1以外
- Group_Settingsに存在しないFactor_Group
- 許可されていないAggregation_Method
- Factor_Masterにあるが入力データにないFA列
- 入力データにあるがFactor_MasterにないFA列

## 7. 変更履歴

Factor_MasterはGit、SharePoint、文書管理システムなどで版管理してください。出力Excelには使用したマスタと解決後設定がコピーされるため、分析結果と設定の対応を確認できます。

## 差分・移動平均乖離の管理

`Feature_Engineering_Control` と `Derived_Feature_Rules` を使用します。

- `Scope_Type=group`: Value、Momentum等へ一括適用
- `Scope_Type=factor`: FA0101等へ個別適用。グループ設定より優先
- `Generation_Mode=all`: Enabledなルールをすべて使用
- `Generation_Mode=selected`: EnabledかつSelected=1のルールだけ使用
- `Include_Raw=1`: 原系列も説明変数として残す

標準の `Source_Lag_Periods=1` では、スコア時点tの派生値にt-1以前の情報だけを使用します。翌期リターンt+1と評価するため、最新使用情報からターゲットまで2時点の間隔を確保します。

詳細は `docs/derived_factor_features.md` を参照してください。
