# Bloomberg取得・事前処理モジュール

## 1. 目的

`BBG_Config.xlsx`でTicker、Field、必須判定、取得期間を管理し、Bloombergから取得したデータを既存モデルが要求するExcel形式へ変換します。

生成先は次の通りです。

- `data/input/Universe_Master.xlsx`
- `data/input/market/Market_{asset_id}.xlsx`
- `data/input/Cross_Asset_Data.xlsx`
- `bbg_preprocess/logs/BBG_Query_Log.xlsx`
- `bbg_preprocess/logs/BBG_Data_Quality.xlsx`

## 2. 大量再取得を避ける設計

Bloombergから毎回全期間を取得しません。デフォルトは **incremental update** です。

1. 取得済みデータを `data/bbg_cache/` に保存します。
2. 各Tickerのキャッシュ最終日を確認します。
3. 原則として、最終日以降の不足期間だけをBDHで取得します。
4. データ訂正、休日差、遅延更新を吸収するため、末尾10営業日は再取得して上書きします。
5. Ticker、Field、ロール仕様を変更した場合だけ `full_refresh` または `force_refresh_tickers` を使います。

### Raw cacheをParquetにする理由

モデル入力、中間結果、監査ログはExcelへ出力します。一方、BloombergのRaw cacheだけはParquetを推奨します。

- 日次系列をExcelで繰り返し読み書きするより高速
- ファイル容量が小さい
- 日付型と欠損値を保ちやすい
- 何度もBloombergへ同じ要求を送る必要がなくなる

Raw cacheはユーザーが編集する入力ではなく、機械用の再取得防止領域です。したがって「人が確認・編集するものはExcel」「機械が差分更新するRaw履歴はParquet」と分離しています。

## 3. BBG_Config.xlsx

### Settings

| Parameter | 推奨値 | 説明 |
|---|---|---|
| provider | `mock` / `xbbg` | 実データ時は`xbbg` |
| run_mode | `incremental` | 通常運用 |
| default_start_date | 2014-01-02 | 初回取得開始日 |
| overlap_business_days | 10 | キャッシュ末尾の再取得幅 |
| cache_backend | parquet | Raw cache形式 |
| fail_on_required_missing | TRUE | 必須系列不足時に停止 |

### Futures_Universe

12指数先物のTickerを管理します。サンプルTickerは代表例であり、実行前にBloomberg Terminal上で利用可能なTickerを確認してください。特にFTSE先物Tickerは環境に応じて入力が必要です。

### Futures_Fields

デフォルト取得Fieldは次の通りです。

- `PX_OPEN`
- `PX_HIGH`
- `PX_LOW`
- `PX_LAST`
- `PX_VOLUME`
- `OPEN_INT`

### Cross_Asset

VIX、MOVE、DXY、米国金利、商品、為替を管理します。Credit spreadは契約・データソースにより利用Tickerが異なるため、使用する場合はExcelでTickerを入力します。

## 4. 実行方法

### 4.1 Mock modeによる動作確認

配布済みダミーデータをBloomberg取得結果として使用します。

```bash
cd bbg_preprocess
python run_bbg_preprocess.py
```

### 4.2 Bloomberg実データ

Bloomberg接続可能なPCで、`BBG_Config.xlsx`の`provider`を`xbbg`へ変更します。

```bash
pip install -r requirements_bbg.txt
# Bloombergの案内に従ってblpapiを導入し、xbbgを導入
python run_bbg_preprocess.py --config config/BBG_Config.xlsx
```

`xbbg`はBDH等の要求ヘルパーを提供しますが、Bloomberg Terminalまたは適切なBloomberg API接続が必要です。

## 5. 通常運用

### 日次または週次

1. `run_mode=incremental`
2. 事前処理を実行
3. `BBG_Data_Quality.xlsx`を確認
4. 問題がなければモデルを実行

```bash
python bbg_preprocess/run_bbg_preprocess.py
python run_pipeline.py
python src/build_excel_summary.py
```

### 設定変更時

TickerやFieldを変更した系列だけ再取得します。

`Run_Control`の`force_refresh_tickers`へカンマ区切りでTickerを入力します。

```text
ES1 Index,VIX Index
```

全系列の再取得は例外対応です。通常は行いません。

## 6. 品質チェック

`BBG_Data_Quality.xlsx`に次を出力します。

- 取得開始日・終了日
- 行数
- 必須Fieldの欠損数
- 重複日付
- 最終データのStale営業日数
- `PX_LAST`の日次15%以上変動件数
- OK / WARNING / INVALID

先物Generic系列ではロール仕様により価格ジャンプが発生し得ます。`PX_LAST`をそのままモデルへ使う前に、各Generic tickerのロール方式と連続系列の特性をBloomberg上で確認してください。

## 7. 実装上の判断

### なぜ1Tickerずつキャッシュするか

Ticker単位に分けることで、一部Tickerの失敗や設定変更があっても全データを再取得せずに済みます。

### なぜ末尾を重複取得するか

- 前営業日の値が後から更新される場合
- 市場休日の違い
- 通信失敗で最終日が欠けた場合
- 遅延して更新されたVolume / Open Interest

を吸収するためです。重複期間は日付でDeduplicateし、新しい取得値を優先します。

### なぜ取得ログをExcelへ残すか

Bloomberg要求量とキャッシュ利用状況を監査できるためです。どのTickerを、いつ、どの期間取得したかを確認できます。

## 8. 本番化前に確認する事項

- 全TickerをTerminalで検証
- Generic seriesのロール方式
- Volume / Open Interestのロール日前後処理
- 各市場の終値確定時刻
- Cross Assetを何営業日Lagするか
- Bloomberg契約上利用できるCredit spread ticker
- Bloomberg APIの利用上限・社内運用ルール
