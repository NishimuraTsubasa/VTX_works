# Changelog

## v0.13.1

- 第1層の単一FA回帰SubScoreを本番スコア生成から削除
- Raw Factor中心のスコアリングへ変更
- Centered Percentile / Gaussian Rank / Z-scoreを選択可能化
- FA別Q5-Q1 Factor Return履歴を追加
- 同一Factor Group内のFactor Return相関によるウェイトを追加
- 相関行列縮小、最大ウェイト、時系列平滑化、Equal Weight縮小を追加
- N00～N07の新比較パターンへ刷新
- 第3層は主効果OLS、主効果Ridge、選択交差項Ridgeを比較
- セクター主効果を既定で削除し、階段状予測を抑制
- 出力をRaw Factor新仕様に必要なExcel・PDFだけへ整理
- 日本語グリフ検証付きフォント自動検出へ変更
- WindowsのYu Gothic / Meiryoを実ファイルパスから登録
- 日本語フォントが見つからない場合は文字化けPDFを作らず明示的に停止
- 日本語フォント確認用PDFスクリプトを追加
- Copilot向けfactor_master.xlsx詳細作成指示書を追加
