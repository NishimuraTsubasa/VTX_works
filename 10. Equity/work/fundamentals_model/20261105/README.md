# Value × Growth Momentum — 指示書＋単一Pythonサンプル

**最初に読むファイルは `SAMPLE_README.md`（実行手順）と `INSTRUCTIONS.md`（設計・引継ぎ）です。**

| ファイル | 用途 |
|---|---|
| value_gm_research.py | 単独実行できるサンプル。既存の自作モジュールに依存しない |
| INSTRUCTIONS.md | 数式・比較設計・実装指示 |
| Value_GM_implementation_instructions.docx | 同じ指示書のWord版・12ページ |
| SAMPLE_README.md | 入力例・CLI・モデル・実装範囲 |
| CHATGPT_PROMPT.txt | 次のChatGPTに貼り付ける依頼文 |
| ACCEPTANCE_TESTS.md | 実プロジェクト接続時の検収要件 |
| sample_runtime_templates/ | 実行用runtime_config.jsonと空CSVひな形 |
| templates/ | 全体設計用の辞書・設定ひな形。config_spec.jsonは実行用ではない |
| sample_validation/ | 39項目テスト・実行環境・納品チェック |
| sample_demo/ | 13因子・161か月・89か月OOSの合成データ結果 |

## 実行

```bash
python -m pip install -r requirements.txt
python value_gm_research.py self-test
python value_gm_research.py demo --out my_demo_results
```

デモのA1はサンプルのlocal Ridgeであり、現行の13本モデルの再現ではありません。実データのA1は既存の予測CSVを使ってください。

モデルの実装・動作確認と、実データでの投資効果は別です。P3のTVP・残差補正・推定校正・特徴量選択器は指示書に仕様を記載し、サンプルでは未実装です。既存コード・結果は上書きしません。
