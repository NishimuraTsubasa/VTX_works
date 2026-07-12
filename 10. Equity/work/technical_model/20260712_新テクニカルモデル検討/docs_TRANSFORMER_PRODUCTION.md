# Transformer Production Upgrade

本デモの `TemporalPatchAttentionRegressor` は、CPU上で短時間に再現できるよう、5営業日Patchと正則化線形ヘッドから時点寄与を算出するTransformer-style surrogateです。

本番化では、同じインターフェースを維持したまま以下へ交換します。

- Temporal Fusion Transformer (TFT)
- PatchTST
- iTransformer（変数間Attentionを重視する場合）

必須インターフェース:

```python
model.fit(sequence_x, y)
pred = model.predict(sequence_x)
pred, temporal_attention = model.predict_with_attention(sequence_x)
```

説明はAttention単独で断定せず、期間Occlusion・Seed/Fold安定性・Evidence状態と合わせて実施します。
