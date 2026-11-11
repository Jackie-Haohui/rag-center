# v2索引下不开启精排与Qwen3精排对比实验报告

## 结论

**精排改善召回与证据排序，建议保留并继续优化退化题。**

在 golden_basic_20_v2 的 20 道题上，Context Precision 从 0.891 变为 0.880，变化 -1.1 个百分点；Context Recall 从 0.933 变为 0.983，变化 +5.0 个百分点；P95 延迟变化 861.7 ms（+46.2%）；平均应用模型调用数变化 +1.00 次。Qwen3精排将Context Recall提升5.0个百分点、nDCG@10提升2.9个百分点，说明关键证据更容易进入Top10并排到前面；Context Precision下降1.1个百分点，说明少数题仍有噪声片段前移。P95由1.9秒增加到2.7秒，每题增加1次专用精排调用，在线成本仍在本实验门槛内。综合本次20题的质量、排序和延迟，建议保留Qwen3作为默认精排能力，但不能宣称所有质量指标都已提升；应优先修复Precision退化题，并在扩充回归集后复测。

## 核心结果

| 指标 | baseline | candidate | 变化 | 门槛 | 是否通过 |
|------|----------|-----------|------|------|----------|
| Context Precision | 0.891 | 0.880 | -1.1 个百分点 | 至少提升 +0.1 个百分点 | 未通过 |
| Context Recall | 0.933 | 0.983 | +5.0 个百分点 | 最低变化 -2.0 个百分点 | 通过 |
| P95 延迟 | 1866.7 ms | 2728.4 ms | +861.7 ms（+46.2%） | 增幅不超过 +200.0% | 通过 |
| 平均应用模型调用数 | 0.00 次 | 1.00 次 | +1.00 次 | 最多增加 1.00 次 | 通过 |

## 实验目标

- 验证能力：rerank_effect_v2
- baseline：v2 + RRF原始Top10（不精排）
- candidate：v2 + Qwen3精排Top10
- 唯一允许变化的字段：`rerank_options.enabled`、`rerank_provider`

## 数据与运行条件

- 数据集：`golden_basic_20_v2`
- suite：`basic`
- 题目数量：20 道
- 每道题每组正式执行 1 次
- 索引版本：baseline=v2，candidate=v2
- RAGAS 遥测关闭，离线 Langfuse 关闭
- 每道题只生成一份共同 RRF Top20 候选
- A组按RRF原始顺序取Top10，B组使用Qwen3精排Top10
- Query Rewrite、同义词扩展和证据编排全部关闭

## 失败与异常

- baseline 请求失败：0 道；RAGAS 失败：0 道
- candidate 请求失败：0 道；RAGAS 失败：0 道

## 排序诊断

- nDCG@10（Golden Set期望文档与章节标注）：0.835 → 0.864，变化+2.9 个百分点。
- nDCG@10只用于解释排序效果，不替代四项正式指标。

## 改善最多的题目

- `refund_unshipped`：东西还没寄出来，我不要了，钱多久能退回来？，变化 +37.5 个百分点
- `payment_callback_delay`：银行卡已经扣钱，但订单还显示没付款，应该怎么办？，变化 +28.2 个百分点
- `platinum_clearance_price_protection`：白金会员购买清仓商品后很快降价，能按30天价保申请退差价吗？，变化 +16.7 个百分点

## 退化最多的题目

- `price_protection`：刚买完就降价了，平台能把多花的钱补给我吗？，变化 -45.6 个百分点
- `return_without_reason`：申请七天无理由退货需要满足什么期限和商品条件？，变化 -25.0 个百分点
- `pop_operation_difference`：自营和POP在定价权、发货方式、客服责任上分别有什么差异？，变化 -16.7 个百分点

## 启用建议

Qwen3精排将Context Recall提升5.0个百分点、nDCG@10提升2.9个百分点，说明关键证据更容易进入Top10并排到前面；Context Precision下降1.1个百分点，说明少数题仍有噪声片段前移。P95由1.9秒增加到2.7秒，每题增加1次专用精排调用，在线成本仍在本实验门槛内。综合本次20题的质量、排序和延迟，建议保留Qwen3作为默认精排能力，但不能宣称所有质量指标都已提升；应优先修复Precision退化题，并在扩充回归集后复测。

## 原始结果

- `baseline.raw.jsonl`
- `candidate.raw.jsonl`
- `shared_candidates.raw.jsonl`
- `baseline.metrics.json`
- `candidate.metrics.json`
- `comparison.json`
