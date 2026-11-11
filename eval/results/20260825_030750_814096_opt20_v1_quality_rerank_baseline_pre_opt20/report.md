# 优化二十改造前：v1 Quality旧LLM重排基线报告

## 结论

**出现负向效果，建议回退。**

在 golden_basic_20_v2 的 20 道题上，Context Precision 从 0.976 变为 0.782，变化 -19.4 个百分点；Context Recall 从 0.933 变为 0.925，变化 -0.8 个百分点；P95 延迟变化 -58026.3 ms（-92.8%）；平均应用模型调用数变化 -1.00 次。主指标明显下降或质量护栏未通过，建议保持「v1 Quality目标口径：Rewrite + LLM重排Top10」并排查退化题目。

## 核心结果

| 指标 | baseline | candidate | 变化 | 门槛 | 是否通过 |
|------|----------|-----------|------|------|----------|
| Context Precision | 0.976 | 0.782 | -19.4 个百分点 | 至少提升 +0.0 个百分点 | 未通过 |
| Context Recall | 0.933 | 0.925 | -0.8 个百分点 | 最低变化 -5.0 个百分点 | 通过 |
| P95 延迟 | 62549.9 ms | 4523.6 ms | -58026.3 ms（-92.8%） | 增幅不超过 +100.0% | 通过 |
| 平均应用模型调用数 | 2.00 次 | 1.00 次 | -1.00 次 | 最多增加 0.00 次 | 通过 |

## 实验目标

- 验证能力：opt20_prechange_quality_baseline
- baseline：v1 Quality目标口径：Rewrite + LLM重排Top10
- candidate：同候选RRF顺序参考组（不作为v1旧链路基线）
- 唯一允许变化的字段：`rerank_provider`

## 数据与运行条件

- 数据集：`golden_basic_20_v2`
- suite：`basic`
- 题目数量：20 道
- 每道题每组正式执行 1 次
- 索引版本：baseline=v1，candidate=v1
- RAGAS 遥测关闭，离线 Langfuse 关闭
- 每道题只生成一份共同 RRF Top20 候选
- A、B 两组分别精排并最终返回 Top10
- Query Rewrite、同义词扩展和证据编排全部关闭

## 失败与异常

- baseline 请求失败：0 道；RAGAS 失败：0 道
- candidate 请求失败：0 道；RAGAS 失败：0 道

## 改善最多的题目

- 无可比较结果

## 退化最多的题目

- `pickup_definition`：商家只打了快递单但快递还没扫描，这算按时发货了吗？，变化 -75.0 个百分点
- `combined_payment_refund`：组合支付的订单发生部分退款时，现金、优惠券和积分分别怎么退？，变化 -53.1 个百分点
- `platform_dispute`：买卖双方协商多久后可以申请平台介入，平台多久完成裁决和执行？，变化 -50.0 个百分点

## 启用建议

主指标明显下降或质量护栏未通过，建议保持「v1 Quality目标口径：Rewrite + LLM重排Top10」并排查退化题目。

## 原始结果

- `baseline.raw.jsonl`
- `candidate.raw.jsonl`
- `shared_candidates.raw.jsonl`
- `baseline.metrics.json`
- `candidate.metrics.json`
- `comparison.json`
