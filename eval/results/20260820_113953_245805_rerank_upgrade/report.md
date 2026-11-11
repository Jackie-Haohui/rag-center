# LLM 重排与 Qwen3-Rerank 精排对比实验报告

## 结论

**Qwen3-Rerank 暂未取得质量优势，但速度优势显著，保留精排方案并在轻量图谱改造后复测。**

在 golden_basic_20_v2 的 20 道题上，Context Precision 从 0.930 变为 0.889，变化 -4.1 个百分点；Context Recall 从 0.950 变为 0.858，变化 -9.2 个百分点，说明 Qwen3-Rerank 在当前索引和候选内容下暂未超过原有 LLM 重排。与此同时，P95 延迟从 50097.4 ms 降至 4555.6 ms，下降 90.9%，平均应用模型调用数没有增加，专用精排的工程性能优势已经得到明确验证。

因此，本轮不将 Qwen3-Rerank 判定为“不适合”，也不回退已经完成的精排实现。当前结论是：**保留 Qwen3-Rerank，认可其显著的速度收益，但暂不宣称质量提升。**轻量图谱改造完成后，候选片段将获得更完整的章节和关系上下文，可能改善精排模型对相似片段、跨章节证据和关联证据的判断；届时继续使用统一 Golden Set、共同候选 Top20 和最终 Top10 重新评测，再确定是否完成对 LLM 重排的质量替代。

## 核心结果

| 指标 | baseline | candidate | 变化 | 门槛 | 是否通过 |
|------|----------|-----------|------|------|----------|
| Context Precision | 0.930 | 0.889 | -4.1 个百分点 | 至少提升 +0.0 个百分点 | 未通过 |
| Context Recall | 0.950 | 0.858 | -9.2 个百分点 | 最低变化 +0.0 个百分点 | 未通过 |
| P95 延迟 | 50097.4 ms | 4555.6 ms | -45541.7 ms（-90.9%） | 增幅不超过 +50.0% | 通过 |
| 平均应用模型调用数 | 1.00 次 | 1.00 次 | +0.00 次 | 最多增加 0.00 次 | 通过 |

## 实验目标

- 验证能力：rerank_upgrade
- baseline：原有 LLM 重排
- candidate：Qwen3-Rerank 精排
- 唯一允许变化的字段：`rerank_provider`

## 数据与运行条件

- 数据集：`golden_basic_20_v2`
- suite：`basic`
- 题目数量：20 道
- 每道题每组正式执行 1 次
- 检索请求使用第一期已有索引
- RAGAS 遥测关闭，离线 Langfuse 关闭
- 每道题只生成一份共同 RRF Top20 候选
- A、B 两组分别精排并最终返回 Top10
- Query Rewrite、同义词扩展和证据编排全部关闭

## 失败与异常

- baseline 请求失败：0 道；RAGAS 失败：0 道
- candidate 请求失败：0 道；RAGAS 失败：0 道

## 改善最多的题目

- `warehouse_mode_codes`：FBP、SOP和WMS在仓储履约中分别表示什么？，变化 +25.0 个百分点
- `pickup_definition`：商家只打了快递单但快递还没扫描，这算按时发货了吗？，变化 +19.4 个百分点
- `platinum_clearance_price_protection`：白金会员购买清仓商品后很快降价，能按30天价保申请退差价吗？，变化 +10.6 个百分点

## 退化最多的题目

- `price_protection`：刚买完就降价了，平台能把多花的钱补给我吗？，变化 -32.5 个百分点
- `refund_unshipped`：东西还没寄出来，我不要了，钱多久能退回来？，变化 -29.3 个百分点
- `livestream_campaign_shipping`：直播大促订单最晚多久揽收，超时后怎么赔付？，变化 -16.7 个百分点

## 启用建议

保留 Qwen3-Rerank 精排实现及其生产接入，不因本轮质量回落删除或回退。当前报告将其认定为“工程性能收益明确、质量收益待验证”：速度要求较高的链路可以继续使用；对检索质量要求严格的场景，应保留本轮结果作为风险基线。

轻量图谱改造完成后需要再次执行同口径实验，重点观察当前退化题目的 Context Precision 和 Context Recall 是否恢复。只有复测通过质量门槛后，才能把最终结论升级为“精排升级整体有效”。

## 原始结果

- `baseline.raw.jsonl`
- `candidate.raw.jsonl`
- `shared_candidates.raw.jsonl`
- `baseline.metrics.json`
- `candidate.metrics.json`
- `comparison.json`
