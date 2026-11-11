# 优化二十：Qwen3 instruct 与 rerank query 消融

评测日期：2026-08-29

## 测试目标

验证 Qwen3 rerank 的 Precision 优先中文 instruct，以及 Rewrite-only rerank query，能否提高第二期 Quality Top5 的 Context Precision。

## 固定条件

- 数据集：`golden_basic_20_v2`，20 题。
- 候选：复用当前正式 Quality 已返回的同一批 Top10。
- A、B 使用用户原始问题；C 只使用正式 Quality 运行时保存的 Rewrite Query。
- Document：保持当前 `retrieval_text` 格式。
- 最终结果：Qwen3 从固定 Top10 中重新选择 Top5。
- 没有重新检索、重新 Query Rewrite 或修改生产代码。
- RAGAS 并发 4、超时 120 秒、seed 42、telemetry 关闭。

B 相对 A 的唯一变化是 instruct：

```text
根据用户问题对候选片段进行严格相关性排序。只提高能够直接回答问题、提供明确规则、条件、时间、金额或结论的片段。仅主题相似、只提供背景信息、需要推测才能关联的片段应降低排名。对于包含多个子问题的查询，优先覆盖所有子问题。重复或信息高度重叠的片段应降低排名。
```

## 结果

| 版本 | Precision | Recall | 评分失败 |
|---|---:|---:|---:|
| A：当前 Quality Top5 | 0.9300 | 0.9333 | 0/20 |
| B：A + Precision 优先中文 instruct | 0.9342 | 0.9833 | 0/20 |
| C：B + Rewrite-only rerank query | **0.9658** | **0.9833** | 0/20 |

变化：

- B 相对 A：Precision +0.0042，Recall +0.0500。
- C 相对 B：Precision +0.0317，Recall +0.0000。
- C 相对 A：Precision +0.0358，Recall +0.0500。

B 的 Qwen3 调用 20/20 成功，单次重排平均 234.3ms、最大 395.6ms；C 同样 20/20 成功，平均 250.5ms、最大 573.1ms。该时间只包含复用 Top10 后的 Qwen3 调用，不是端到端检索延迟。

## B 的关键逐题变化

- `coupon_expire`：Recall 从 0.0000 恢复到 1.0000，说明必要证据被移入前五。
- `pickup_definition`：Precision 从 0.7000 提高到 1.0000。
- `platinum_clearance_price_protection`：Precision 从 0.7500 提高到 0.9167。
- `enterprise_purchase_payment_invoice`：Precision 从 1.0000 降到 0.7000，仍存在排序取舍。
- `refund_unshipped`：Precision 从 0.8333 降到 0.7000。

## C 的结果判断

| 对照 | Precision | Recall |
|---|---:|---:|
| 第一期 Quality Top5 | 0.9542 | 0.9333 |
| 第二期 Speed Top5 | 0.9517 | 0.9333 |
| 第二期 Balanced Top5 | 0.9500 | 1.0000 |
| 第二期 Quality C Top5 | **0.9658** | **0.9833** |

C 的 Precision 高于第二期 Balanced 1.58 个百分点，也高于第一期 Quality 1.17 个百分点；Recall 高于第二期 Speed和第一期 Quality 5 个百分点，仅比 Recall 已达 1.0000 的 Balanced 低 1.67 个百分点。该结果符合“Balanced 召回最高、Quality 精度最高”的档位定位。

## 结论

严格中文 instruct 能修复必要证据未进入前五的问题；在此基础上，使用 Rewrite-only Query 带来决定性的 Precision 提升。当前证据说明问题主要是 Qwen3 的任务指令与 Query 对齐方式，而不是模型能力不足。

由于 C 已达到目标，不建议立即继续叠加结构化 Document 或 Qwen/RRF 融合，以免增加变量和过拟合当前 20 题。严格中文 instruct 和 Rewrite-only rerank query 已接入真实 Quality Top5 链路；本轮按决策不再正式复跑。只有后续真实流量或专项复测未稳定达到 Precision 0.9500 时，再继续后两项调优。

本消融复用的是原 Quality 已选出的 Top10 候选，不代表完整候选池下的最终生产结果，因此该指标用于配置决策，不能冒充新生产链路的正式端到端结果。
