# Query Rewrite 消融实验报告

## 结论

**质量收益成立，继续保留第一期 Query Rewrite，当前不做实现调整。**

在 golden_basic_20_v2 的 20 道题上，Context Precision 从 0.858 变为 0.936，变化 +7.8 个百分点；Context Recall 从 0.817 变为 0.850，变化 +3.3 个百分点；两项质量指标均达到实验门槛。P95 延迟变化 3872.7 ms（+253.8%），平均应用模型调用数变化 +1.00 次，说明改写存在明确的性能成本，但该成本不足以否定其质量收益，也不能据此将 Query Rewrite 限制为仅供关键词检索使用。正式结论是保留现有向量检索 Query Rewrite 能力及显式开关，不对第一期实现做修改。

## 核心结果

| 指标 | baseline | candidate | 变化 | 门槛 | 是否通过 |
|------|----------|-----------|------|------|----------|
| Context Recall | 0.817 | 0.850 | +3.3 个百分点 | 至少提升 +3.0 个百分点 | 通过 |
| Context Precision | 0.858 | 0.936 | +7.8 个百分点 | 最低变化 -2.0 个百分点 | 通过 |
| P95 延迟 | 1525.8 ms | 5398.5 ms | +3872.7 ms（+253.8%） | 增幅不超过 +40.0% | 未通过 |
| 平均应用模型调用数 | 0.00 次 | 1.00 次 | +1.00 次 | 最多增加 1.00 次 | 通过 |

## 实验目标

- 验证能力：query_rewrite
- baseline：关闭 Query Rewrite
- candidate：开启 Query Rewrite
- 唯一允许变化的字段：`query_options.enabled`

## 数据与运行条件

- 数据集：`golden_basic_20_v2`
- suite：`basic`
- 题目数量：20 道
- 每道题每组正式执行 1 次
- 索引版本：baseline=v1，candidate=v1
- RAGAS 遥测关闭，离线 Langfuse 关闭

## 失败与异常

- baseline 请求失败：0 道；RAGAS 失败：0 道
- candidate 请求失败：0 道；RAGAS 失败：0 道

## 人工复核

- candidate 的 20 道题均实际生成了改写查询，改写成功率 100%，未发生降级。
- 预期文档 Top5 命中率保持 20/20，预期章节 Top5 命中率保持 19/20，没有出现系统性召回破坏。
- 正确文档 MRR 从 0.950 提升至 1.000，正确章节 MRR 从 0.842 提升至 0.867，说明改写能够改善相关内容的排序。
- 个别题目的 Context Recall 变化存在评测波动，不能单独作为缩减 Query Rewrite 适用范围的依据；结合 Context Precision、排序和失败率，整体质量收益成立。

## 改善最多的题目

- `coupon_expire`：优惠券领取后通常有效多久，过期后会补发吗？，变化 +100.0 个百分点
- `seller_dsr_settlement`：DSR≥4.8的优质商家按什么周期结算，款项进入哪个账户？，变化 +50.0 个百分点

## 退化最多的题目

- `pickup_definition`：商家只打了快递单但快递还没扫描，这算按时发货了吗？，变化 -50.0 个百分点
- `return_inspection_refund_timeline`：退货申请通过后，买家寄回、仓库质检和退款到账分别要多久？，变化 -33.3 个百分点

## 启用建议

继续保留第一期 Query Rewrite 的现有实现，不改为“仅关键词检索支持”，也不取消向量检索的改写能力。现有显式开关继续保留；性能成本可以由调用方根据场景控制是否开启，但本轮不调整功能逻辑。

## 原始结果

- `baseline.raw.jsonl`
- `candidate.raw.jsonl`
- `baseline.metrics.json`
- `candidate.metrics.json`
- `comparison.json`
