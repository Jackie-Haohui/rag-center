# 优化二十改造前：v1 Speed 与 Balanced 基线报告

## 结论

**有效但成本较高，建议按场景启用。**

在 golden_basic_20_v2 的 20 道题上，Context Precision 从 0.883 变为 0.889，变化 +0.6 个百分点；Context Recall 从 0.804 变为 0.858，变化 +5.4 个百分点；P95 延迟变化 1394.1 ms（+131.6%）；平均应用模型调用数变化 +0.00 次。质量收益达到门槛，但延迟或模型调用成本超限，建议仅在质量优先场景启用「v1 Balanced目标口径：Hybrid 10+10 Top10」。

## 核心结果

| 指标 | baseline | candidate | 变化 | 门槛 | 是否通过 |
|------|----------|-----------|------|------|----------|
| Context Recall | 0.804 | 0.858 | +5.4 个百分点 | 至少提升 +0.0 个百分点 | 通过 |
| Context Precision | 0.883 | 0.889 | +0.6 个百分点 | 最低变化 -5.0 个百分点 | 通过 |
| P95 延迟 | 1059.6 ms | 2453.8 ms | +1394.1 ms（+131.6%） | 增幅不超过 +100.0% | 未通过 |
| 平均应用模型调用数 | 0.00 次 | 0.00 次 | +0.00 次 | 最多增加 0.00 次 | 通过 |

## 实验目标

- 验证能力：opt20_prechange_profile_baseline
- baseline：v1 Speed目标口径：Vector Top5
- candidate：v1 Balanced目标口径：Hybrid 10+10 Top10
- 唯一允许变化的字段：`top_k`、`retrieval_options.mode`

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

## 改善最多的题目

- `seller_dsr_settlement`：DSR≥4.8的优质商家按什么周期结算，款项进入哪个账户？，变化 +50.0 个百分点
- `combined_payment_refund`：组合支付的订单发生部分退款时，现金、优惠券和积分分别怎么退？，变化 +33.3 个百分点
- `enterprise_purchase_payment_invoice`：企业购客户可以怎么付款，申请月结要什么条件，发票怎么开？，变化 +25.0 个百分点

## 退化最多的题目

- 无可比较结果

## 启用建议

质量收益达到门槛，但延迟或模型调用成本超限，建议仅在质量优先场景启用「v1 Balanced目标口径：Hybrid 10+10 Top10」。

## 原始结果

- `baseline.raw.jsonl`
- `candidate.raw.jsonl`
- `baseline.metrics.json`
- `candidate.metrics.json`
- `comparison.json`
