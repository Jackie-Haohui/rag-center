# 优化二十改造前：单库固定Hybrid配置v1/v2基线报告

## 结论

**有效，建议启用。**

在 golden_basic_20_v2 的 20 道题上，Context Precision 从 0.825 变为 0.909，变化 +8.4 个百分点；Context Recall 从 0.908 变为 0.950，变化 +4.2 个百分点；P95 延迟变化 565.0 ms（+28.6%）；平均应用模型调用数变化 +0.00 次。核心质量指标有提升，质量护栏和成本限制均通过，建议启用「改造前单库性能冻结组：v2 Hybrid 20+20 Top10」。

## 核心结果

| 指标 | baseline | candidate | 变化 | 门槛 | 是否通过 |
|------|----------|-----------|------|------|----------|
| Context Recall | 0.908 | 0.950 | +4.2 个百分点 | 至少提升 +0.0 个百分点 | 通过 |
| Context Precision | 0.825 | 0.909 | +8.4 个百分点 | 最低变化 -5.0 个百分点 | 通过 |
| P95 延迟 | 1973.9 ms | 2538.9 ms | +565.0 ms（+28.6%） | 增幅不超过 +200.0% | 通过 |
| 平均应用模型调用数 | 0.00 次 | 0.00 次 | +0.00 次 | 最多增加 0.00 次 | 通过 |

## 实验目标

- 验证能力：opt20_prechange_single_kb_performance
- baseline：改造前单库固定配置：v1 Hybrid 20+20 Top10
- candidate：改造前单库性能冻结组：v2 Hybrid 20+20 Top10
- 唯一允许变化的字段：`index_version`

## 数据与运行条件

- 数据集：`golden_basic_20_v2`
- suite：`basic`
- 题目数量：20 道
- 每道题每组正式执行 1 次
- 索引版本：baseline=v1，candidate=v2
- RAGAS 遥测关闭，离线 Langfuse 关闭

## 失败与异常

- baseline 请求失败：0 道；RAGAS 失败：0 道
- candidate 请求失败：0 道；RAGAS 失败：0 道

## 改善最多的题目

- `pickup_definition`：商家只打了快递单但快递还没扫描，这算按时发货了吗？，变化 +100.0 个百分点
- `shipping_sla`：现货普通订单付款后多久必须完成揽收？，变化 +50.0 个百分点
- `warehouse_mode_codes`：FBP、SOP和WMS在仓储履约中分别表示什么？，变化 +33.3 个百分点

## 退化最多的题目

- `coupon_expire`：优惠券领取后通常有效多久，过期后会补发吗？，变化 -100.0 个百分点

## 启用建议

核心质量指标有提升，质量护栏和成本限制均通过，建议启用「改造前单库性能冻结组：v2 Hybrid 20+20 Top10」。

## 原始结果

- `baseline.raw.jsonl`
- `candidate.raw.jsonl`
- `baseline.metrics.json`
- `candidate.metrics.json`
- `comparison.json`
