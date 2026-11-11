# 优化二十 v2 Speed 与 Balanced 三档回归报告

## 结论

**主质量指标有收益，但另一项质量指标下降，暂不启用。**

在 golden_basic_20_v2 的 20 道题上，Context Precision 从 0.952 变为 0.877，变化 -7.4 个百分点；Context Recall 从 0.933 变为 1.000，变化 +6.7 个百分点；P95 延迟变化 -777.6 ms（-40.3%）；平均应用模型调用数变化 +0.00 次。主指标提升达到门槛，但质量护栏未通过。这说明当前实现有局部收益，但还不适合默认启用；建议保持「v2 Speed」，先排查退化题目再复测。

## 核心结果

| 指标 | baseline | candidate | 变化 | 门槛 | 是否通过 |
|------|----------|-----------|------|------|----------|
| Context Recall | 0.933 | 1.000 | +6.7 个百分点 | 至少提升 +0.0 个百分点 | 通过 |
| Context Precision | 0.952 | 0.877 | -7.4 个百分点 | 最低变化 -5.0 个百分点 | 未通过 |
| P95 延迟 | 1928.3 ms | 1150.7 ms | -777.6 ms（-40.3%） | 增幅不超过 +200.0% | 通过 |
| 平均应用模型调用数 | 0.00 次 | 0.00 次 | +0.00 次 | 最多增加 0.00 次 | 通过 |

## 实验目标

- 验证能力：retrieve_profile
- baseline：v2 Speed
- candidate：v2 Balanced
- 唯一允许变化的字段：`profile`、`top_k`、`retrieval_options.mode`、`retrieval_options.bm25_top_k`

## 数据与运行条件

- 数据集：`golden_basic_20_v2`
- suite：`basic`
- 题目数量：20 道
- 每道题每组正式执行 1 次
- 索引版本：baseline=v2，candidate=v2
- RAGAS 遥测关闭，离线 Langfuse 关闭

## 失败与异常

- baseline 请求失败：0 道；RAGAS 失败：0 道
- candidate 请求失败：0 道；RAGAS 失败：0 道

## 改善最多的题目

- `coupon_expire`：优惠券领取后通常有效多久，过期后会补发吗？，变化 +100.0 个百分点
- `warehouse_mode_codes`：FBP、SOP和WMS在仓储履约中分别表示什么？，变化 +33.3 个百分点

## 退化最多的题目

- 无可比较结果

## 启用建议

主指标提升达到门槛，但质量护栏未通过。这说明当前实现有局部收益，但还不适合默认启用；建议保持「v2 Speed」，先排查退化题目再复测。

## 原始结果

- `baseline.raw.jsonl`
- `candidate.raw.jsonl`
- `baseline.metrics.json`
- `candidate.metrics.json`
- `comparison.json`
