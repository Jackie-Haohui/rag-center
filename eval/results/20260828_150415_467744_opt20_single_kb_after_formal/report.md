# 优化二十：单库固定 v2 Hybrid 配置改造前后报告

## 结论

**有效，建议启用。**

在 golden_basic_20_v2 的 20 道题上，Context Precision 从 0.909 变为 0.921，变化 +1.2 个百分点；Context Recall 从 0.950 变为 0.950，变化 +0.0 个百分点；P95 延迟变化 -900.6 ms（-35.5%）；平均应用模型调用数变化 +0.00 次。核心质量指标有提升，质量护栏和成本限制均通过，建议启用「改造后：v2 Hybrid 20+20 Top10」。

## 核心结果

| 指标 | baseline | candidate | 变化 | 门槛 | 是否通过 |
|------|----------|-----------|------|------|----------|
| Context Recall | 0.950 | 0.950 | +0.0 个百分点 | 至少提升 +0.0 个百分点 | 通过 |
| Context Precision | 0.909 | 0.921 | +1.2 个百分点 | 最低变化 -2.0 个百分点 | 通过 |
| P95 延迟 | 2538.9 ms | 1638.3 ms | -900.6 ms（-35.5%） | 增幅不超过 +5.0% | 通过 |
| 平均应用模型调用数 | 0.00 次 | 0.00 次 | +0.00 次 | 最多增加 0.00 次 | 通过 |

## 实验目标

- 验证能力：opt20_single_kb_after
- baseline：改造前冻结：v2 Hybrid 20+20 Top10
- candidate：改造后：v2 Hybrid 20+20 Top10
- 唯一允许变化的字段：

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

- 无可比较结果

## 退化最多的题目

- 无可比较结果

## 启用建议

核心质量指标有提升，质量护栏和成本限制均通过，建议启用「改造后：v2 Hybrid 20+20 Top10」。

## 原始结果

- `baseline.raw.jsonl`
- `candidate.raw.jsonl`
- `baseline.metrics.json`
- `candidate.metrics.json`
- `comparison.json`
