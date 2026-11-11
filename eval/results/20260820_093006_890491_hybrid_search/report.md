# Hybrid Search 消融实验报告

## 结论

**有效，建议启用。**

在 golden_basic_20_v2 的 20 道题上，Context Recall 从 0.908 变为 0.908，变化 +0.0 个百分点；Context Precision 变化 +0.7 个百分点；P95 延迟变化 -846.8 ms（-37.2%）；平均应用模型调用数变化 +0.00 次。核心质量指标有提升，质量护栏和成本限制均通过，建议启用 candidate 配置。

## 核心结果

| 指标 | baseline | candidate | 变化 | 门槛 | 是否通过 |
|------|----------|-----------|------|------|----------|
| Context Recall | 0.908 | 0.908 | +0.0 个百分点 | 至少提升 +0.0 个百分点 | 通过 |
| Context Precision | 0.870 | 0.876 | +0.7 个百分点 | 最低变化 -2.0 个百分点 | 通过 |
| P95 延迟 | 2278.3 ms | 1431.5 ms | -846.8 ms（-37.2%） | 增幅不超过 +30.0% | 通过 |
| 平均应用模型调用数 | 0.00 次 | 0.00 次 | +0.00 次 | 最多增加 0.00 次 | 通过 |

## 实验目标

- 验证能力：hybrid_search
- baseline：仅使用向量检索
- candidate：使用向量、BM25和RRF
- 唯一允许变化的字段：`retrieval_options.mode`

## 数据与运行条件

- 数据集：`golden_basic_20_v2`
- suite：`basic`
- 题目数量：20 道
- 每道题每组正式执行 1 次
- 检索请求使用第一期已有索引
- RAGAS 遥测关闭，离线 Langfuse 关闭

## 失败与异常

- baseline 请求失败：0 道；RAGAS 失败：0 道
- candidate 请求失败：0 道；RAGAS 失败：0 道

## 改善最多的题目

- 无可比较结果

## 退化最多的题目

- 无可比较结果

## 启用建议

核心质量指标有提升，质量护栏和成本限制均通过，建议启用 candidate 配置。

## 原始结果

- `baseline.raw.jsonl`
- `candidate.raw.jsonl`
- `baseline.metrics.json`
- `candidate.metrics.json`
- `comparison.json`
