# v2索引下单次检索与完整Research对比实验报告

## 结论

**无明显收益，暂不启用。**

在 golden_basic_20_v2 的 10 道题上，Context Precision 从 0.889 变为 0.923，变化 +3.4 个百分点；Context Recall 从 0.944 变为 0.944，变化 +0.0 个百分点；P95 延迟变化 -19107.9 ms（-59.8%）；平均应用模型调用数变化 +2.90 次。主指标未达到预设提升门槛，当前数据不足以证明该能力有明确收益，暂不默认启用「v2 + Controlled Agentic Research」。

## 核心结果

| 指标 | baseline | candidate | 变化 | 门槛 | 是否通过 |
|------|----------|-----------|------|------|----------|
| Context Recall | 0.944 | 0.944 | +0.0 个百分点 | 至少提升 +1.0 个百分点 | 未通过 |
| Context Precision | 0.889 | 0.923 | +3.4 个百分点 | 最低变化 -5.0 个百分点 | 通过 |
| P95 延迟 | 31976.4 ms | 12868.5 ms | -19107.9 ms（-59.8%） | 增幅不超过 +200.0% | 通过 |
| 平均应用模型调用数 | 2.00 次 | 4.90 次 | +2.90 次 | 最多增加 6.00 次 | 通过 |

## 实验目标

- 验证能力：research_effect_v2
- baseline：v2 + 单次检索 + Qwen3精排 + 普通证据包
- candidate：v2 + Controlled Agentic Research
- 唯一允许变化的字段：`execution_mode`

## 数据与运行条件

- 数据集：`golden_basic_20_v2`
- suite：`research`
- 题目数量：10 道
- 每道题每组正式执行 1 次
- 索引版本：baseline=v2，candidate=v2
- RAGAS 遥测关闭，离线 Langfuse 关闭
- 每个检索任务使用v2 Hybrid + RRF Top20，再由Qwen3精排至Top10
- 两组最终都只用evidence_pack.items原始正文进行RAGAS评分
- Query Rewrite和同义词扩展全部关闭
- Qwen3、Planner、Decider、Finalizer均计入模型调用数

## 失败与异常

- baseline 请求失败：0 道；RAGAS 失败：0 道
- candidate 请求失败：0 道；RAGAS 失败：0 道

## Research诊断

- Context指标共同可评价题目：9/10 道
- 质量汇总排除 `control_no_answer_offline_parking`：baseline=expected_missing_no_context，candidate=excluded_to_match_baseline
- 最终答案非空：10/10 道
- 答案引用映射有效：10/10 道
- 引用回溯率：100.0%（硬门槛100%）
- 发生第二轮补证：2 道
- 降级：0 道
- 平均Research LLM调用：3.00 次
- 平均检索任务：1.90 次
- 平均输入/输出Token：8057.00 / 489.40

## 改善最多的题目

- 无可比较结果

## 退化最多的题目

- 无可比较结果

## 启用建议

主指标未达到预设提升门槛，当前数据不足以证明该能力有明确收益，暂不默认启用「v2 + Controlled Agentic Research」。

## 原始结果

- `baseline.raw.jsonl`
- `candidate.raw.jsonl`
- `shared_pipeline.raw.jsonl`
- `baseline.metrics.json`
- `candidate.metrics.json`
- `comparison.json`
