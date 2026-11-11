# v2索引下禁止补证与允许一次补证对比实验报告

## 结论

**补证实验条件不成立，暂时不能判断第二轮是否有效。**

在 golden_basic_20_v2 的 4 道题上，Context Precision 从 0.896 变为 0.958，变化 +6.3 个百分点；Context Recall 从 1.000 变为 1.000，变化 +0.0 个百分点；P95 延迟变化 1988.3 ms（+14.3%）；平均应用模型调用数变化 +0.50 次。本轮4题没有形成至少2道合法修复分母，且两组只有2/4道首轮快照一致。Precision变化不能可靠归因于第二轮补证；应先固定复用Planner与首轮候选快照，并补充首轮确实缺证、知识库中又存在目标证据的多跳题后重测。

## 核心结果

| 指标 | baseline | candidate | 变化 | 门槛 | 是否通过 |
|------|----------|-----------|------|------|----------|
| Context Recall | 1.000 | 1.000 | +0.0 个百分点 | 至少提升 +1.0 个百分点 | 未通过 |
| Context Precision | 0.896 | 0.958 | +6.3 个百分点 | 最低变化 -5.0 个百分点 | 通过 |
| P95 延迟 | 13940.7 ms | 15929.1 ms | +1988.3 ms（+14.3%） | 增幅不超过 +100.0% | 通过 |
| 平均应用模型调用数 | 4.50 次 | 5.00 次 | +0.50 次 | 最多增加 2.00 次 | 通过 |

## 实验目标

- 验证能力：research_supplement_v2
- baseline：v2 + 查询分解 + 首轮并行检索（禁止补证）
- candidate：v2 + 查询分解 + 一次定向补证
- 唯一允许变化的字段：`execution_mode`

## 数据与运行条件

- 数据集：`golden_basic_20_v2`
- suite：`research_multihop`
- 题目数量：4 道
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

- Context指标共同可评价题目：4/4 道
- 最终答案非空：4/4 道
- 答案引用映射有效：4/4 道
- 引用回溯率：100.0%（硬门槛100%）
- 发生第二轮补证：1 道
- 降级：0 道
- 平均Research LLM调用：3.00 次
- 平均检索任务：2.00 次
- 平均输入/输出Token：8342.00 / 490.25

## 第二轮补证诊断

- Multi-hop Repair Rate：not_evaluable
- 可修复分母：0 道；实际修复：0 道
- 两组首轮快照完全一致：2/4 道
- Multi-hop Repair Rate为诊断指标，不替代四项正式指标。

## 改善最多的题目

- 无可比较结果

## 退化最多的题目

- 无可比较结果

## 启用建议

本轮4题没有形成至少2道合法修复分母，且两组只有2/4道首轮快照一致。Precision变化不能可靠归因于第二轮补证；应先固定复用Planner与首轮候选快照，并补充首轮确实缺证、知识库中又存在目标证据的多跳题后重测。

## 原始结果

- `baseline.raw.jsonl`
- `candidate.raw.jsonl`
- `shared_pipeline.raw.jsonl`
- `baseline.metrics.json`
- `candidate.metrics.json`
- `comparison.json`
