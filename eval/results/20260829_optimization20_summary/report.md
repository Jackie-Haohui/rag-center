# 优化二十：检索链路收口与性能优化总结报告

评测与调整日期：2026-08-28 至 2026-08-29

数据集：`golden_basic_20_v2`（20 题）

本报告范围：单知识库检索链路；多知识库结果保留归档，不作为本次配置决策阻塞项

## 一、最终结论

优化二十的单库改造和三档配置决策完成，生产命名档位统一为最终返回 **5/5/5**：

| 档位 | 查询处理 | 内部召回 | 精排 | 最终返回 |
|---|---|---|---|---:|
| Speed | 不改写、同义词关闭 | Vector Top10 | 关闭 | Top5 |
| Balanced | 不改写、同义词关闭 | Vector Top10 + BM25 Top10 + RRF 60 | 关闭 | Top5 |
| Quality | Query Rewrite、同义词关闭 | Vector Top20 + BM25 Top20 + RRF 60 | Qwen3 严格中文 instruct，使用 Rewrite Query | Top5 |

三档统一 Top5 后，第二期 Speed 与 Balanced 相对第一期同名档位的 Precision、Recall 均提高。Quality 调优诊断达到 Precision 0.9658、Recall 0.9833，符合“Balanced 召回最高、Quality 精度最高”的产品定位。

本次按约定不重新运行正式端到端评测。Balanced/Quality 的 Top5 Precision、Recall 来自冻结结果的离线统一评分；Quality 调优值来自冻结 Quality Top10 候选上的 Qwen3 消融。旧 P95 只作为延迟量级参考，不能表述为新 5/5/5 配置的精确 P95。

## 二、工程验收

| 项目 | 结果 | 说明 |
|---|---|---|
| 三档固定为 5/5/5 | PASS | 后端 preset、策略测试和前端说明一致 |
| 命名档位禁止覆盖参数 | PASS | 仅 Custom 可提交高级参数 |
| Quality 精排 Top5 | PASS | `top_k=5`、`rerank top_n=5` |
| Quality 严格中文 instruct | PASS | Precision 优先，抑制背景、推测和重复片段 |
| Quality 使用 Rewrite Query 精排 | PASS | 精排接收实际 `search_query`；改写失败自然回退原问题 |
| Balanced 候选宽度保持 10+10 | PASS | 只收窄最终结果，不牺牲混合召回范围 |
| Quality 候选宽度保持 20+20 | PASS | 只收窄精排输出，不收窄前置召回范围 |
| Research 隔离 | PASS | 优化十九仍使用自己的固定 Top10 配置，不读取三档 preset |
| 全局 RRF / 单次 Embedding | PASS | 沿用优化二十已完成实现 |
| Hybrid 并行 / HTTP 连接复用 | PASS | 沿用优化二十已完成实现 |

自动化验证结果：

- 直接相关测试：38 passed；
- 扩展回归：59 passed；
- 完整后端：225 passed、1 个改造前已知失败、5 warnings；
- 前端生产构建：PASS，1979 modules；
- 已知失败仍为 `tests/test_markdown_splitter.py::test_mixed_heading_paragraph_table`，本专题没有修改切块逻辑或索引数据。

## 三、第一期与第二期统一 Top5 质量对比

| 阶段 | 档位 | Precision | Recall | P95 | 应用模型调用/题 |
|---|---|---:|---:|---:|---:|
| 第一期 | Speed | 0.8828 | 0.8042 | 1059.6ms | 0 |
| 第一期 | Balanced | 0.8717 | 0.8042 | 2453.8ms* | 0 |
| 第一期 | Quality（旧 LLM） | 0.9542 | 0.9333 | 62549.9ms* | 2 |
| 第二期 | Speed | 0.9517 | 0.9333 | 1928.3ms | 0 |
| 第二期 | Balanced | 0.9500 | 1.0000 | 1150.7ms* | 0 |
| 第二期 | Quality（调优方案） | **0.9658** | **0.9833** | 3213.2ms* | 2 |

`*` 第一、二期 Balanced/Quality 的 P95 来自原 Top10 正式请求，只用于说明延迟量级。Speed 本来就是 Top5，因此其 P95 是正式请求值。Quality 调优方案的 Qwen3 单次重排平均约 250.5ms、最大 573.1ms，但这只是精排调用耗时，不是端到端 P95。

同名档位的质量变化：

| 档位 | Precision 变化 | Recall 变化 | 结论 |
|---|---:|---:|---|
| Speed | +0.0689 | +0.1291 | 两项均提升 |
| Balanced | +0.0783 | +0.1958 | 两项均明显提升 |
| Quality | +0.0117 | +0.0500 | 两项均提升，且第二期 Precision 最高 |

第二期档位之间：

- Speed 提供最低复杂度的向量检索；
- Balanced 通过 Hybrid 把 Recall 提升到 1.0000，同时 Precision 与 Speed 基本持平；
- Quality 的 Precision 为 0.9658，比 Balanced 高 1.58 个百分点；Recall 为 0.9833，比 Balanced 低 1.67 个百分点，但高于 Speed 5 个百分点。

## 四、Quality 调优依据

固定 Top10 候选上的消融结果：

| 版本 | Precision | Recall |
|---|---:|---:|
| A：原 Qwen3 排序截 Top5 | 0.9300 | 0.9333 |
| B：A + Precision 优先中文 instruct | 0.9342 | 0.9833 |
| C：B + Rewrite-only rerank query | **0.9658** | **0.9833** |

结论：问题主要不是 Qwen3 模型能力不足，而是任务指令和精排 query 与检索 query 没有对齐。严格 instruct 将必要证据移回前五；只使用 Rewrite Query 又带来 3.17 个百分点 Precision 提升，因此最终代码采用 C。

限制：C 是在原正式 Quality 已返回的 Top10 候选上重新精排，并非重新执行完整的 20+20 端到端链路。因此它可以作为配置决策证据，但不能冒充新生产链路的正式最终指标。

## 五、单库性能改造结果

固定条件：v2、Custom、Hybrid 20+20、RRF 60、Top10，关闭 Rewrite、同义词、精排和 Evidence。该实验用于隔离 Hybrid 并行和连接复用收益，不随三档最终 Top5 决策改变。

| 阶段 | Precision | Recall | P50 | P95 | Embedding/题 | 失败 |
|---|---:|---:|---:|---:|---:|---:|
| 改造前 | 0.9087 | 0.9500 | 1664.8ms | 2538.9ms | 1 | 0/20 |
| 改造后 | 0.9212 | 0.9500 | 903.7ms | 1638.3ms | 1 | 0/20 |
| 变化 | +0.0125 | 0.0000 | -45.7% | -35.5% | 0 | 0 |

单库质量护栏通过，P50 和 P95 均显著降低，证明 Hybrid 并行与外部连接复用有效。

## 六、多知识库结果处理

多知识库实验产物继续保留归档。工程行为已经验证为一次全局 Vector、一次全局 BM25、一次全局 RRF，并且双库只计算一次 Embedding；但双库子集 Precision 从 0.3431 降至 0.2820。

在不重新检索的前提下，将改造前、改造后的双库 Top10 原始结果都截取前 5 条重新执行 RAGAS，得到：

| 双库口径 | Precision | Recall |
|---|---:|---:|
| 改造前 Top10 正式结果 | 0.3431 | 1.0000 |
| 改造后 Top10 正式结果 | 0.2820 | 1.0000 |
| 改造前 Top5 离线评分 | 0.6000 | 0.5000 |
| 改造后 Top5 离线评分 | 0.1625 | 0.5000 |

改造后两道题的 Top5 明细：

| 测例 | Precision | Recall |
|---|---:|---:|
| Agent 能力核验 + Offer 邮件要点 | 0.3250 | 1.0000 |
| 淘汰通知时限 + 简历造假类型 | 0.0000 | 0.0000 |

这说明减少返回数量只能删除尾部噪声，不能修复跨库多子问题的全局排序；第二道题的必要证据位于第 6～10 条，直接截断后 Precision、Recall 同时归零。本次只重新执行 4 条 RAGAS 输入，没有调用检索、Embedding、数据库或 Qwen3，也没有产生新的端到端延迟数据。样本只有两题，适合作为问题诊断，不足以代表所有双库流量。

当前阶段按决策只验收单库，因此该结果不阻塞 5/5/5 配置和单库优化二十结论。后续重新进入多库优化时，应单独建立专题修复和复测，不能把现有结果描述为多库质量提升。

## 七、报告与数据保留

保留：

- 三份第一期冻结基线报告；
- `20260825_opt20_prechange_baseline_summary.md` 冻结基线汇总；
- 三份第二期正式实验报告；
- `20260829_opt20_all_profiles_top5_diagnostic/report.md`；
- `20260829_opt20_qwen_instruct_ablation/report.md`；
- 本总结报告；
- 四份多知识库 JSON 正式产物。

删除：

- `20260829_opt20_quality_top5_diagnostic/report.md`，其内容已被统一 Top5 报告和 Qwen3 消融报告完整覆盖。

旧 5/10/10 正式报告继续作为历史原始证据保留，不回写、不删除，也不再代表当前生产 preset。

## 八、最终判定

在单知识库范围内，优化二十可以判定为通过：

- 生产三档配置已经统一为 5/5/5；
- 第二期三档在统一 Top5 质量口径下定位清晰；
- Quality 的 instruct/query 调优已经接入代码；
- 单库固定链路质量无退化，P50/P95 明显改善；
- 相关测试、完整后端回归和前端构建通过，只有一项既有切块测试失败；
- 本次没有重新跑正式端到端评测，因此 P95 参考值和 Quality C 指标均已明确标注适用边界。
