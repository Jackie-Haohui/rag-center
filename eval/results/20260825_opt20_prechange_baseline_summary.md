# 优化二十改造前三组冻结基线

冻结日期：2026-08-25  
统一原则：所有业务请求均显式锁定索引版本、TopK、召回参数和功能开关；同义词扩展与 Evidence 均关闭。RAGAS 遥测和离线 Langfuse 均关闭。

## 一、v1 三档整体基线

数据集：`golden_basic_20_v2`，20 道固定电商题。

| 档位 | 冻结配置 | Context Precision | Context Recall | P50 | P95 | 应用模型调用/题 |
|---|---|---:|---:|---:|---:|---:|
| Speed | v1 Vector Top5 | 0.8828 | 0.8042 | 1014.3ms | 1059.6ms | 0 |
| Balanced | v1 Hybrid 10+10 Top10 | 0.8892 | 0.8583 | 1602.9ms | 2453.8ms | 0 |
| Quality | v1 Rewrite + Hybrid 20+20 + 旧 LLM 重排 Top10 | 0.9758 | 0.9333 | 32152.6ms | 62549.9ms | 2 |

结论：

- Balanced 相对 Speed：Precision +0.0064，Recall +0.0542，P95 +1394.1ms。
- Quality 相对 Balanced：Precision +0.0867，Recall +0.0750，但 P95 增加约 60.1 秒。
- Quality 只使用 Query Rewrite 和旧 LLM 重排，没有调用 Qwen3 精排。
- Quality 首轮旧 LLM 重排成功 18/20，2 道读取超时；复用共同候选并发 1 补跑后达到 20/20。该现象应保留为第一期旧重排的稳定性基线。
- Quality 实验中的 candidate 是同一候选池的原始 RRF 顺序，仅用于让双组评测器生成参考结果；v1 三档汇总取 baseline 的旧 LLM 重排结果。报告中的“出现负向效果”表示关闭旧重排后的 RRF 参考组变差，不代表 Quality 本身应回退。

产物：

- Speed/Balanced：`eval/results/20260825_024218_614462_opt20_v1_speed_vs_balanced_baseline_pre_opt20/`
- Quality：`eval/results/20260825_030750_814096_opt20_v1_quality_rerank_baseline_pre_opt20/`

## 二、单知识库固定配置性能与质量基线

数据集：同一套 20 道固定电商题。  
固定配置：Custom、Hybrid 20+20、RRF k=60、Top10，关闭 Rewrite、同义词、重排与 Evidence；业务并发 4，预热 2 题。

| 索引 | Context Precision | Context Recall | P50 | P95 | 请求失败 |
|---|---:|---:|---:|---:|---:|
| v1 | 0.8249 | 0.9083 | 1482.0ms | 1973.9ms | 0/20 |
| v2 | 0.9087 | 0.9500 | 1664.8ms | 2538.9ms | 0/20 |

改造前同配置下，v2 相对 v1：Precision +0.0839，Recall +0.0417，P95 +565.0ms。优化二十完成后应复跑同一 v2 配置，以隔离混合检索并行和连接复用的性能收益。

产物：`eval/results/20260825_034128_199038_opt20_single_kb_performance_baseline_pre_opt20/`

## 三、1 库/2 库融合基线

场景：招聘知识库。  
固定配置：v1、Custom、Hybrid 20+20、RRF k=60、Top10，关闭 Rewrite、同义词、重排与 Evidence。

质量集包含 4 道题：2 道单库、2 道必须跨两个知识库回答的双库题。

| 范围 | 题数 | Context Precision | Context Recall |
|---|---:|---:|---:|
| 1 库 | 2 | 0.7054 | 1.0000 |
| 2 库 | 2 | 0.3431 | 1.0000 |

双库 Recall 保持 1.0，但 Precision 比单库低 0.3623。两道双库题均严格返回 5+5 条，说明当前旧融合会按知识库平均分配结果席位。

性能采用单请求串行口径：每道题预热 1 次、正式重复 10 次，因此 1 库和 2 库各有 20 个正式样本。

| 范围 | 正式样本 | 平均 | P50 | P95 | 最大值 | 失败 |
|---|---:|---:|---:|---:|---:|---:|
| 1 库 | 20 | 1100.8ms | 1005.5ms | 1579.6ms | 1932.1ms | 0 |
| 2 库 | 20 | 1221.6ms | 1162.4ms | 1512.1ms | 1801.2ms | 0 |

2 库平均与 P50 分别比 1 库高 120.9ms、156.9ms；本轮 2 库 P95 略低，属于小样本网络波动，不能解释为双库更快。20/20 个双库正式样本均为严格 5+5 来源分配，这是优化二十多库全局融合改造最直接的前置基线。

产物：

- 质量与单次原始结果：`eval/results/20260825_opt20_multi_kb_prechange_baseline.json`
- 20+20 性能样本：`eval/results/20260825_opt20_multi_kb_prechange_performance.json`

## 四、改造后复跑要求

1. 不修改上述数据集、请求配置、索引版本和题目。
2. 单库性能复跑 v2 固定 Hybrid 20+20 Top10，只比较改造前后 P50/P95 与失败率。
3. 多库复跑相同 1 库/2 库题和 20+20 性能样本，重点检查双库 Precision、结果来源分布、重复 Embedding 次数和端到端延迟。
4. 最终整体回归再运行 v2 Speed/Balanced/Quality 5/10/10，与本报告的 v1 三档做阶段性能力对比；该对比用于评价二期整体收益，不用于单独归因优化二十。
