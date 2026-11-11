# 第一期/第二期三档统一 Top5 诊断报告

评测日期：2026-08-29

## 一、测试口径

- v1、v2 的 Speed 均使用正式 Top5 结果。
- v1、v2 的 Balanced 均复用各自正式 Top10 原始结果，只截取已经排好序的前 5 条重新执行 RAGAS。
- v1 Quality 同样截取旧 LLM 重排结果前 5 条；v2 Quality 使用严格中文 instruct 与 Rewrite-only rerank query 的 C 方案，从冻结 Quality Top10 候选中重新选择 Top5。
- 没有重新检索或重新调用 Query Rewrite；仅 Quality C 重新调用 Qwen3。
- 因此两期 Balanced、Quality 的 Precision/Recall 都是有效的 Top5 离线诊断值；P50/P95 仍是原 Top10 请求参考值，不是改成真实 Top5 后的正式延迟。
- RAGAS 并发 4、超时 120 秒、seed 42、telemetry 关闭；20/20 题均评分成功。

## 二、第一期统一 Top5 诊断版

第一期生产档位原本是 5/10/10。本表将冻结的 Balanced、Quality 结果截断为 Top5 后重新评分，与第二期形成相同口径。

| 档位 | 最终评分 TopK | Precision | Recall | P50/P95 参考值 | 应用模型调用/题 | 评分失败 |
|---|---:|---:|---:|---:|---:|---:|
| Speed | 5 | 0.8828 | 0.8042 | 1014.3/1059.6ms | 0 | 0/20 |
| Balanced | 5 | 0.8717 | 0.8042 | 1602.9/2453.8ms* | 0 | 0/20 |
| Quality | 5 | 0.9542 | 0.9333 | 32152.6/62549.9ms* | 2 | 0/20 |

`*` 第一阶段 Balanced、Quality 的延迟来自原 Top10 正式请求。

## 三、第二期统一 Top5 诊断版

| 档位 | 内部链路 | 最终评分 TopK | Precision | Recall | P50/P95 参考值 | 应用模型调用/题 | 评分失败 |
|---|---|---:|---:|---:|---:|---:|---:|
| Speed | Vector Top10 | 5 | 0.9517 | 0.9333 | 990.8/1928.3ms | 0 | 0/20 |
| Balanced | Hybrid 10+10 + RRF | 5 | 0.9500 | 1.0000 | 783.4/1150.7ms* | 0 | 0/20 |
| Quality | Rewrite + Hybrid 20+20 + Qwen3（严格 instruct + Rewrite Query） | 5 | **0.9658** | **0.9833** | 2860.5/3213.2ms* | 2 | 0/20 |

`*` Balanced、Quality 延迟来自原正式 Top10 请求，只用于量级参考。真实改为 Top5 后必须重新请求，才能形成正式 P50/P95。

## 四、第一期与第二期变化

这里按照同名档位、统一 Top5 质量口径比较。该对比仍同时包含索引、Chunk、上下文扩展和精排 Provider 等差异，不能把变化只归因于优化二十。

| 档位 | Precision 变化 | Recall 变化 | 观察 |
|---|---:|---:|---|
| Speed | +0.0689 | +0.1291 | Precision、Recall 均明显提升 |
| Balanced | +0.0783 | +0.1958 | Precision、Recall 均明显提升 |
| Quality | +0.0117 | +0.0500 | Precision、Recall 均提升，且 Precision 为三档最高 |

## 五、结论

第一期统一 Top5 后并不是逐档增长：Balanced 相对 Speed 的 Precision 从 0.8828 降到 0.8717，Recall 均为 0.8042；只有使用旧 LLM 重排的 Quality 明显提高到 Precision 0.9542、Recall 0.9333。此前第一期 Balanced Top10 的小幅增长不能用于说明其 Top5 优于 Speed。

第二期统一 Top5 后，Speed 与 Balanced 的 Precision 基本持平，Balanced Recall 达到 1.0000，说明 Hybrid 确实把缺失证据带入了前五，第二期 Balanced 的设计成立，并且相对第一期 Balanced 的两项质量指标均明显提升。

第二期 Quality 采用严格中文 instruct 与 Rewrite-only rerank query 后，Precision 达到 0.9658，为第二期三档最高；Recall 为 0.9833，高于 Speed，仅比 Balanced 的 1.0000 低 1.67 个百分点。该结果符合“Balanced 召回最高、Quality 精度最高”的档位定位。

原始 Quality Top10 的 Precision 偏低主要由第 6～10 条噪声造成；直接截断旧 Qwen3 排序还会把 `coupon_expire` 的必要证据排在前五之外。严格 instruct 修复了必要证据排序，Rewrite Query 进一步提高了 Precision。

因此：

1. 三个命名档位的最终结果统一为 Top5。
2. Balanced 保留 Hybrid 10+10 候选，以最高 Recall 为定位。
3. Quality 保留 Hybrid 20+20 候选，使用严格中文 instruct 与 Rewrite Query 精排 Top5，以最高 Precision 为定位。
4. 本报告足以支持配置决策，但 Balanced/Quality 的 P50/P95 仍来自旧 Top10 请求；后续真实流量或专项性能复跑后再替换参考延迟。

## 六、相关产物

- Qwen3 instruct/query 消融：`eval/results/20260829_opt20_qwen_instruct_ablation/report.md`
- 第一期 Speed/Balanced 冻结原始结果：`eval/results/20260825_024218_614462_opt20_v1_speed_vs_balanced_baseline_pre_opt20/`
- 第一期 Quality 冻结原始结果：`eval/results/20260825_030750_814096_opt20_v1_quality_rerank_baseline_pre_opt20/`
- Speed/Balanced 正式原始结果：`eval/results/20260828_152657_160723_opt20_v2_speed_vs_balanced_regression_formal/`
- Balanced/Quality 正式原始结果：`eval/results/20260829_012141_117367_opt20_v2_balanced_vs_quality_regression_formal/`
