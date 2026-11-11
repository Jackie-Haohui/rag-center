# v2证据编排统一对比实验报告

## 结论

**证据筛选有效，建议保留并按需启用，不作为默认检索链路。**

在 golden_basic_20_v2 的 20 道题上，Context Precision 从 0.865 变为 1.000，变化 +13.5 个百分点；Context Recall 从 0.983 变为 0.958，变化 -2.5 个百分点；P95 延迟变化 33162.8 ms（+1237.5%）；平均应用模型调用数变化 +1.00 次。证据编排已经证明能够显著减少无关材料：Context Precision提升13.5个百分点，平均上下文由10.00条压缩到1.75条，引用回溯率为100.0%。但Context Recall下降2.5个百分点，P95由2.7秒增加到35.8秒，且每题增加1次模型调用，因此不适合作为所有在线检索的默认步骤。建议保留该能力，在多要点问答、可信引用、异步研究或高价值查询中按需开启；普通检索继续直接使用Qwen3精排Top10。

## 核心结果

| 指标 | baseline | candidate | 变化 | 门槛 | 是否通过 |
|------|----------|-----------|------|------|----------|
| Context Precision | 0.865 | 1.000 | +13.5 个百分点 | 至少提升 +0.1 个百分点 | 通过 |
| Context Recall | 0.983 | 0.958 | -2.5 个百分点 | 最低变化 -2.0 个百分点 | 未通过 |
| P95 延迟 | 2679.7 ms | 35842.5 ms | +33162.8 ms（+1237.5%） | 增幅不超过 +500.0% | 未通过 |
| 平均应用模型调用数 | 1.00 次 | 2.00 次 | +1.00 次 | 最多增加 1.00 次 | 通过 |

## 实验目标

- 验证能力：evidence_orchestration
- baseline：v2 + Qwen3精排Top10
- candidate：v2 + Qwen3精排Top10 + 证据包
- 唯一允许变化的字段：`evidence_options.enabled`

## 数据与运行条件

- 数据集：`golden_basic_20_v2`
- suite：`basic`
- 题目数量：20 道
- 每道题每组正式执行 1 次
- 索引版本：baseline=v2，candidate=v2
- RAGAS 遥测关闭，离线 Langfuse 关闭
- 每道题只执行一次 v2 Hybrid + RRF Top20 候选链路
- 两组共用同一次 Qwen3 精排 Top10 结果
- A组评价原始Top10，B组只评价证据包原始正文
- Query Rewrite和同义词扩展全部关闭

## 失败与异常

- baseline 请求失败：0 道；RAGAS 失败：0 道
- candidate 请求失败：0 道；RAGAS 失败：0 道

## 证据编排诊断

- 平均证据数量：1.75
- 平均证据字符数：364.60
- 证据状态分布：{"partial": 2, "complete": 18}
- 引用回溯率：100.0%（硬门槛 100%）
- 平均材料压缩：10.00条 / 2136.00字符 → 1.75条 / 364.60字符

### Recall退化定位

- `enterprise_purchase_payment_invoice`：企业购客户可以怎么付款，申请月结要什么条件，发票怎么开？，变化 -50.0 个百分点
- `enterprise_purchase_payment_invoice`证据包状态为complete，选中2条；未覆盖期望文档：商家入驻结算与对账流程.md。
- 本轮证据包最多只选择3条，没有触发10条数量上限；Recall下降来自证据选择遗漏，不是max_items截断。

## 改善最多的题目

- `price_protection`：刚买完就降价了，平台能把多花的钱补给我吗？，变化 +43.1 个百分点
- `pickup_definition`：商家只打了快递单但快递还没扫描，这算按时发货了吗？，变化 +37.5 个百分点
- `store_coupon_stack`：店里发的券能和平台跨店优惠一起用吗？，变化 +33.3 个百分点

## 退化最多的题目

- 无可比较结果

## 启用建议

证据编排已经证明能够显著减少无关材料：Context Precision提升13.5个百分点，平均上下文由10.00条压缩到1.75条，引用回溯率为100.0%。但Context Recall下降2.5个百分点，P95由2.7秒增加到35.8秒，且每题增加1次模型调用，因此不适合作为所有在线检索的默认步骤。建议保留该能力，在多要点问答、可信引用、异步研究或高价值查询中按需开启；普通检索继续直接使用Qwen3精排Top10。

## 原始结果

- `baseline.raw.jsonl`
- `candidate.raw.jsonl`
- `shared_pipeline.raw.jsonl`
- `baseline.metrics.json`
- `candidate.metrics.json`
- `comparison.json`
