# 07 - ReRank 精排与 Evidence 编排

> **ReRank & Evidence Orchestration**  
> 检索后的候选集还需要两轮精加工：ReRank 用专用模型重新打分排序；Evidence 用 LLM 从候选中筛选和组织支撑答案的证据。

---

## 1. ReRank 重排序

### 1.1 为什么需要 ReRank？

```
向量初检 → 可能把 "不太相关但用了相似词" 的块排到前面

例如:
  query: "Python 中如何实现单例模式"
  vector top-3:
    1. "Java 单例模式实现" (score 0.91) ← 误判!
    2. "Python 设计模式概述" (score 0.89)
    3. "Python 单例模式最佳实践" (score 0.87) ← 真正最相关

ReRank 后:
    1. "Python 单例模式最佳实践" ← 精准
    2. "Python 设计模式概述"
    3. "Java 单例模式实现" ← 降到最后
```

**原因**：Embedding 模型侧重语义相似，但不够精细。ReRank 模型是专门训练来做 "query-chunk 相关度判断" 的，更精准。

### 1.2 rag-center 的 ReRank 架构

```python
# app/providers/rerank/base.py
class RerankProvider(ABC):
    @abstractmethod
    async def rerank(self, *, query, chunks, top_n) -> list[dict]:
        """输入候选列表，输出重新排序后的 top_n"""

# 两种实现:
# - NoopRerankProvider: 不做重排，保持原序
# - Qwen3RerankProvider: 调用 Qwen3-Rerank 模型
```

### 1.3 ReRank 在流水线中的位置

```python
# app/services/rag_service.py
async def _maybe_rerank(self, *, request, rerank_query, chunks, top_k):
    if not enabled:
        # 使用 NoopRerankProvider, 保持原序
        return NoopRerankProvider().rerank(query, chunks, top_n)

    # Qwen3Rerank 精排
    rerank_provider = Qwen3RerankProvider()
    reranked = await rerank_provider.rerank(
        query=rerank_query,
        chunks=chunks[:rerank_provider.max_candidates],  # 限制候选数
        top_n=top_n,
    )

    # 容错: 如果ReRank失败, 取前top_n个原序结果
    if failed:
        return chunks[:top_n]  # graceful degradation
```

### 1.4 ReRank vs No ReRank

| 维度 | 无ReRank | 有ReRank |
|------|---------|---------|
| 精度 | 中等 | ✅ 高 |
| 延迟 | 快 | +200~500ms |
| 成本 | 无额外 | LLM调用费用 |
| 候选限制 | 无 | 通常 max_candidates=50 |
| 可用套餐 | 全部 | Pro (Standard 无 ReRank 但可关) |

---

## 2. Evidence 证据编排 (Pro 专属)

### 2.1 什么是 Evidence 编排？

不是简单地返回一堆 chunks，而是让 LLM 分析 "这个问题的答案需要哪些要点"，然后从候选中挑出支撑每个要点的证据。

```
输入:
  query: "员工年假的申请条件和天数如何计算？"
  candidates: 25个 chunk

LLM 分析输出:
  {
    "aspects": [
      {
        "aspect": "年假申请条件",
        "evidence": [
          {"index": 5, "role": "core"},       // 核心证据: "入职满1年可申请"
          {"index": 12, "role": "supporting"}  // 支撑证据: "试用期不计入"
        ]
      },
      {
        "aspect": "年假天数计算",
        "evidence": [
          {"index": 3, "role": "core"},       // "P1-P3 每年5天"
          {"index": 8, "role": "core"},       // "P4-P6 每年10天"
          {"index": 18, "role": "supporting"}  // "工龄额外增加"
        ]
      }
    ]
  }
```

### 2.2 Evidence 编排流程

```python
# app/services/evidence_orchestration_service.py
async def orchestrate(self, *, query, tenant_id, kb_ids, candidates, ...):
    # 1. 构建证据候选池 (最多20个)
    evidence_candidates = self.build_candidates(
        candidates=candidates,          # 图注入后的候选
        retrieved_chunks=retrieved_chunks,  # 检索+rerank结果(含context)
    )
    # 优先选 top-ranked + context扩展的块
    # 保留至少6个基础池位置

    # 2. LLM 分析并编排
    raw = await self.llm_provider.chat_json(
        system_prompt="你是RAG证据编排器...",
        user_payload={
            "question": query,
            "max_evidence_items": 8,
            "candidates": [...]
        },
        temperature=0.0,  # 确定性输出
    )

    # 3. 校验输出
    aspects, selected_indexes = self._validate_model_output(raw, ...)
    # 校验: aspects非空, index在范围内, role合法, 去重

    # 4. 构建 EvidencePack
    pack = EvidencePackBuilder.build(
        aspects=research_aspects,
        selections={aspect_id: [EvidenceReference(...)]},
        candidates=research_candidates,
    )
    return EvidenceOrchestrationResult(pack=pack, metadata=...)
```

### 2.3 Evidence 输出结构

```json
{
    "evidence_pack": {
        "status": "complete",           // complete | partial | missing
        "missing_aspects": [],
        "groups": [
            {
                "aspect": "年假申请条件",
                "evidence": [
                    {"evidence_id": "E1", "role": "core"},
                    {"evidence_id": "E2", "role": "supporting"}
                ],
                "covered": true
            },
            {
                "aspect": "年假天数计算",
                "evidence": [
                    {"evidence_id": "E3", "role": "core"},
                    {"evidence_id": "E4", "role": "core"}
                ],
                "covered": true
            }
        ],
        "items": [
            {
                "evidence_id": "E1",
                "chunk_id": "...",
                "title": "员工手册",
                "content": "入职满1年的员工享有年假...",
                "source": "retrieval",
                "retrieved_rank": 5
            }
        ]
    }
}
```

### 2.4 Evidence Status 含义

| Status | 含义 | 后续建议 |
|--------|------|---------|
| `complete` | 所有必要要点都有核心证据 | 可用于生成最终答案 |
| `partial` | 部分要点有证据支撑 | 触发 Supplement 补充检索 |
| `missing` | 候选池中无任何相关证据 | 检查知识库内容覆盖 |

---

## 3. Context Expansion 上下文扩展

### 3.1 扩展策略

```
Anchor (命中块)
    │
    ├── previous → 前一个块 (基于 order_index)
    ├── next     → 后一个块 (基于 order_index)
    ├── parent_section → 父级章节的第一个块
    └── reference → 引用目标块 (基于 chunk_relations)
```

### 3.2 扩展限制

```python
MAX_NEIGHBORS_PER_ANCHOR = 2        # 最多取前后各几个
MAX_REFERENCE_TARGETS_PER_ANCHOR = 3  # 最多取几个引用目标
MAX_SUPPLEMENTAL_CHUNKS_PER_ANCHOR = 3 # 总共最多扩展几块
MAX_CONTEXT_CHARS_PER_ANCHOR = 4096   # 扩展后总字符数上限
```

### 3.3 扩展后的数据结构

```json
{
    "content": "命中块原文...",
    "context": {
        "content": "前块\n\n命中块\n\n后块\n\n父章节...",
        "chunk_ids": ["anchor_id", "prev_id", "next_id"],
        "sources": [
            {"relation": "anchor", "content": "..."},
            {"relation": "previous", "content": "..."},
            {"relation": "next", "content": "..."},
            {"relation": "parent_section", "content": "..."},
            {"relation": "reference", "content": "..."}
        ]
    }
}
```

---

## 4. Graph Injection 图注入

### 4.1 工作原理

```
检索命中 chunk (seed)
    │
    ▼ 查询 chunk_relations
  source_chunk_id == seed.chunk_id
    │
    ▼ 获取 target_chunk_id 列表
    │
    ▼ 查询 chunks 表获取完整内容
    │
    ▼ 作为新候选注入
  retrieval_source = "graph_injection"
```

### 4.2 注入限制

```python
MAX_GRAPH_SEEDS = 10            # 最多用10个seed
MAX_INJECTED_CHUNKS_PER_SEED = 3  # 每个seed最多注入3个
MAX_TOTAL_CANDIDATES = 30       # 注入后总候选最多30
```

### 4.3 注入关系类型

```python
class GraphCandidateExpansionService:
    def _collect_related_chunks(self, seed, ...):
        # 1. 引用目标 (chunk_relations)
        for target_id in target_ids: related.append((target, "reference"))

        # 2. 父级章节 (parent_section_id)
        parent = parent_map.get((kb_id, doc_id, version, parent_section_id))
        if parent: related.append((parent, "parent_section"))

        # 3. 相邻块 (order_index ± 1)
        if prose类型 and section_id:
            previous_key = (..., order_index - 1)  → relation="previous"
            next_key = (..., order_index + 1)      → relation="next"
```

---

## 下一步阅读

- [08 - 混合检索与RRF融合](./08-混合检索与RRF融合.md)
- [09 - Research深度研究](./09-Research深度研究.md)
- [10 - 多租户权限与套餐体系](./10-多租户权限与套餐体系.md)