# 08 - 混合检索与 RRF 融合

> **Hybrid Search & RRF Fusion**  
> 混合检索是 rag-center 推荐的核心检索方式，通过 RRF（Reciprocal Rank Fusion）将语义检索和关键词检索的结果智能融合。

---

## 1. 为什么需要混合检索？

### 1.1 单路检索的盲区

```
查询: "P7 级别员工的股权激励方案"

纯向量检索 (vector):
  ✅ 理解 "股权激励" = "期权" = "stock options"
  ✅ 召回了 "股权激励管理办法"
  ❌ 漏掉了明确提到 "P7" 但用词不同的文档

纯关键词检索 (bm25):
  ✅ 精准命中 "P7" 编号
  ❌ "股权激励方案" 和 "期权管理办法" 视为无关
  ❌ 召回碎片化、覆盖不全
```

### 1.2 混合检索的效果

```
查询: "P7 级别员工的股权激励方案"

混合检索 (vector + bm25 + RRF):
  ✅ 语义: 召回 "股权激励管理办法" (含 P7 相关条款)
  ✅ 关键词: 召回 "P7 职级权益清单" (含期权说明)
  ✅ RRF融合: 双路都命中的块排名最高
  ✅ 最终结果: 既精准又全面
```

---

## 2. 混合检索执行流程

```python
# app/services/rag_service.py
async def _retrieve_multi_kb(self, *, tenant_id, request, knowledge_bases, 
                              search_query, mode, top_k, index_versions):
    
    if mode == "hybrid":
        # 两路并行检索
        vector_result, bm25_result = await asyncio.gather(
            # 路径1: 语义检索
            self._run_vector_scope_branch(
                tenant_id=tenant_id,
                index_versions=index_versions,
                search_query=search_query,
                top_k=vector_top_k,  # 默认 20
            ),
            # 路径2: 关键词检索
            self._run_bm25_scope_branch(
                tenant_id=tenant_id,
                index_versions=index_versions,
                search_query=search_query,
                top_k=bm25_top_k,    # 默认 20
            ),
        )
        
        # 容错: 如果只有一路成功, 降级为单路
        if vector_result.error and bm25_result.error:
            raise RETRIEVAL_FAILED
        
        if vector_result.error:
            return bm25_result.chunks[:top_k]
        if bm25_result.error:
            return vector_result.chunks[:top_k]
        
        # 正常情况: RRF 融合
        return self.hybrid_search_service.fuse_by_rrf(
            vector_chunks=vector_result.chunks,
            bm25_chunks=bm25_result.chunks,
            rrf_k=rrf_k,    # 默认 60
            top_n=top_k,     # 最终返回数
        )
```

### 2.1 容错设计

```
┌─────────────────────────────┐
│      两路同时执行            │
├──────────────┬──────────────┤
│  Vector ✅   │   BM25 ✅    │  → RRF融合, 最佳结果
│  Vector ✅   │   BM25 ❌    │  → 降级为纯vector
│  Vector ❌   │   BM25 ✅    │  → 降级为纯bm25
│  Vector ❌   │   BM25 ❌    │  → 返回错误 50002
└──────────────┴──────────────┘
```

---

## 3. RRF 融合详解

### 3.1 核心公式

```
RRF(chunk) = Σ 1 / (k + rank_i)

k = 60 (平滑常数, 可配置)

rank_i: chunk 在第 i 路检索中的排名 (1-based)
```

### 3.2 为什么用 k=60？

```
k 越小 → 排名靠前的块优势越大 → 可能忽略某一路的好结果
k 越大 → 各路权重越平均 → 失去双路命中的优势

60 是经过业界验证的经验值，兼顾了:
  - 双路命中奖励 (排名优先)
  - 单路质量不埋没 (不因另一路未命中而过度惩罚)
```

### 3.3 排序优先级

```python
# app/services/hybrid_search_service.py
fused.sort(key=lambda item: (
    -item["score"],                               # ① RRF分数降序
    -int(item["retrieval_source"] == "hybrid"),   # ② 双路命中优先
    min(rank for rank in (vector_rank, bm25_rank) # ③ 最小原始排名
        if rank is not None),
    str(item.get("kb_id") or ""),                # ④ 稳定排序
    str(item.get("chunk_id") or ""),             # ⑤ 稳定排序
))
```

### 3.4 融合过程示意图

```
向量路 top-3:             BM25路 top-3:
┌────┬──────────┬──────┐  ┌────┬──────────┬──────┐
│Rank│ chunk_id │ score│  │Rank│ chunk_id │ score│
├────┼──────────┼──────┤  ├────┼──────────┼──────┤
│ 1  │   A      │ 0.95 │  │ 1  │   C      │ 18.2 │
│ 2  │   B      │ 0.91 │  │ 2  │   A      │ 15.5 │
│ 3  │   D      │ 0.87 │  │ 3  │   E      │ 14.1 │
└────┴──────────┴──────┘  └────┴──────────┴──────┘

RRF 融合 (k=60):

┌──────┬────────┬────────┬────────┬──────────┬──────────┐
│Chunk │ v_rank │ b_rank │ RRF分  │ 来源     │ 最终排名 │
├──────┼────────┼────────┼────────┼──────────┼──────────┤
│  A   │   1    │   2    │ 0.0326 │ hybrid   │    1     │
│  C   │   -    │   1    │ 0.0164 │ bm25     │    2     │
│  B   │   2    │   -    │ 0.0161 │ vector   │    3     │
│  D   │   3    │   -    │ 0.0159 │ vector   │    4     │
│  E   │   -    │   3    │ 0.0159 │ bm25     │    5     │
└──────┴────────┴────────┴────────┴──────────┴──────────┘

A 双路命中 → 排名最高!
```

---

## 4. 配置参数详解

### 4.1 请求级别参数

```json
{
    "retrieval_options": {
        "mode": "hybrid",
        "vector_top_k": 20,  // 语义路候选数 (1-50)
        "bm25_top_k": 20,    // 关键词路候选数 (1-50)
        "rrf_k": 60          // RRF平滑常数 (≥1)
    },
    "top_k": 10              // 最终返回条数 (1-50)
}
```

### 4.2 全局默认配置

```python
# 环境变量/配置文件
RETRIEVAL_MODE = "hybrid"         # 默认检索模式
HYBRID_VECTOR_TOP_K = 20         # 语义路默认候选数
HYBRID_BM25_TOP_K = 20           # 关键词路默认候选数
TOP_K = 10                        # 默认返回条数
```

### 4.3 参数调优建议

| 场景 | vector_top_k | bm25_top_k | rrf_k | top_k |
|------|:-----------:|:----------:|:-----:|:-----:|
| 快速响应 | 10 | 10 | 60 | 5 |
| 平衡 (默认) | 20 | 20 | 60 | 10 |
| 高召回 | 40 | 40 | 30 | 20 |
| 精确匹配为主 | 10 | 30 | 60 | 10 |
| 语义理解为主 | 30 | 10 | 60 | 10 |

---

## 5. 跨知识库联合检索

### 5.1 请求多知识库

```json
{
    "kb_ids": ["kb-hr", "kb-legal", "kb-finance"],
    "query": "产假期间的工资如何发放",
    ...
}
```

### 5.2 多KB检索策略

```python
# 每个KB有独立的 index_version
index_versions = {
    "kb-hr":      "v2",  # HR知识库用v2
    "kb-legal":   "v1",  # 法律知识库暂用v1
    "kb-finance": "v2",  # 财务知识库用v2
}

# 检索时对每个KB独立执行, 然后合并
for kb_id, version in index_versions.items():
    chunks += pgvector.search(tenant_id, kb_id, version, query_vector, top_k)

# RRF融合时, (kb_id, chunk_id)作为唯一键
```

### 5.3 多KB限制

```python
# 套餐限制
free:      max_kb_per_retrieve = 1   # 只能查1个KB
standard:  max_kb_per_retrieve = 3   # 最多3个KB
pro:       max_kb_per_retrieve = 5   # 最多5个KB

# 系统硬限制
settings.multi_kb_max  # 全局上限
```

---

## 6. 关键词检索配置

### 6.1 Elasticsearch 集成

```python
# 配置
keyword_search_provider = "elasticsearch"

# 实现
class ElasticsearchKeywordSearchProvider(KeywordSearchProvider):
    async def keyword_search_scope(self, *, query, tenant_id, 
                                    index_versions, top_k):
        # Elasticsearch DSL查询
        # 按 tenant_id + kb_id + index_version 过滤
        # BM25 打分
        # 返回 top_k 结果
```

### 6.2 如果不配置 ES

```python
# 禁用关键词检索
keyword_search_provider = None

# 此时:
# - vector 模式正常工作
# - hybrid 模式退化为 vector (bm25路返回空)
# - bm25 模式返回错误
```

---

## 7. 性能分析

```
一次 hybrid 检索的典型耗时:

向量路径:
  embedding_query()  →  100-200ms  (API调用)
  pgvector_search()  →   10-30ms   (数据库查询)
  合计:                   110-230ms

BM25路径:
  es_search()        →   20-50ms   (ES查询)
  合计:                    20-50ms

RRF融合:
  fuse_by_rrf()      →    <5ms     (内存计算)

总耗时: max(110-230, 20-50) + 5 ≈ 115-235ms
                 (两路并行, 取最长)
```

---

## 下一步阅读

- [09 - Research深度研究](./09-Research深度研究.md)
- [10 - 多租户权限与套餐体系](./10-多租户权限与套餐体系.md)
- [API接口与最佳实践](./10-API接口与最佳实践.md)