# 01 - RAG 概述与核心概念

> **RAG（Retrieval-Augmented Generation）检索增强生成**  
> 一种让大模型在生成答案前先从外部知识库中检索相关信息的技术，显著提升回答的准确性、时效性和可追溯性。

---

## 1. 什么是 RAG？

在没有 RAG 的情况下，大语言模型（LLM）只能依赖训练时学到的知识来回答问题，存在以下问题：

| 问题 | 描述 |
|------|------|
| **时效性差** | 模型只有训练截止日期前的知识 |
| **幻觉** | 模型可能编造不存在的事实 |
| **领域不足** | 对垂直领域（如公司内部制度、法律法规）了解有限 |
| **不可追溯** | 无法验证回答的来源是否可靠 |

RAG 的解决方案很简单：

```
用户提问 → 从知识库检索相关文档 → 将检索结果作为上下文喂给LLM → LLM生成带引用的答案
```

## 2. RAG 的核心流程

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  ① 摄入   │ → │  ② 索引   │ → │  ③ 检索   │ → │  ④ 生成   │
│ Ingestion │    │ Indexing  │    │ Retrieval │    │Generation │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
     │               │               │               │
  文档上传       分块+向量化      相似度搜索       LLM生成答案
  格式解析       存入向量库      关键词匹配       证据引用
  内容提取       建立倒排索引    结果融合        可追溯输出
```

### ① 摄入（Ingestion）
- 支持 `.md`、`.txt`、`.pdf`、`.docx` 等多种格式
- 自动识别文件类型，调用对应解析器
- 将二进制文件转换为纯文本 Markdown

### ② 索引（Indexing）
- **智能分块**：按标题层级、段落、代码块、表格等结构切分
- **向量化**：用 Embedding 模型将文本转为高维向量（如 1024 维）
- **存储**：向量存入 pgvector，原文存入 PostgreSQL

### ③ 检索（Retrieval）
- **语义检索**：向量余弦距离计算相似度
- **关键词检索**：BM25 算法精确匹配
- **混合检索**：两路并行 + RRF 融合，取长补短

### ④ 生成（Generation）
- 将检索到的 chunks 作为上下文
- 拼入 Prompt 发送给 LLM
- LLM 基于上下文生成答案，注明来源

---

## 3. rag-center 的 RAG 架构

```
┌──────────────────────────────────────────────────────────┐
│                      API 层 (FastAPI)                      │
│   auth | knowledge-bases | documents | rag/retrieve       │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────┼────────────────────────────────┐
│                   服务层 (Services)                        │
│   RagService       ResearchService     IndexingService   │
│   HybridSearch     EvidenceOrchestration                  │
│   GraphExpansion   ContextExpansion                       │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────┼────────────────────────────────┐
│                  提供者层 (Providers)                      │
│   Embedding │ VectorStore │ ReRank │ KeywordSearch │ LLM  │
│   (OpenAI)  │  (pgvector) │(Qwen3)│(Elasticsearch)│     │
└─────────────────────────┴────────────────────────────────┘

数据存储:
  PostgreSQL + pgvector  →  向量 + 结构化数据
  Elasticsearch          →  关键词倒排索引
  Redis                  →  限流计数器
  Celery                 →  异步索引任务
```

---

## 4. 本项目 RAG 的独特优势

### 4.1 层次化分块（v2 索引架构）

并非简单按字符数切分，而是：

```
原始文档
  ├── # 一级标题 → section_id = UUID5(doc_id + version + "一级标题")
  │   ├── 段落块 (prose)
  │   ├── 代码块 (code)
  │   └── ## 二级标题 → section_id = UUID5(...)
  │       ├── 表格块 (table) + 表头摘要
  │       └── 段落块
  └── # 另一个一级标题
      └── ...
```

每个 chunk 包含：
- `content`：原始内容
- `retrieval_text`：增强检索文本（`文档：xxx\n章节：A > B > C\n正文：xxx`）
- `section_id`：稳定可寻址的章节 ID
- `parent_section_id`：父级章节，支持向上追溯上下文
- `order_index`：章节内顺序，支持前后相邻块扩展

### 4.2 文档引用图（Graph Injection）

```
文档 A section X 中引用 "详见文档 B 第 Y 条"
         │
         ▼ chunk_relations
  source_chunk_id (A-X) → relation_type="reference" → target_chunk_id (B-Y)
         │
         ▼ 检索时
  命中 chunk A-X → 自动注入 chunk B-Y 作为候选
```

### 4.3 多轮 Research 能力

不是简单的一次检索就结束，而是：

```
Planner(规划) → Retrieve(第1轮) → Decider(判断是否完整)
                                      │
                            ┌─ 完整 → Finalizer(生成答案)
                            │
                            └─ 不完整 → Supplement(补充检索)
                                              │
                                              └→ Finalizer(生成答案)
```

---

## 5. 检索 Profile 对比

| Profile | 向量 | BM25 | 融合 | ReRank | Evidence | 适用场景 |
|---------|:---:|:----:|:----:|:------:|:--------:|----------|
| **speed** | ✅ | ❌ | ❌ | ❌ | ❌ | 快速原型、实时对话 |
| **balanced** | ✅ | ✅ | RRF | ❌ | ❌ | 日常检索、一般精度需求 |
| **quality** | ✅ | ✅ | RRF | ✅ | ✅ | 高精度问答、合规审查 |
| **custom** | 自定义 | 自定义 | 自定义 | 自定义 | 自定义 | 深度调试、A/B 测试 |

---

## 6. 关键术语表

| 术语 | 英文 | 含义 |
|------|------|------|
| 知识库 | Knowledge Base (KB) | 一个独立的文档集合，用户组织知识的容器 |
| 文档 | Document | 上传到知识库的单个文件 |
| 分块 | Chunk | 文档切分后的最小检索单元 |
| 向量 | Vector/Embedding | 文本的数学表示，高维浮点数数组 |
| RRF | Reciprocal Rank Fusion | 将多路检索结果融合排序的算法 |
| ReRank | 重排序 | 用更精准的模型对初检结果重新打分 |
| Evidence | 证据编排 | LLM 从候选中筛选并分组支撑答案的证据 |
| Research | 深度研究 | 多轮自动检索+分析，生成完整答案+证据包 |

---

## 下一步阅读

- [02 - 知识库管理指南](./02-知识库管理指南.md)
- [03 - 文档上传与解析](./03-文档上传与解析.md)
- [04 - 智能分块策略](./04-智能分块策略.md)
- [05 - 向量化与索引](./05-向量化与索引.md)
- [06 - 检索模式详解](./06-检索模式详解.md)