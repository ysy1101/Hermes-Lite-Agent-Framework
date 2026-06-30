# Mini-Hermes Agent Runtime | 轻量级 Agent 运行时框架

参考 Hermes Agent 架构实现轻量级 Agent Runtime，在 ReAct Agent 基础上引入 State 管理、Workflow 编排与工具注册机制，实现支持长期记忆与知识库检索的智能问答系统。

## 技术栈

Python / ChromaDB / OpenAI API / ReAct / RAG / Tool Calling / Workflow / State Management

## 核心工作

- 设计 AgentState 管理运行时上下文，实现多节点间状态共享与传递
- 基于 ReAct 实现 Thought → Action → Observation 推理循环，支持多轮工具调用
- 实现 Tool Registry，支持知识检索、文档读取、索引构建等工具动态注册与调用
- 基于 ChromaDB 构建 RAG 检索链路，实现 PDF/Markdown/TXT 文档向量化与语义搜索
- 实现 Trace 机制，记录 Agent 推理过程、工具调用结果与执行轨迹，支持调试与回放
- 兼容 OpenAI API 标准协议，支持 DeepSeek、OpenRouter、Ollama 等模型接入

## 安装

```bash
cd RAGagent

# 安装依赖
pip install -r requirements.txt

# 配置 API Key
cp .env.example .env
# 编辑 .env 文件，填入你的 API Key
```

## 使用

### 1. 添加文档

将你的文档放入 `data/` 目录：

```bash
cp ~/Documents/产品手册.pdf data/
cp ~/Documents/技术笔记.md data/
```

### 2. 构建索引

```bash
python main.py index
```

### 3. 开始问答

**交互模式：**
```bash
python main.py i
```

**单次查询：**
```bash
python main.py query "产品的主要功能有哪些？"
```

## 项目结构

```
mini-hermes-agent/
├── agent/
│   ├── __init__.py      # 模块导出
│   ├── loop.py          # Agent 主循环（ReAct 推理）
│   ├── state.py         # AgentState 运行时状态管理
│   ├── registry.py      # Tool Registry 工具注册中心
│   ├── trace.py         # Trace 执行轨迹记录
│   ├── workflow.py      # Workflow 编排引擎
│   ├── memory.py        # 长期记忆系统
│   ├── skills.py        # 技能系统
│   ├── document.py      # 文档解析（PDF/MD/TXT）
│   ├── retriever.py     # 向量检索（ChromaDB）
│   └── tools.py         # 5 个 RAG 工具
├── data/                # 文档存放
├── chroma_db/           # 向量索引（自动生成）
├── memory/              # 记忆文件
├── skills/              # 技能模板
├── main.py              # 主入口
├── requirements.txt     # 依赖
└── .env.example         # 配置模板
```

## 内置工具

| 工具 | 功能 |
|------|------|
| `search_knowledge` | 语义搜索知识库 |
| `read_document` | 读取文档完整内容 |
| `list_documents` | 列出所有文档 |
| `build_index` | 构建/重建索引 |
| `get_stats` | 查看知识库统计 |

## 架构

```
用户提问 → Agent Loop (ReAct) → Thought → Action → Tool Call → Observation → 生成回答
                 │
     ┌───────────┼───────────┐
     ▼           ▼           ▼
  Retriever   Memory     Skills
  (ChromaDB)  (Markdown)  (模板)
```

## 支持的 API

兼容任何 OpenAI 格式的 API：

- OpenAI / Azure OpenAI
- DeepSeek / 通义千问 / 豆包 / 智谱
- OpenRouter
- 本地 Ollama / vLLM

只需在 `.env` 中配置对应的 `OPENAI_BASE_URL` 和 `MODEL_NAME`。

## License

MIT

