"""
工具定义与执行模块（RAG 知识库版）
"""
import os
import json
from typing import Dict, Any

from .document import parse_file, chunk_text, load_documents
from .retriever import VectorRetriever, build_index


# 全局检索器实例
_retriever: VectorRetriever = None


def get_retriever(persist_dir: str = "chroma_db") -> VectorRetriever:
    """获取检索器实例（单例）"""
    global _retriever
    if _retriever is None:
        _retriever = VectorRetriever(persist_dir=persist_dir)
    return _retriever


def tool_search_knowledge(query: str, top_k: int = 5) -> str:
    """搜索知识库"""
    retriever = get_retriever()
    if not retriever.available:
        return "ChromaDB 未安装，无法检索。请先运行索引构建。"
    
    results = retriever.search(query, top_k=top_k)
    
    if not results:
        return "未找到相关内容。知识库可能为空，请先构建索引。"
    
    output = []
    for i, (doc, source, score) in enumerate(results, 1):
        output.append(f"【结果 {i}】来源：{source} | 相关度：{score:.2%}")
        output.append(doc[:500])
        output.append("---")
    
    return "\n".join(output)


def tool_read_document(filepath: str) -> str:
    """读取单个文档"""
    if not os.path.exists(filepath):
        return f"文件不存在：{filepath}"
    
    content = parse_file(filepath)
    return content[:3000]


def tool_list_documents(data_dir: str = "data") -> str:
    """列出知识库中的所有文档"""
    docs = load_documents(data_dir)
    
    if not docs:
        return f"知识库为空（{data_dir}/ 目录下没有文档）。请将文档放入该目录后构建索引。"
    
    lines = []
    for doc in docs:
        lines.append(f"- {doc['filename']} ({doc['chunk_count']} 个文本块)")
    
    return "\n".join(lines)


def tool_build_index(data_dir: str = "data", persist_dir: str = "chroma_db") -> str:
    """构建/重建知识库索引"""
    return build_index(data_dir, persist_dir)


def tool_get_stats() -> str:
    """查看知识库统计信息"""
    retriever = get_retriever()
    stats = retriever.get_stats()
    return json.dumps(stats, ensure_ascii=False, indent=2)


# ---- 工具注册中心集成 ----

from .registry import BaseTool, ToolRegistry


class SearchKnowledgeTool(BaseTool):
    name = "search_knowledge"
    description = "在知识库中语义搜索相关内容。当你需要回答用户问题时，先用此工具检索相关文档。"
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索查询，用自然语言描述你想找什么"
            },
            "top_k": {
                "type": "integer",
                "description": "返回结果数量，默认 5"
            }
        },
        "required": ["query"]
    }

    def execute(self, args: dict) -> str:
        return tool_search_knowledge(**args)


class ReadDocumentTool(BaseTool):
    name = "read_document"
    description = "读取知识库中某个文档的完整内容"
    parameters = {
        "type": "object",
        "properties": {
            "filepath": {
                "type": "string",
                "description": "文档路径，如 data/xxx.md"
            }
        },
        "required": ["filepath"]
    }

    def execute(self, args: dict) -> str:
        return tool_read_document(**args)


class ListDocumentsTool(BaseTool):
    name = "list_documents"
    description = "列出知识库中所有文档"
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    def execute(self, args: dict) -> str:
        return tool_list_documents(**args)


class BuildIndexTool(BaseTool):
    name = "build_index"
    description = "构建或重建知识库的向量索引。添加新文档后需要调用此工具。"
    parameters = {
        "type": "object",
        "properties": {
            "data_dir": {
                "type": "string",
                "description": "文档目录，默认 data"
            }
        },
        "required": []
    }

    def execute(self, args: dict) -> str:
        return tool_build_index(**args)


class GetStatsTool(BaseTool):
    name = "get_stats"
    description = "查看知识库统计信息，了解有多少文档和文本块"
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    def execute(self, args: dict) -> str:
        return tool_get_stats()


# 模块级全局注册中心
registry = ToolRegistry()
registry.register(SearchKnowledgeTool())
registry.register(ReadDocumentTool())
registry.register(ListDocumentsTool())
registry.register(BuildIndexTool())
registry.register(GetStatsTool())

# 向后兼容：TOOLS 和 TOOL_FUNCTIONS 从 registry 生成
TOOLS = registry.export_openai_tools()

TOOL_FUNCTIONS = {
    name: tool.execute
    for name, tool in registry._tools.items()
}


def execute_tool(name: str, args: Dict[str, Any]) -> str:
    """执行工具调用（委托给 registry）"""
    return registry.execute(name, args)