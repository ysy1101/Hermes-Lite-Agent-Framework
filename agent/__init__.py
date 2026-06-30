"""
Mini-Hermes Agent Runtime
轻量级 Agent 运行时框架 — RAG 知识库问答智能体
"""

from .memory import MemoryManager
from .skills import SkillManager
from .document import parse_file, chunk_text, load_documents
from .retriever import VectorRetriever, build_index
from .tools import TOOLS, execute_tool, registry
from .loop import RAGAgent

# 新增模块导出
from .state import AgentState, AgentStatus
from .registry import BaseTool, ToolRegistry
from .trace import Trace, TraceStep
from .workflow import (
    Workflow,
    WorkflowStep,
    WorkflowEngine,
    create_knowledge_inspection_workflow,
)
from .eval import RAGEvaluator, create_sample_test_cases

__all__ = [
    # 原有模块
    "MemoryManager",
    "SkillManager",
    "parse_file",
    "chunk_text",
    "load_documents",
    "VectorRetriever",
    "build_index",
    "TOOLS",
    "execute_tool",
    "RAGAgent",
    "registry",
    # 新增模块
    "AgentState",
    "AgentStatus",
    "BaseTool",
    "ToolRegistry",
    "Trace",
    "TraceStep",
    "Workflow",
    "WorkflowStep",
    "WorkflowEngine",
    "create_knowledge_inspection_workflow",
    "RAGEvaluator",
    "create_sample_test_cases",
]
