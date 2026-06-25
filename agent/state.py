"""
Agent 运行时状态管理模块
支持多节点间状态共享与传递
"""
import json
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any


class AgentStatus(Enum):
    """Agent 生命周期状态"""
    IDLE = "idle"
    RUNNING = "running"
    THINKING = "thinking"
    CALLING_TOOL = "calling_tool"
    FINISHED = "finished"
    ERROR = "error"
    MAX_STEPS = "max_steps"


@dataclass
class AgentState:
    """Agent 运行时状态

    管理当前任务的全部上下文，包括对话消息、工具调用历史和执行状态。
    """

    task: str = ""
    step: int = 0
    max_steps: int = 10
    status: AgentStatus = AgentStatus.IDLE
    messages: List[Dict[str, Any]] = field(default_factory=list)
    tool_history: List[Dict[str, Any]] = field(default_factory=list)
    final_answer: str = ""
    start_time: str = ""

    def reset(self, task: str = "") -> None:
        """重置状态，准备执行新任务"""
        self.task = task
        self.step = 0
        self.status = AgentStatus.IDLE
        self.messages = []
        self.tool_history = []
        self.final_answer = ""
        self.start_time = datetime.now().isoformat()

    def record_tool_call(self, name: str, args: dict, result: str) -> None:
        """记录一次工具调用（结果截断以控制内存）"""
        self.tool_history.append({
            "tool_name": name,
            "tool_args": args,
            "result_preview": str(result)[:200],
        })

    def to_dict(self) -> dict:
        """导出为字典，用于序列化和调试"""
        return {
            "task": self.task,
            "step": self.step,
            "max_steps": self.max_steps,
            "status": self.status.value,
            "messages_count": len(self.messages),
            "tool_history": self.tool_history,
            "final_answer": self.final_answer,
            "start_time": self.start_time,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentState":
        """从字典恢复状态（用于 trace 回放）"""
        state = cls(
            task=data.get("task", ""),
            step=data.get("step", 0),
            max_steps=data.get("max_steps", 10),
            status=AgentStatus(data.get("status", "idle")),
            final_answer=data.get("final_answer", ""),
            start_time=data.get("start_time", ""),
        )
        state.tool_history = data.get("tool_history", [])
        return state

    def to_json(self) -> str:
        """导出为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
