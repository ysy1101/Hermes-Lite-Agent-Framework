"""
工具注册中心模块
支持工具的动态注册、移除与调用
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional


class BaseTool(ABC):
    """工具基类 — 所有工具必须实现此协议"""

    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称，对应 OpenAI function calling 的 name 字段"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """工具描述"""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """工具参数 JSON Schema"""
        ...

    @abstractmethod
    def execute(self, args: dict) -> str:
        """执行工具逻辑，返回字符串结果"""
        ...

    def validate(self, args: dict) -> bool:
        """验证参数是否满足 required 字段"""
        required = self.parameters.get("required", [])
        return all(r in args for r in required)

    def to_openai_schema(self) -> dict:
        """导出为 OpenAI function calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """工具注册中心

    支持运行时动态注册/移除工具，并导出为 OpenAI function calling 格式。
    """

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """注册一个工具。同名工具将被覆盖。"""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> bool:
        """移除一个工具。返回 True 表示成功移除。"""
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def get(self, name: str) -> Optional[BaseTool]:
        """获取指定名称的工具"""
        return self._tools.get(name)

    def list(self) -> List[str]:
        """列出所有已注册的工具名称"""
        return list(self._tools.keys())

    def execute(self, name: str, args: dict) -> str:
        """按名称执行工具（与旧 execute_tool 签名兼容）"""
        tool = self._tools.get(name)
        if not tool:
            return f"错误：未知工具 {name}"
        try:
            return tool.execute(args)
        except Exception as e:
            return f"执行 {name} 时出错：{str(e)}"

    def export_openai_tools(self) -> List[dict]:
        """导出所有工具为 OpenAI function calling 格式"""
        return [t.to_openai_schema() for t in self._tools.values()]

    def clear(self) -> None:
        """清空所有已注册的工具"""
        self._tools.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)
