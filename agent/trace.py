"""
Trace 执行轨迹记录模块
记录 Agent 推理过程、工具调用结果与执行轨迹，支持调试与回放
"""
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class TraceStep:
    """单步推理记录"""
    step: int
    timestamp: str
    thought: str           # 模型在工具调用前的思考内容
    action: str            # 工具名称
    action_args: dict      # 工具参数
    observation: str       # 工具返回结果（截断前 1000 字符）
    duration_ms: float     # 工具执行耗时（毫秒）
    token_usage: Optional[dict] = None  # API 返回的 usage 信息


class Trace:
    """Agent 执行轨迹记录器

    记录每一步 ReAct 推理的 Thought → Action → Observation，
    支持导出为 JSON 供调试和回放使用。

    当 enabled=False 时所有方法为空操作，性能零影响。
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.steps: List[TraceStep] = []
        self._start_time: float = 0.0
        self.final_answer: str = ""

    def start(self) -> None:
        """开始记录（重置之前的数据）"""
        if not self.enabled:
            return
        self._start_time = time.time()
        self.steps = []
        self.final_answer = ""

    def record_step(
        self,
        step: int,
        thought: str,
        action: str,
        action_args: dict,
        observation: str,
        duration_ms: float,
        token_usage: Optional[dict] = None,
    ) -> None:
        """记录一个推理步骤"""
        if not self.enabled:
            return
        obs_truncated = str(observation)[:1000]
        self.steps.append(TraceStep(
            step=step,
            timestamp=datetime.now().isoformat(),
            thought=thought,
            action=action,
            action_args=action_args,
            observation=obs_truncated,
            duration_ms=round(duration_ms, 2),
            token_usage=token_usage,
        ))

    def record_final(self, answer: str) -> None:
        """记录最终回答"""
        if self.enabled:
            self.final_answer = answer

    def export(self) -> dict:
        """导出为字典格式"""
        total_duration = (time.time() - self._start_time) * 1000 if self._start_time else 0
        return {
            "enabled": self.enabled,
            "start_time": datetime.fromtimestamp(self._start_time).isoformat() if self._start_time else "",
            "total_steps": len(self.steps),
            "total_duration_ms": round(total_duration, 2),
            "final_answer": self.final_answer,
            "steps": [
                {
                    "step": s.step,
                    "timestamp": s.timestamp,
                    "thought": s.thought,
                    "action": s.action,
                    "action_args": s.action_args,
                    "observation": s.observation,
                    "duration_ms": s.duration_ms,
                    "token_usage": s.token_usage,
                }
                for s in self.steps
            ],
        }

    def to_json(self) -> str:
        """导出为 JSON 字符串"""
        return json.dumps(self.export(), ensure_ascii=False, indent=2)

    def summary(self) -> str:
        """返回人类可读的摘要"""
        if not self.enabled or not self.steps:
            return "（无 Trace 数据）"
        lines = [
            f"Trace: {len(self.steps)} 步, "
            f"最终回答长度: {len(self.final_answer)} 字符"
        ]
        for s in self.steps:
            keys = list(s.action_args.keys()) if s.action_args else []
            lines.append(
                f"  Step {s.step}: {s.action}({keys}) -> {s.duration_ms}ms"
            )
        return "\n".join(lines)
