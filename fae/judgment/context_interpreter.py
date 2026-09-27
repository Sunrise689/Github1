"""Interpret objective pattern names in trend and location context."""

from __future__ import annotations

from typing import Any, ClassVar

from .schemas import DetectedSignal, Direction


class ContextInterpreter:
    """Separate objective detection from context-dependent meaning."""

    _BOTTOM_REVERSALS: ClassVar[set[str]] = {
        "hammer",
        "morning_star",
        "bullish_abandoned_baby",
        "bullish_engulfing",
        "red_three_soldiers",
        "double_bottom",
        "triple_bottom",
        "head_and_shoulders_bottom",
        "rounding_bottom",
        "v_bottom",
        "island_reversal_bottom",
        "dormant_bottom",
    }
    _TOP_REVERSALS: ClassVar[set[str]] = {
        "hanging_man",
        "shooting_star",
        "evening_star",
        "bearish_abandoned_baby",
        "bearish_engulfing",
        "three_black_crows",
        "double_top",
        "triple_top",
        "head_and_shoulders_top",
        "rounding_top",
        "inverted_v_top",
        "island_reversal_top",
    }

    def interpret(
        self,
        signal: DetectedSignal,
        *,
        trend: str,
        position_pct: float,
        ma_state: str = "unknown",
    ) -> dict[str, Any]:
        """Return interpretation, direction, and confidence adjustment."""
        pattern = signal.pattern
        interpretation = "形态成立，等待上下文确认"
        adjustment = 0.0
        direction = signal.direction
        if (
            pattern == "red_three_soldiers"
            and trend == "down"
        ):
            interpretation = "下降趋势中的反弹观察，不能直接解释为反转"
            adjustment = -0.25
            direction = Direction.NEUTRAL
        elif signal.metadata.get("family") == "cup_with_handle":
            interpretation = "圆弧杯底后的浅幅柄部整理，等待杯口阻力有效突破"
        elif signal.metadata.get("family") == "base":
            interpretation = "下跌后的狭窄低量筑底，突破前只作为潜伏底候选"
        elif pattern in self._BOTTOM_REVERSALS:
            if position_pct <= 0.4 and trend in {"down", "range"}:
                interpretation = "低位潜在反转，需突破或后续K线确认"
                adjustment = 0.08
            elif position_pct >= 0.7:
                interpretation = "位置偏高，底部型解释降级"
                adjustment = -0.2
        elif pattern in self._TOP_REVERSALS:
            if position_pct >= 0.6 and trend in {"up", "range"}:
                interpretation = "高位潜在反转，需破位确认"
                adjustment = 0.08
            elif position_pct <= 0.3:
                interpretation = "位置偏低，顶部型解释降级"
                adjustment = -0.2
        elif signal.metadata.get("family") == "consolidation":
            interpretation = "整理方向由边界突破决定，量度目标仅作辅助"
        if ma_state == "bearish" and direction == Direction.BULLISH:
            adjustment -= 0.1
            interpretation += "；均线背景偏弱"
        elif ma_state == "bullish" and direction == Direction.BEARISH:
            adjustment -= 0.1
            interpretation += "；均线背景偏强"
        return {
            "pattern": pattern,
            "objective_detection_preserved": True,
            "direction": direction.value,
            "interpretation": interpretation,
            "confidence_adjustment": adjustment,
        }
