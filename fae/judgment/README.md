# FAE Judgment Layer

本目录把“形态是什么”和“当前上下文中意味着什么”分开处理：

1. `EvidenceExtractor` 提取趋势、位置、ATR、缺口、量能和破位证据。
2. `QuantitativeGate` 先否决不可能、过密或几何不足的识别结果。
3. `ChartPatternDetector` 最多保留少量互不冲突的技术图形，并输出
   直线、颈线、箱体、杯柄或圆弧绘图几何。当前支持头肩/复合头肩、
   双重/多重顶底、圆弧顶底、V底/尖顶、岛形、收敛/上升/下降/扩散三角形、
   旗形、楔形、箱形整理、潜伏底和杯柄；三角旗是收敛三角的别名，历史
   菱形标签归入箱形整理。
4. `RuleEngine` 从 `expert_rules.json` 匹配教材规则。
5. `SignalLifecycle` 管理 detected/candidate/confirmed/degraded/
   invalidated/expired。
6. `ConflictResolver` 执行技术图形、K线形态与多时间框架优先级。
7. `CaseKnowledgeBase` 从教材结构化案例库检索同形态、同证据和同市场
   位置的历史案例；案例共识只能有限调整候选信号，不能覆盖已确认规则
   或绕过量化门控。
8. `FeedbackStore` 记录人工裁定，只提出复核建议，不自动改写规则。

`pattern_registry.json` 是上述各层共用的形态身份证目录。检测器可以保留
历史 raw ID，但进入规则、案例检索、序列化和前端契约前统一转换为
canonical `pattern_id`；多重顶底、复合头肩、三角旗等通过 `variant` 表示。
别名和变体由 `pattern_registry.py` 集中加载，避免适配器、案例库和编译器
分别维护同一套映射。

`conflict_rules.json` 是唯一的形态冲突表。`ConflictResolver` 先使用已经
通过的量化门控结果，再按该表处理形态互斥、确认优先级、反向反转降级和
结构去重；`JudgmentEngine` 不会让 Resolver 重复执行同一轮量化门控。

校验 Registry：

```powershell
python -m fae.judgment.validate_registry
python -m fae.judgment.pattern_consistency_audit
```

## 快速调用

```python
from fae.judgment import JudgmentEngine

engine = JudgmentEngine()
result = engine.evaluate(ohlcv_dataframe)
```

返回值中的 `case_matches` 给出相似教材案例及书名、PDF 页码；
`case_consensus` 给出有上限的方向共识。运行时只使用质量分不低于
0.65 的案例。

接入现有 `pattern_core_v7.py`：

```python
from fae.judgment import from_v7_events
from scripts.pattern_core_v7 import detect_all

signals = from_v7_events(detect_all(ohlcv_dataframe))
result = engine.evaluate(ohlcv_dataframe, signals)
```

量度目标位于技术图形信号的
`metadata["measurement_target"]`，始终标记为辅助参考，不进入方向
仲裁分数。

## 全量教材案例编译

```powershell
python -m fae.judgment.textbook_case_compiler
```

编译器逐页读取教材，文本层不可用时自动 OCR，并按页缓存以支持续跑。
公开产物 `cases/compiled_cases.jsonl` 只包含可执行结构字段，不保存教材
正文；无法形成明确判断的图例进入
`cases/case_compilation_review.jsonl`，供人工按书名和 PDF 页码复核。
