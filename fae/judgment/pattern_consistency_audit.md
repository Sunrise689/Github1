# FAE 形态一致性审计报告

审计时间：自动生成（本地文件扫描）  
审计范围：`chart_patterns.py`、`expert_rules.json`、`adapters.py`、`textbook_case_compiler.py`、`case_library_index.json`、`frontend_pattern_knowledge_prompt.md`。  
本报告只读比较现有层，不修改检测器、规则或案例。

## 结论摘要

- 图形检测器候选 ID：**25** 个；源码显式 `family`：**9** 个。
- 图形规则覆盖 ID：**25** 个；规则文件总数：**21** 条。
- 图形 canonical 候选集合：**21** 个。
- 案例索引总量：**1601** 条；其中图形/变体相关 ID：**26** 个。
- 当前最重要的结构性问题是：案例编译和图形检测都保留了若干历史变体 ID，而 canonical 化主要发生在适配/编译路径；Registry 应把这些变体正式声明为“主 ID + variant”，避免规则、案例和前端各自解释。

## 1. 检测器清单

### ChartPatternDetector 输出

- `ascending_flag`
- `ascending_triangle`
- `broadening_triangle`
- `compound_head_and_shoulders_bottom`
- `compound_head_and_shoulders_top`
- `cup_with_handle`
- `descending_flag`
- `descending_triangle`
- `dormant_bottom`
- `double_bottom`
- `double_top`
- `falling_wedge`
- `head_and_shoulders_bottom`
- `head_and_shoulders_top`
- `inverted_v_top`
- `island_reversal_bottom`
- `island_reversal_top`
- `rectangle`
- `rising_wedge`
- `rounding_bottom`
- `rounding_top`
- `symmetrical_triangle`
- `triple_bottom`
- `triple_top`
- `v_bottom`

### Detector family（源码）

- `base`
- `consolidation`
- `cup_with_handle`
- `flag`
- `head_and_shoulders`
- `island`
- `multiple_top_bottom`
- `rounding`
- `v_reversal`

### 源码中可直接看到的 `_signal` 字面量

- `cup_with_handle`
- `dormant_bottom`
- `island_reversal_top`
- `rounding_bottom`
- `v_bottom`

动态 f-string（如 `double_{suffix}`、`{compound_}head_and_shoulders_{suffix}`）已通过显式清单补入审计。

## 1.1 Registry 覆盖

- Registry 条目：**102** 个。
- 冲突规则：**21** 条。
- 按类别：`{"simple_candlestick": 14, "combination_candlestick": 40, "trend_candlestick": 13, "gap": 11, "chart_pattern": 20, "structure": 4}`
- 按状态：`{"implemented": 89, "partial": 12, "missing": 1}`

### 检测器 canonical ID 未登记

- 无

### Registry 中明确缺失检测器的条目

- `inverted_cup_with_handle`

### 冲突规则引用但 Registry 中不存在的 ID

- 无

## 2. 规则覆盖

### `chart_*` 规则涉及的 ID

- `ascending_flag`
- `ascending_triangle`
- `broadening_triangle`
- `compound_head_and_shoulders_bottom`
- `compound_head_and_shoulders_top`
- `cup_with_handle`
- `descending_flag`
- `descending_triangle`
- `dormant_bottom`
- `double_bottom`
- `double_top`
- `falling_wedge`
- `head_and_shoulders_bottom`
- `head_and_shoulders_top`
- `inverted_v_top`
- `island_reversal_bottom`
- `island_reversal_top`
- `rectangle`
- `rising_wedge`
- `rounding_bottom`
- `rounding_top`
- `symmetrical_triangle`
- `triple_bottom`
- `triple_top`
- `v_bottom`

### 规则引用但当前没有对应检测器/兼容映射

- 无

### 已有 canonical 检测器但没有 `chart_*` 规则

- 无

## 3. 案例库与编译器

### 案例索引中的图形/变体 ID

- `ascending_flag`
- `ascending_triangle`
- `broadening_triangle`
- `cup_with_handle`
- `descending_flag`
- `descending_triangle`
- `diamond`
- `double_bottom`
- `double_top`
- `falling_wedge`
- `head_and_shoulders_bottom`
- `head_and_shoulders_top`
- `inverted_cup_with_handle`
- `inverted_v_top`
- `island_reversal_bottom`
- `island_reversal_top`
- `pennant`
- `rectangle`
- `rising_wedge`
- `rounding_bottom`
- `rounding_top`
- `symmetrical_triangle`
- `triple_bottom`
- `triple_top`
- `v_bottom`
- `v_top`

### 案例索引中的历史变体（应由 Registry 的 `variant_of`/兼容别名承接）

- `diamond`
- `pennant`
- `triple_bottom`
- `triple_top`
- `v_top`

### 编译器 `CHART_PATTERNS` 中的历史变体

- `diamond`
- `pennant`
- `triple_bottom`
- `triple_top`
- `v_top`

### 无法归入当前图形检测器或显式变体映射的案例 ID

- `inverted_cup_with_handle`

案例库存在大量 K 线、趋势线和指标类 ID；它们不应被误报为 chart pattern 缺失。后续 Registry 应按 category 分开登记。

## 4. 前端契约

### 前端提示词中出现的相关 ID

- `cup_with_handle`
- `diamond`
- `dormant_bottom`
- `double_bottom`
- `double_top`
- `head_and_shoulders_bottom`
- `head_and_shoulders_top`
- `inverted_v_top`
- `pennant`
- `rectangle`
- `symmetrical_triangle`
- `triple_bottom`
- `triple_top`
- `v_bottom`
- `v_top`

### 前端声称 canonical、但检测器当前没有输出的 ID

- 无

### 别名键与 canonical ID 同名（需要 Registry 明确优先级）

- 无

前端应始终以序列化的 `pattern_id` 为统计/去重键，以 `variant` 保存多重、复合、三角旗等历史称呼；旧 `pattern` 字段只作兼容展示。

## 5. 已确认的归一化关系

| 历史/变体 ID | canonical ID | 处理方式 |
|---|---|---|
| `compound_head_and_shoulders_bottom` | `head_and_shoulders_bottom` | 作为 variant/兼容输入，不单独建立主形态 |
| `compound_head_and_shoulders_top` | `head_and_shoulders_top` | 作为 variant/兼容输入，不单独建立主形态 |
| `cup_handle` | `cup_with_handle` | 作为 variant/兼容输入，不单独建立主形态 |
| `diamond` | `rectangle` | 作为 variant/兼容输入，不单独建立主形态 |
| `latent_bottom` | `dormant_bottom` | 作为 variant/兼容输入，不单独建立主形态 |
| `pennant` | `symmetrical_triangle` | 作为 variant/兼容输入，不单独建立主形态 |
| `triple_bottom` | `double_bottom` | 作为 variant/兼容输入，不单独建立主形态 |
| `triple_top` | `double_top` | 作为 variant/兼容输入，不单独建立主形态 |
| `v_top` | `inverted_v_top` | 作为 variant/兼容输入，不单独建立主形态 |
|

## 6. 风险与建议

1. `triple_top`/`triple_bottom`、复合头肩形、`pennant`、`diamond`、`v_top` 等名称仍可能出现在旧案例或规则边界；Registry 应集中声明 canonical ID、alias、variant，而不是继续扩展检测器输出数量。
2. `chart_triple_top_bottom` 规则目前仍把三重顶/底作为 `applies_to`，但用户确认它们属于双顶/底的多重变体；Registry 建立后应把规则引用改成 canonical `double_top`/`double_bottom`，保留 `variant=multiple`。
3. `inverted_cup_with_handle` 在案例编译器/案例索引中存在，但当前图形检测器没有对应实现；本阶段标记为 `case_only/needs_review`，不自动伪造检测器。
4. 结构层（趋势线、支撑/阻力、Pivot）和缺口层不应混入 chart pattern 统计；Registry 需用 category 进行硬隔离。
5. 建议下一步按本报告冻结 canonical 集合，再建立 `pattern_registry.json` 和 `conflict_rules.json`，最后让规则、案例检索与前端都只消费 Registry。

## 7. 机器可复核字段

- 变体映射：`textbook_case_compiler.py::CANONICAL_PATTERN_VARIANTS`
- 输入别名：`adapters.py::_ALIASES`
- Registry：`pattern_registry.json`
- Registry 校验：`validate_registry.py`
- 图形规则：`expert_rules.json` 中 `rule_id` 以 `chart_` 开头的条目
- 报告生成脚本：`pattern_consistency_audit.py`
