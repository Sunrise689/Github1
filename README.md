# Technical Analysis Master | 技术分析大师

A full-stack WeChat Mini Program for explainable, multi-timeframe market-structure study.

[English](#english) · [简体中文](#简体中文)

## English

### What it does

- Studies monthly, weekly, and daily K-line structure in one workflow.
- Draws detected patterns, trend lines, support and resistance on interactive charts.
- Includes moving-average tools, MACD/KDJ/RSI/SuperTrend indicators, a Chan Theory beta page, and a technical-analysis knowledge library.
- Explains pattern definitions, evidence, and confirmation conditions using deterministic rules.
- Combines a native WeChat frontend with a Python FastAPI analysis service, CloudBase container calls, and the FAE/QTE analysis modules.

### Project layout

- `wechat-miniprogram/miniprogram/` — WeChat Mini Program pages, components, charting, and services.
- `wechat-miniprogram/backend/` — FastAPI endpoints and supporting analysis services.
- `scripts/`, `fae/judgment/`, and `qte/` — runtime analysis engines and supporting data used by the backend.
- `Dockerfile` — container build for the backend service.

### Run locally

1. Open `wechat-miniprogram/` in WeChat Developer Tools (its `project.config.json` is in that directory).
2. Install the backend dependencies and start the API:

   ```bash
   cd wechat-miniprogram
   python -m pip install -r backend/requirements.txt
   python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
   ```

3. For device or cloud use, configure an authorized HTTPS or CloudBase endpoint in `wechat-miniprogram/miniprogram/config.js`. A local `127.0.0.1` address is only reachable from the development machine.

### Notes

This project is for technical-analysis learning and research. Its “AI recognition” refers to objective, rule-based pattern matching; it does not generate price forecasts or trading instructions. Chan Theory features are marked beta.

## 简体中文

### 项目简介

技术分析大师是一款用于学习和研究行情结构的微信小程序，提供可解释的多周期技术分析。

### 主要特点

- 在同一工作流中查看月线、周线、日线结构。
- 在交互式图表上展示形态、趋势线、支撑位和阻力位。
- 提供均线工具、MACD/KDJ/RSI/SuperTrend 指标、缠中说禅 Beta 页面和技术分析知识库。
- 通过规则化逻辑说明形态定义、识别证据和确认条件。
- 采用微信原生前端，并通过 CloudBase 云托管连接 Python FastAPI 分析服务，后端集成 FAE 与 QTE 分析模块。

### 目录结构

- `wechat-miniprogram/miniprogram/`：小程序页面、组件、图表和请求服务。
- `wechat-miniprogram/backend/`：FastAPI 接口和配套分析服务。
- `scripts/`、`fae/judgment/`、`qte/`：后端运行依赖的分析引擎与数据。
- `Dockerfile`：后端容器构建配置。

### 本地运行

1. 在微信开发者工具中导入 `wechat-miniprogram/`（该目录内包含 `project.config.json`）。
2. 安装后端依赖并启动 API：

   ```bash
   cd wechat-miniprogram
   python -m pip install -r backend/requirements.txt
   python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
   ```

3. 真机或云端联调时，在 `wechat-miniprogram/miniprogram/config.js` 中配置有权限的 HTTPS 或 CloudBase 服务地址；本机的 `127.0.0.1` 只能由开发机访问。

### 使用说明

本项目用于技术分析学习与研究，不构成投资建议。“AI 识别”指客观、规则化的形态匹配，不生成价格预测或交易指令。缠论相关页面标注为 Beta。

