# 云端部署

环境 ID：`cloud1-d5gt0lb2xcd72babe`

服务名称：`technical-analysis-api`

部署方式：CloudBase 云托管，从本地源码部署，运行环境选择 Dockerfile。

源码根目录必须包含：

- `Dockerfile`
- `.dockerignore`
- `scripts`
- `fae`
- `wechat-miniprogram/backend`

服务启动后检查：

- `/` 返回 `status: ok`
- `/health` 返回 `status: ok`
- `/api/v1/analyze?symbol=sh000001` 返回月、周、日三个周期
- `/api/v1/moving-average?symbol=sh000001&years=3` 返回均线分析
- `/api/v1/indicators?symbol=sh000001&indicator=macd&years=3` 返回指标分析
- `/api/v1/chan?symbol=sh000001&years=3` 返回缠中说禅 Beta 结构

小程序通过 `wx.cloud.callContainer` 访问服务，不需要配置 request 合法域名。
