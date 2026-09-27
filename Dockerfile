FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MPLCONFIGDIR=/tmp/matplotlib

# QTE 报告图优先使用随包的抖音美好体；Noto CJK 作为缺字与容灾回退。
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY wechat-miniprogram/backend/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt -i https://mirrors.cloud.tencent.com/pypi/simple --trusted-host mirrors.cloud.tencent.com \
    || pip install --no-cache-dir -r /tmp/requirements.txt

COPY scripts/*.py /app/scripts/
COPY fae /app/fae
COPY qte /app/qte
COPY wechat-miniprogram/backend /app/wechat-miniprogram/backend

WORKDIR /app/wechat-miniprogram
EXPOSE 80

CMD ["sh", "-c", "exec uvicorn backend.app:app --host 0.0.0.0 --port ${PORT:-80}"]
