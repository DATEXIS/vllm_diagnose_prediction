# Build with the repository root as context (not `UI/`), otherwise paths like
# `COPY UI/dashboard/...` and `COPY config.json` will not exist in the context.
#
#   cd /path/to/vllm_diagnose_prediction
#   docker build --platform linux/amd64 -f UI/Dockerfile -t registry.datexis.com/yxsg9647/merlin_2_ui:latest .
#
# On the cluster, `WANDB_API_KEY` is injected via the Deployment secret; the baked
# `config.json` is optional for local/offline convenience (see `pythonFetch.ts`).

FROM node:20-bookworm AS builder

WORKDIR /app/dashboard

COPY UI/dashboard/package.json UI/dashboard/package-lock.json ./
RUN npm ci

COPY UI/dashboard/ ./
RUN npm run build

FROM node:20-bookworm-slim

WORKDIR /app

RUN apt-get update \
  && apt-get install -y --no-install-recommends python3 python3-pip \
  && rm -rf /var/lib/apt/lists/*

COPY UI/py/requirements-fetch.txt /tmp/requirements-fetch.txt
RUN pip3 install --no-cache-dir --break-system-packages \
  -r /tmp/requirements-fetch.txt \
  && rm /tmp/requirements-fetch.txt

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME=0.0.0.0
ENV MERLIN_UI_ROOT=/app/UI
ENV MERLIN_CONFIG_JSON=/app/config.json

COPY --from=builder /app/dashboard/.next/standalone ./
COPY --from=builder /app/dashboard/.next/static ./.next/static

COPY UI/ICD_names ./UI/ICD_names
COPY UI/py ./UI/py

RUN mkdir -p ./UI/wandb_cache && chmod -R u+rwx ./UI

COPY config.json ./config.json

EXPOSE 3000

CMD ["node", "server.js"]
