# src/k8s_templates.py

server_template = """apiVersion: v1
kind: ConfigMap
metadata:
  name: diagnose-config-{{ cfg.job_name }}
  namespace: {{ cfg.k8s.namespace }}
data:
  config.yaml: |
    {{ cfg | tojson }}
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server-{{ cfg.job_name }}
  namespace: {{ cfg.k8s.namespace }}
  labels:
    app: vllm-server-{{ cfg.job_name }}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-server-{{ cfg.job_name }}
  template:
    metadata:
      labels:
        app: vllm-server-{{ cfg.job_name }}
    spec:
      containers:
        - name: vllm-server
          image: {{ cfg.docker.server_image }}
          command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
          args:
            - "--model={{ cfg.model.name }}"
            - "--max-model-len={{ cfg.model.max_model_len }}"
            - "--tensor-parallel-size={{ cfg.k8s.server.gpu_count }}"
            - "--max-num-seqs={{ cfg.model.max_num_seqs }}"
            - "--max-num-batched-tokens={{ cfg.model.max_num_batched_tokens }}"
            {% if cfg.model.enable_chunked_prefill %}- "--enable-chunked-prefill"{% endif %}
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: "{{ cfg.k8s.server.gpu_count }}"
              memory: "{{ cfg.k8s.server.memory_limit }}"
            requests:
              nvidia.com/gpu: "{{ cfg.k8s.server.gpu_count }}"
          volumeMounts:
            - name: dshm
              mountPath: /dev/shm
            - name: config
              mountPath: /app/config
          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: HF_TOKEN
            - name: CONFIG_PATH
              value: /app/config/config.yaml
      imagePullSecrets:
        - name: {{ cfg.k8s.image_pull_secrets }}
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
        - name: config
          configMap:
            name: diagnose-config-{{ cfg.job_name }}
      nodeSelector:
        gpu: {{ cfg.k8s.server.gpu_type }}
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-server-{{ cfg.job_name }}
  namespace: {{ cfg.k8s.namespace }}
spec:
  selector:
    app: vllm-server-{{ cfg.job_name }}
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
"""

client_template = """apiVersion: v1
kind: ConfigMap
metadata:
  name: diagnose-config-{{ cfg.job_name }}
  namespace: {{ cfg.k8s.namespace }}
data:
  config.yaml: |
    {{ cfg | tojson }}
---
apiVersion: batch/v1
kind: Job
metadata:
  name: diagnose-prediction-{{ cfg.job_name }}
  namespace: {{ cfg.k8s.namespace }}
spec:
  template:
    spec:
      containers:
        - name: inference-client
          image: {{ cfg.docker.registry }}/{{ cfg.docker.image_name }}:{{ cfg.docker.tag }}
          command: ["python3", "-u", "-m", "src.main", "--config", "/app/config/config.yaml"]
          volumeMounts:
            - name: config
              mountPath: /app/config
          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: HF_TOKEN
            - name: WANDB_API_KEY
              valueFrom:
                secretKeyRef:
                  name: wandb-secret
                  key: api-key
          resources:
            limits:
              memory: "{{ cfg.k8s.client.memory_limit }}"
      volumes:
        - name: config
          configMap:
            name: diagnose-config-{{ cfg.job_name }}
      imagePullSecrets:
        - name: {{ cfg.k8s.image_pull_secrets }}
      restartPolicy: Never
  backoffLimit: 0
"""