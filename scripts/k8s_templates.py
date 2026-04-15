# src/k8s_templates.py

server_template = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-server
  namespace: {{ cfg.k8s.namespace }}
  labels:
    app: vllm-server
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-server
  template:
    metadata:
      labels:
        app: vllm-server
    spec:
      containers:
        - name: vllm-server
          image: vllm/vllm-openai:v0.4.0 
          command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
          args:
            - "--model={{ cfg.model.name }}"
            - "--max-model-len=16384"
            - "--tensor-parallel-size={{ cfg.k8s.server.gpu_count }}"
            - "--download-dir=/models"
          ports:
            - containerPort: 8000
          resources:
            limits:
              nvidia.com/gpu: "{{ cfg.k8s.server.gpu_count }}"
              memory: "{{ cfg.k8s.server.memory_limit }}"
            requests:
              nvidia.com/gpu: "{{ cfg.k8s.server.gpu_count }}"
              memory: "{{ cfg.k8s.server.memory_request }}"
          volumeMounts:
            - name: dshm
              mountPath: /dev/shm
            - name: model-volume
              mountPath: /models
          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: HF_TOKEN
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory
        - name: model-volume
          persistentVolumeClaim:
            claimName: model-volume
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-server
  namespace: {{ cfg.k8s.namespace }}
spec:
  selector:
    app: vllm-server
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
"""

client_template = """apiVersion: batch/v1
kind: Job
metadata:
  name: diagnose-prediction-client
  namespace: {{ cfg.k8s.namespace }}
spec:
  template:
    spec:
      containers:
        - name: inference-client
          image: {{ cfg.k8s.client.image }} 
          command: ["python3", "-u", "src/main.py", "--config", "configs/config.yaml"]
          resources:
            limits:
              memory: "{{ cfg.k8s.client.memory_limit }}"
              cpu: "{{ cfg.k8s.client.cpu_limit }}"
            requests:
              memory: "{{ cfg.k8s.client.memory_request }}"
              cpu: "{{ cfg.k8s.client.cpu_request }}"
      restartPolicy: Never
  backoffLimit: 0
"""
