#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/GuMiShDo666/RumerDetection-rag.git}"
APP_DIR="${APP_DIR:-/opt/rumordetection-rag}"
APP_PORT="${APP_PORT:-7860}"
PUBLIC_HOST="${PUBLIC_HOST:-}"
BASIC_AUTH_USER="${BASIC_AUTH_USER:-admin}"
BASIC_AUTH_PASSWORD="${BASIC_AUTH_PASSWORD:-Dongzexuan}"
SERVICE_NAME="${SERVICE_NAME:-rumordetection-rag}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
PIP_TRUSTED_HOST="${PIP_TRUSTED_HOST:-pypi.tuna.tsinghua.edu.cn}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
QWEN_MODEL="${QWEN_MODEL:-qwen-plus}"
QWEN_BASE_URL="${QWEN_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
BUILD_RAG_INDEX="${BUILD_RAG_INDEX:-1}"
CHILD_CHUNK_SIZE="${CHILD_CHUNK_SIZE:-3500}"
CHILD_CHUNK_OVERLAP="${CHILD_CHUNK_OVERLAP:-150}"
MIN_PARENT_SIZE="${MIN_PARENT_SIZE:-2500}"
MAX_PARENT_SIZE="${MAX_PARENT_SIZE:-4500}"
INDEX_BATCH_SIZE="${INDEX_BATCH_SIZE:-256}"

if [[ "$(id -u)" != "0" ]]; then
  echo "Run this script as root, for example: sudo DASHSCOPE_API_KEY=... bash deploy/one_click_qwen_server.sh" >&2
  exit 1
fi

if [[ -z "${DASHSCOPE_API_KEY:-}" && -z "${QWEN_API_KEY:-}" ]]; then
  read -r -s -p "DashScope/Qwen API key: " DASHSCOPE_API_KEY
  echo
fi

if [[ -z "${PUBLIC_HOST}" ]]; then
  PUBLIC_HOST="$(curl -fsS --max-time 3 https://api.ipify.org 2>/dev/null || hostname -I | awk '{print $1}')"
fi

install_packages() {
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y \
      git nginx apache2-utils openssl curl ca-certificates \
      python3 python3-venv python3-pip
    if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
      apt-get install -y python3.11 python3.11-venv || true
    fi
  elif command -v dnf >/dev/null 2>&1; then
    dnf install -y git nginx httpd-tools openssl curl ca-certificates python3 python3-pip
  elif command -v yum >/dev/null 2>&1; then
    yum install -y git nginx httpd-tools openssl curl ca-certificates python3 python3-pip
  else
    echo "Unsupported package manager. Install git, nginx, htpasswd, openssl, curl, and Python manually." >&2
    exit 1
  fi
}

resolve_python() {
  if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    command -v "${PYTHON_BIN}"
  elif command -v python3.11 >/dev/null 2>&1; then
    command -v python3.11
  else
    command -v python3
  fi
}

echo "[1/7] Installing system packages"
install_packages

echo "[2/7] Cloning application into ${APP_DIR}"
rm -rf "${APP_DIR}"
git clone --depth 1 "${REPO_URL}" "${APP_DIR}"

PYTHON_PATH="$(resolve_python)"
echo "[3/7] Creating Python environment with ${PYTHON_PATH}"
"${PYTHON_PATH}" -m venv "${APP_DIR}/.venv"
"${APP_DIR}/.venv/bin/python" -m pip install --no-cache-dir --upgrade pip setuptools wheel \
  -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}"
"${APP_DIR}/.venv/bin/python" -m pip install --no-cache-dir paddlepaddle \
  -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ || \
  "${APP_DIR}/.venv/bin/python" -m pip install --no-cache-dir paddlepaddle \
    -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}"
"${APP_DIR}/.venv/bin/python" -m pip install --no-cache-dir -r "${APP_DIR}/requirements.txt" \
  -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}"

echo "[4/7] Writing runtime environment"
cat > "${APP_DIR}/project/.env" <<EOF
HF_HOME=/root/.cache/huggingface
HF_ENDPOINT=${HF_ENDPOINT}
TOKENIZERS_PARALLELISM=false
RAG_SERVER_NAME=127.0.0.1
RAG_SERVER_PORT=${APP_PORT}
RAG_AUTH_USERNAME=
RAG_AUTH_PASSWORD=
QWEN_MODEL=${QWEN_MODEL}
QWEN_BASE_URL=${QWEN_BASE_URL}
DASHSCOPE_API_KEY=${DASHSCOPE_API_KEY:-${QWEN_API_KEY:-}}
RAG_FORCE_RETRIEVAL_FALLBACK=false
DENSE_MODEL=BAAI/bge-small-zh-v1.5
SPARSE_MODEL=Qdrant/bm25
CHILD_CHUNK_SIZE=${CHILD_CHUNK_SIZE}
CHILD_CHUNK_OVERLAP=${CHILD_CHUNK_OVERLAP}
MIN_PARENT_SIZE=${MIN_PARENT_SIZE}
MAX_PARENT_SIZE=${MAX_PARENT_SIZE}
INDEX_BATCH_SIZE=${INDEX_BATCH_SIZE}
EOF
chmod 600 "${APP_DIR}/project/.env"

if [[ "${BUILD_RAG_INDEX}" == "1" ]]; then
  echo "[5/7] Building RAG index"
  cd "${APP_DIR}"
  set -a
  # shellcheck disable=SC1091
  . "${APP_DIR}/project/.env"
  set +a
  PYTHONPATH=project "${APP_DIR}/.venv/bin/python" - <<'PY'
from core.document_manager import DocumentManager
from core.rag_system import RAGSystem

rag = RAGSystem()
rag.initialize()
result, parent_count, child_count = DocumentManager(rag).build_rumor_database()
print(f"Indexed {result['rows']} articles into {parent_count} parent chunks and {child_count} child chunks")
PY
fi

echo "[6/7] Creating systemd service"
cat > "/etc/systemd/system/${SERVICE_NAME}.service" <<EOF
[Unit]
Description=RumorDetection Agentic RAG
After=network.target

[Service]
Type=simple
WorkingDirectory=${APP_DIR}
EnvironmentFile=${APP_DIR}/project/.env
ExecStart=${APP_DIR}/.venv/bin/python ${APP_DIR}/project/app.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now "${SERVICE_NAME}"

echo "[7/7] Configuring Nginx HTTPS reverse proxy"
mkdir -p /etc/nginx/ssl
openssl req -x509 -nodes -days 825 -newkey rsa:2048 \
  -keyout /etc/nginx/ssl/rumordetection-rag.key \
  -out /etc/nginx/ssl/rumordetection-rag.crt \
  -subj "/CN=${PUBLIC_HOST}"

htpasswd -bc /etc/nginx/.rumordetection-rag.htpasswd "${BASIC_AUTH_USER}" "${BASIC_AUTH_PASSWORD}"

cat > /etc/nginx/conf.d/rumordetection-rag.conf <<EOF
server {
    listen 80 default_server;
    server_name _;
    return 301 https://\$host\$request_uri;
}

server {
    listen 443 ssl default_server;
    server_name _;

    ssl_certificate /etc/nginx/ssl/rumordetection-rag.crt;
    ssl_certificate_key /etc/nginx/ssl/rumordetection-rag.key;

    auth_basic "RumorDetection RAG";
    auth_basic_user_file /etc/nginx/.rumordetection-rag.htpasswd;

    client_max_body_size 50m;

    location / {
        proxy_pass http://127.0.0.1:${APP_PORT};
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_buffering off;
        proxy_read_timeout 3600;
    }
}
EOF

nginx -t
systemctl enable --now nginx
systemctl reload nginx

echo
echo "Deployment complete."
echo "URL: https://${PUBLIC_HOST}/"
echo "Username: ${BASIC_AUTH_USER}"
echo "Password: ${BASIC_AUTH_PASSWORD}"
echo
echo "Useful checks:"
echo "  systemctl status ${SERVICE_NAME} --no-pager"
echo "  journalctl -u ${SERVICE_NAME} -f"
