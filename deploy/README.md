# Qwen API Server Deployment

The default deployment path uses DashScope's OpenAI-compatible Qwen API. It does not require Ollama or local LLM weights.

## One-Click Deployment

On a fresh Linux server:

```bash
git clone https://github.com/GuMiShDo666/RumerDetection-rag.git
cd RumerDetection-rag
sudo DASHSCOPE_API_KEY="your_dashscope_api_key" \
  PUBLIC_HOST="your_server_public_ip" \
  BASIC_AUTH_PASSWORD="Dongzexuan" \
  bash deploy/one_click_qwen_server.sh
```

After installation, open:

```text
https://your_server_public_ip/
```

Default login:

- Username: `admin`
- Password: `Dongzexuan`

The HTTPS certificate is self-signed, so the browser may show a certificate warning.

## What The Script Does

- Installs system packages: Git, Python, Nginx, OpenSSL, curl, and Basic Auth tools.
- Clones the GitHub repository into `/opt/rumordetection-rag`.
- Creates a Python virtual environment.
- Installs Python dependencies with the Tsinghua PyPI mirror by default.
- Writes `project/.env` with Qwen API settings.
- Builds the generated CSV and local Qdrant retrieval index.
- Creates a `rumordetection-rag` systemd service.
- Exposes the app through Nginx HTTPS with Basic Auth.

## Useful Options

```bash
sudo DASHSCOPE_API_KEY="..." \
  PUBLIC_HOST="114.215.253.51" \
  BASIC_AUTH_USER="admin" \
  BASIC_AUTH_PASSWORD="Dongzexuan" \
  QWEN_MODEL="qwen-plus" \
  QWEN_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1" \
  bash deploy/one_click_qwen_server.sh
```

For international DashScope accounts, use:

```bash
QWEN_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
```

## Diagnostics

```bash
systemctl status rumordetection-rag --no-pager
journalctl -u rumordetection-rag -f
systemctl status nginx --no-pager
```

The app listens on `127.0.0.1:7860`; Nginx exposes it through HTTPS.
