<h1 align="center">RumerDetection-rag</h1>

<p align="center">
  <strong>Agentic RAG system for Chinese text and image rumor detection</strong>
  <br />
  <em>OCR + BLIP image parsing · Reference article knowledge base · Qdrant hybrid retrieval · Qwen API · LangGraph agent</em>
</p>

<p align="center">
  <a href="README.md">English</a> ·
  <a href="README.zh-CN.md">简体中文</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Agent-LangGraph-0F766E?style=flat-square" alt="LangGraph" />
  <img src="https://img.shields.io/badge/Vector_DB-Qdrant-DC244C?style=flat-square" alt="Qdrant" />
  <img src="https://img.shields.io/badge/Task-Rumor_Detection-7C3AED?style=flat-square" alt="Rumor Detection" />
</p>

---

RumerDetection-rag is a retrieval-augmented rumor detection application for Chinese text and image claims. It stores fact-checking and health-science reference articles as a searchable knowledge base, retrieves relevant evidence for a new input, and uses an agent workflow to produce a grounded verdict.

The system is designed for evidence-first answers: if relevant article evidence is missing or weak, the assistant returns `证据不足` instead of forcing a binary classification.

## Interface Preview

### System Dashboard

![System dashboard](docs/screenshots/system-overview.png)

### Detection Result After Text Input

![Detection result after text input](docs/screenshots/detection-result.png)

## Core Features

| Feature | Description |
| --- | --- |
| Text and image detection | Accepts direct text input or uploaded images in the chat UI |
| OCR + BLIP parsing | Uses PaddleOCR for image text extraction and BLIP for image captioning |
| Reference knowledge base | Uses 17 CSV sources under `data/reference_data` as the RAG corpus |
| Article normalization | Builds a generated `data/rumor_database.csv` with `id`, `source`, `title`, `url`, `date`, and `text` |
| Hybrid retrieval | Uses Qdrant dense + sparse retrieval to find relevant reference articles |
| Agentic workflow | Uses LangGraph for query rewriting, retrieval tools, context compression, and synthesis |
| Qwen API generation | Uses DashScope's OpenAI-compatible Qwen API as the default LLM provider |
| Evidence-grounded verdict | Returns `谣言`, `非谣言`, or `证据不足` with supporting retrieved articles |

## Knowledge Base

The source corpus is stored in `data/reference_data/`. It contains 17 CSV files from fact-checking and health-science sources. The build step filters empty articles, removes duplicates, and generates `data/rumor_database.csv` locally for indexing.

Generated database format:

```csv
id,source,title,url,date,text
REF-00001,中华医学健康科普知识库,老年人总穿拖鞋易疲劳，拖鞋挑选有讲究,https://...,2022-05-18,...
```

Current indexed corpus:

| Item | Count |
| --- | ---: |
| Source CSV files | 17 |
| Indexed articles | 17183 |
| Empty-text rows skipped | 691 |
| Duplicate rows skipped | 298 |

## Architecture

```mermaid
flowchart LR
    A["reference_data CSV files"] --> B["Normalized article database"]
    B --> C["Markdown reference knowledge base"]
    C --> D["Parent-child chunking"]
    D --> E["Qdrant hybrid index"]
    F["User claim"] --> G["LangGraph query rewrite"]
    K["Uploaded image"] --> L["PaddleOCR + BLIP"]
    L --> G
    G --> H["search_child_chunks"]
    H --> E
    H --> I["retrieve_parent_chunks"]
    I --> J["Verdict synthesis"]
```

## Quick Start

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r requirements.txt
```

The CPU PaddlePaddle command above enables PaddleOCR on common local environments. For GPU or platform-specific wheels, follow the [PaddleOCR installation guide](https://paddlepaddle.github.io/PaddleOCR/v3.1.1/en/quick_start.html).

### 2. Configure Qwen API

Copy the environment template and set your DashScope API key:

```bash
cp project/.env.example project/.env
export DASHSCOPE_API_KEY="your_dashscope_api_key"
# or write DASHSCOPE_API_KEY=... into project/.env
```

By default, the app uses:

```text
QWEN_MODEL=qwen-plus
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 3. Launch the App

```bash
python project/app.py
```

Open the local web URL, then enter a claim or upload an image on the detection page.

## One-Click Server Deployment

run:

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

The script installs dependencies, creates `project/.env`, builds the local Qdrant index, starts the FastAPI service with systemd, and exposes the Chinese HTML UI through Nginx HTTPS with Basic Auth.

## Evaluation

Run a lightweight QA evaluation:

```bash
python project/evaluation.py \
  --qa project/evaluation_sample.json \
  --output rag_evaluation_results.csv
```

The evaluator rebuilds the RAG database, runs the LangGraph agent, and exports the predicted verdict, final answer, deterministic sources, retrieved context count, reference-overlap proxy score, and expected-source hit rate.

## Project Structure

```text
data/
  reference_data/
    *.csv
  rumor_database.csv  # generated locally
project/
  app.py
  config.py
  rumor_database.py
  document_chunker.py
  core/
    document_manager.py
    image_claim_extractor.py
    rag_system.py
    chat_interface.py
  db/
    vector_db_manager.py
    parent_store_manager.py
  rag_agent/
    graph.py
    nodes.py
    tools.py
    prompts.py
  web/
    index.html
    styles.css
    app.js
  web_app.py
deploy/
  one_click_qwen_server.sh
```

## Validation

```bash
python3 -m compileall -q project
python3 project/evaluation.py --help
python3 -m json.tool project/evaluation_sample.json
```

## Notes

- The assistant uses retrieved articles as evidence rather than relying on a trained classifier alone.
- The reference article corpus is the primary retrieval source for text and image rumor detection.
- Image inputs are parsed before retrieval. OCR text is the primary signal, and BLIP captions are secondary context.
- The first image request may download OCR and BLIP model weights.
- The response starts with `判定：谣言`, `判定：非谣言`, or `判定：证据不足`.
- The system is an evidence-grounded aid, not a general medical or legal authority.

## License

See [LICENSE](LICENSE).
