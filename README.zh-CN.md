<h1 align="center">RumerDetection-rag</h1>

<p align="center">
  <strong>支持文本和图片输入的中文谣言检测 Agentic RAG 系统</strong>
  <br />
  <em>OCR + BLIP 图片解析 · 自然语言案例库 · Qdrant 混合检索 · LangGraph Agent</em>
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

RumerDetection-rag 是一个面向中文文本和图片输入的检索增强谣言检测应用。系统将已标注的谣言案例保存为可检索的自然语言判断，针对新的用户输入检索相似案例，再通过 Agent 工作流生成有依据的判定。

系统优先使用检索证据回答问题；当相似案例不足或证据较弱时，会返回 `证据不足`，而不是强行二分类。

## 核心能力

| 能力 | 说明 |
| --- | --- |
| 文本和图片检测 | 支持在 Chat UI 中直接输入文本，或上传包含待检测信息的图片 |
| OCR + BLIP 解析 | 使用 PaddleOCR 提取图片文字，使用 BLIP 生成图片说明 |
| 自然语言案例库 | 每条案例以 `喝汤比吃菜更有营养是谣言` 这样的可读格式保存 |
| 标签映射 | 保留明确标签：`1 = 谣言`，`0 = 非谣言` |
| 混合检索 | 使用 Qdrant dense + sparse 检索相似已标注文本 |
| Agentic workflow | 使用 LangGraph 负责问题改写、工具检索、上下文压缩和最终生成 |
| 证据化判定 | 输出 `谣言`、`非谣言` 或 `证据不足`，并附相似案例依据 |

## 知识库

主检索数据库保存在 `data/rumor_database.csv`。

示例格式：

```csv
id,statement,label,label_name
RD-00001,喝汤比吃菜更有营养是谣言,1,谣言
RD-02687,年轻人同样可能感染并传播病毒不是谣言,0,非谣言
```

当前数据库统计：

| Label | 含义 | Rows |
| --- | --- | ---: |
| 1 | 谣言 | 1844 |
| 0 | 非谣言 | 1513 |
| total | - | 3357 |

## 架构

```mermaid
flowchart LR
    A["标注文本知识库"] --> B["自然语言案例记录"]
    B --> C["Markdown RAG 知识库"]
    C --> D["Parent-child chunking"]
    D --> E["Qdrant 混合索引"]
    F["用户输入文本"] --> G["LangGraph query rewrite"]
    K["上传图片"] --> L["PaddleOCR + BLIP"]
    L --> G
    G --> H["search_child_chunks"]
    H --> E
    H --> I["retrieve_parent_chunks"]
    I --> J["最终判定生成"]
```

## 快速开始

### 1. 安装依赖

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
python -m pip install -r requirements.txt
```

上面的 CPU PaddlePaddle 命令用于支持 PaddleOCR。GPU 或特殊平台安装方式请参考 [PaddleOCR installation guide](https://paddlepaddle.github.io/PaddleOCR/v3.1.1/en/quick_start.html)。

### 2. 准备 Ollama

从 [ollama.com](https://ollama.com) 安装 Ollama，然后拉取默认聊天模型：

```bash
ollama pull granite4.1:8b
```

默认 embedding 模型是 `Qwen/Qwen3-Embedding-0.6B`。

### 3. 启动应用

```bash
python project/app.py
```

打开 Gradio 地址，点击 **Build / Rebuild Rumor RAG Database**，然后在 Chat 页输入待检测文本或上传图片。

## 评测

运行轻量 QA 评测：

```bash
python project/evaluation.py \
  --qa project/evaluation_sample.json \
  --output rag_evaluation_results.csv
```

评测脚本会重建 RAG 数据库、运行 LangGraph Agent，并导出 predicted verdict、final answer、deterministic sources、retrieved context count、reference-overlap proxy score 和 expected-source hit rate。

## 项目结构

```text
data/
  rumor_database.csv
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
  ui/
    gradio_app.py
```

## 验证

```bash
python3 -m compileall -q project
python3 project/evaluation.py --help
python3 -m json.tool project/evaluation_sample.json
```

## 说明

- 系统使用检索到的案例作为证据，而不是只依赖训练好的分类器。
- 图片输入会先被解析后再进入检索流程，其中 OCR 文本是主要信号，BLIP 图片说明作为补充上下文。
- 第一次使用图片检测时，可能会下载 OCR 和 BLIP 模型权重。
- 回答以 `判定：谣言`、`判定：非谣言` 或 `判定：证据不足` 开头。
- 该系统是基于证据的辅助判断工具，不是通用医学或法律权威。

## License

详见 [LICENSE](LICENSE)。
