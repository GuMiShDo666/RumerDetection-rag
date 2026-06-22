# RumerDetection-rag Project Notes

This folder contains the runnable Agentic RAG application for Chinese rumor detection.

The application builds a local retrieval index from `data/rumor_database.csv`. Each case is stored as a natural-language judgment, for example:

```csv
id,statement,label,label_name
RD-00001,喝汤比吃菜更有营养是谣言,1,谣言
RD-02687,年轻人同样可能感染并传播病毒不是谣言,0,非谣言
```

`project/rumor_database.py` prepares the case database and creates `markdown_docs/rumor_database.md` for indexing. `DocumentManager.build_rumor_database()` clears the local Qdrant collection, chunks the generated Markdown, stores parent chunks, and indexes child chunks.

Run:

```bash
python project/app.py
```

Then click **Build / Rebuild Rumor RAG Database** in the Gradio UI before asking questions.
