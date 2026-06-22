from pathlib import Path
import shutil

import config


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TABLE_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls"}
TEXT_EXTENSIONS = {".md", ".txt"}
DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html", ".htm"}
SUPPORTED_UPLOAD_EXTENSIONS = tuple(
    sorted(IMAGE_EXTENSIONS | TABLE_EXTENSIONS | TEXT_EXTENSIONS | DOCLING_EXTENSIONS)
)


def supported_extensions_text():
    return ", ".join(SUPPORTED_UPLOAD_EXTENSIONS)


def markdown_name_for(source_path):
    source_path = Path(source_path)
    suffix = source_path.suffix.lower()
    if suffix in {".pdf", ".md"}:
        return f"{source_path.stem}.md"
    return f"{source_path.stem}_{suffix.lstrip('.')}.md"


class MultimodalDocumentProcessor:
    """Convert non-Markdown inputs into Markdown for the existing RAG pipeline."""

    def __init__(self):
        self._docling_converter = None
        self._ocr_engine = None
        self._captioner = None

    def convert_to_markdown(self, source_path, target_path):
        source_path = Path(source_path)
        target_path = Path(target_path)
        suffix = source_path.suffix.lower()
        target_path.parent.mkdir(parents=True, exist_ok=True)

        if suffix == ".md":
            shutil.copy(source_path, target_path)
            return

        if suffix == ".txt":
            self._write_markdown(target_path, self._plain_text_to_markdown(source_path))
            return

        if suffix in IMAGE_EXTENSIONS:
            self._write_markdown(target_path, self._image_to_markdown(source_path))
            return

        if suffix in TABLE_EXTENSIONS:
            self._write_markdown(target_path, self._table_file_to_markdown(source_path))
            return

        if suffix in DOCLING_EXTENSIONS:
            self._write_markdown(target_path, self._document_to_markdown(source_path, target_path))
            return

        raise ValueError(
            f"Unsupported file type: {suffix}. Supported types: {supported_extensions_text()}"
        )

    def _write_markdown(self, target_path, markdown):
        markdown = markdown.strip()
        if not markdown:
            raise ValueError("Converted Markdown is empty.")
        Path(target_path).write_text(markdown + "\n", encoding="utf-8")

    def _plain_text_to_markdown(self, source_path):
        text = Path(source_path).read_text(encoding="utf-8", errors="ignore").strip()
        return "\n\n".join(
            [
                f"# Text Source: {source_path.name}",
                f"Source file: `{source_path.name}`",
                "Source type: text",
                "## Content",
                text,
            ]
        )

    def _image_to_markdown(self, source_path):
        caption, caption_error = self._safe_caption_image(source_path)
        ocr_text, ocr_error = self._safe_ocr_image(source_path)

        sections = [
            f"# Image Source: {source_path.name}",
            f"Source file: `{source_path.name}`",
            "Source type: image",
            "## Image Caption",
            caption or "No image caption was generated.",
            "## OCR Text",
            ocr_text or "No OCR text was detected.",
        ]

        notes = [note for note in [caption_error, ocr_error] if note]
        if notes:
            sections.extend(["## Ingestion Notes", "\n".join(f"- {note}" for note in notes)])

        return "\n\n".join(sections)

    def _safe_caption_image(self, source_path):
        try:
            return self._caption_image(source_path), None
        except Exception as exc:
            return "", self._optional_dependency_note("image captioning", "Transformers BLIP", exc)

    def _caption_image(self, source_path):
        from PIL import Image
        from transformers import pipeline

        if self._captioner is None:
            self._captioner = pipeline(
                "image-to-text",
                model=getattr(config, "IMAGE_CAPTION_MODEL", "Salesforce/blip-image-captioning-base"),
            )

        image = Image.open(source_path).convert("RGB")
        result = self._captioner(
            image,
            max_new_tokens=getattr(config, "IMAGE_CAPTION_MAX_NEW_TOKENS", 80),
        )
        if not result:
            return ""
        first = result[0]
        return str(first.get("generated_text", first)).strip()

    def _safe_ocr_image(self, source_path):
        try:
            return self._ocr_image(source_path), None
        except Exception as exc:
            return "", self._optional_dependency_note("OCR", "PaddleOCR", exc)

    def _ocr_image(self, source_path):
        from paddleocr import PaddleOCR

        if self._ocr_engine is None:
            try:
                self._ocr_engine = PaddleOCR(
                    lang=getattr(config, "PADDLEOCR_LANG", "ch"),
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                )
            except TypeError:
                self._ocr_engine = PaddleOCR(
                    lang=getattr(config, "PADDLEOCR_LANG", "ch"),
                    use_angle_cls=True,
                )

        if hasattr(self._ocr_engine, "predict"):
            raw_result = self._ocr_engine.predict(str(source_path))
        else:
            raw_result = self._ocr_engine.ocr(str(source_path), cls=True)

        return self._extract_ocr_text(raw_result)

    def _extract_ocr_text(self, raw_result):
        lines = []

        def add_text(value):
            text = str(value).strip()
            if text:
                lines.append(text)

        def walk(item):
            if item is None:
                return

            result_dict = getattr(item, "res", None)
            if isinstance(result_dict, dict):
                walk(result_dict)
                return

            if isinstance(item, dict):
                for key in ("rec_texts", "texts", "text"):
                    value = item.get(key)
                    if isinstance(value, list):
                        for entry in value:
                            add_text(entry)
                    elif isinstance(value, str):
                        add_text(value)
                return

            if isinstance(item, (list, tuple)):
                if (
                    len(item) >= 2
                    and isinstance(item[1], (list, tuple))
                    and item[1]
                    and isinstance(item[1][0], str)
                ):
                    add_text(item[1][0])
                    return

                for entry in item:
                    walk(entry)

        walk(raw_result)
        return "\n".join(dict.fromkeys(lines))

    def _table_file_to_markdown(self, source_path):
        pd = self._import_pandas()
        suffix = source_path.suffix.lower()

        sections = [
            f"# Table Source: {source_path.name}",
            f"Source file: `{source_path.name}`",
            "Source type: table",
        ]

        if suffix in {".xlsx", ".xls"}:
            workbook = pd.ExcelFile(source_path)
            for sheet_name in workbook.sheet_names:
                frame = pd.read_excel(workbook, sheet_name=sheet_name)
                sections.append(
                    self._dataframe_to_markdown_blocks(
                        frame,
                        f"Sheet: {sheet_name}",
                    )
                )
        else:
            sep = "\t" if suffix == ".tsv" else ","
            frame = pd.read_csv(source_path, sep=sep)
            sections.append(self._dataframe_to_markdown_blocks(frame, "Table Data"))

        return "\n\n".join(sections)

    def _document_to_markdown(self, source_path, target_path):
        suffix = source_path.suffix.lower()
        notes = []
        markdown = ""

        try:
            markdown = self._convert_with_docling(source_path).strip()
        except Exception as exc:
            notes.append(self._optional_dependency_note("document parsing", "Docling", exc))

        if not markdown and suffix == ".pdf":
            markdown = self._pdf_to_markdown_fallback(source_path, target_path).strip()

        if not markdown:
            raise RuntimeError(
                f"Unable to parse {source_path.name}. Install Docling or use a supported PDF/text/table/image format."
            )

        if suffix == ".pdf":
            table_markdown, table_note = self._safe_extract_pdf_tables(source_path)
            if table_markdown:
                markdown += "\n\n# Extracted PDF Tables\n\n" + table_markdown
            if table_note:
                notes.append(table_note)

        if notes:
            markdown += "\n\n# Ingestion Notes\n\n" + "\n".join(f"- {note}" for note in notes)

        return markdown

    def _convert_with_docling(self, source_path):
        from docling.document_converter import DocumentConverter

        if self._docling_converter is None:
            self._docling_converter = DocumentConverter()

        result = self._docling_converter.convert(str(source_path))
        document = getattr(result, "document", None)
        if document is None or not hasattr(document, "export_to_markdown"):
            raise RuntimeError("Docling did not return an exportable document.")
        return document.export_to_markdown()

    def _pdf_to_markdown_fallback(self, source_path, target_path):
        from utils import pdf_to_markdown

        pdf_to_markdown(str(source_path), str(Path(target_path).parent))
        generated_path = Path(target_path).parent / f"{Path(source_path).stem}.md"
        if generated_path.exists():
            markdown = generated_path.read_text(encoding="utf-8", errors="ignore")
            if generated_path != Path(target_path):
                generated_path.unlink()
            return markdown
        return ""

    def _safe_extract_pdf_tables(self, source_path):
        try:
            return self._extract_pdf_tables(source_path), None
        except Exception as exc:
            return "", self._optional_dependency_note("PDF table extraction", "Camelot", exc)

    def _extract_pdf_tables(self, source_path):
        import camelot

        read_attempts = (
            {"flavor": "auto"},
            {"flavor": "lattice"},
            {"flavor": "stream"},
        )

        last_error = None
        tables = None
        for options in read_attempts:
            try:
                tables = camelot.read_pdf(str(source_path), pages="all", **options)
                if tables and len(tables) > 0:
                    break
            except Exception as exc:
                last_error = exc

        if not tables or len(tables) == 0:
            if last_error:
                raise last_error
            return ""

        sections = []
        for index, table in enumerate(tables, start=1):
            frame = getattr(table, "df", None)
            if frame is None:
                continue
            sections.append(self._dataframe_to_markdown_blocks(frame, f"PDF Table {index}"))
        return "\n\n".join(sections)

    def _import_pandas(self):
        try:
            import pandas as pd

            return pd
        except Exception as exc:
            raise RuntimeError("pandas is required for CSV/Excel table ingestion.") from exc

    def _dataframe_to_markdown_blocks(self, frame, title):
        max_rows = getattr(config, "TABLE_ROWS_PER_MARKDOWN_BLOCK", 200)
        total_rows = len(frame)

        sections = [
            f"## {title}",
            f"Rows: {total_rows}",
            f"Columns: {', '.join(str(column) for column in frame.columns)}",
        ]

        if total_rows == 0:
            sections.append("Empty table.")
            return "\n\n".join(sections)

        for start in range(0, total_rows, max_rows):
            end = min(start + max_rows, total_rows)
            block = frame.iloc[start:end].copy().fillna("")
            sections.append(f"### Rows {start + 1}-{end}")
            sections.append(self._frame_to_markdown(block))

        return "\n\n".join(sections)

    def _frame_to_markdown(self, frame):
        headers = [self._escape_table_cell(column) for column in frame.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for _, row in frame.iterrows():
            values = [self._escape_table_cell(value) for value in row.tolist()]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    def _escape_table_cell(self, value):
        return str(value).replace("\n", " ").replace("|", "\\|").strip()

    def _optional_dependency_note(self, capability, project_name, exc):
        return (
            f"{capability} via {project_name} was unavailable "
            f"({type(exc).__name__}: {exc})."
        )
