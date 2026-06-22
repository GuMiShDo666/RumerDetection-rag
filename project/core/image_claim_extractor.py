from pathlib import Path

import config


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def is_supported_image(path):
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


class ImageClaimExtractor:
    """Extract text and visual context from uploaded images for rumor detection."""

    def __init__(self):
        self._ocr_engine = None
        self._captioner = None

    def extract(self, image_path):
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not is_supported_image(image_path):
            supported = ", ".join(sorted(IMAGE_EXTENSIONS))
            raise ValueError(f"Unsupported image type: {image_path.suffix}. Supported: {supported}")

        caption, caption_error = self._safe_caption_image(image_path)
        ocr_text, ocr_error = self._safe_ocr_image(image_path)

        if not caption and not ocr_text:
            notes = "; ".join(note for note in (caption_error, ocr_error) if note)
            if notes:
                raise RuntimeError(f"Unable to extract usable image content. {notes}")
            raise RuntimeError("Unable to extract usable image content from the uploaded image.")

        return {
            "path": str(image_path),
            "name": image_path.name,
            "caption": caption,
            "ocr_text": ocr_text,
            "caption_error": caption_error,
            "ocr_error": ocr_error,
        }

    def to_prompt_section(self, result, index=1):
        lines = [f"图片 {index}: {result['name']}"]
        if result.get("ocr_text"):
            lines.extend(["OCR 识别文本:", result["ocr_text"]])
        if result.get("caption"):
            lines.extend(["BLIP 图片说明:", result["caption"]])
        return "\n".join(lines)

    def to_summary_markdown(self, result, index=1):
        lines = [f"**Image {index}: `{result['name']}`**"]
        if result.get("ocr_text"):
            lines.extend(["", "**OCR Text**", result["ocr_text"]])
        if result.get("caption"):
            lines.extend(["", "**BLIP Caption**", result["caption"]])
        notes = [note for note in (result.get("ocr_error"), result.get("caption_error")) if note]
        if notes:
            lines.extend(["", "**Extraction Notes**"])
            lines.extend(f"- {note}" for note in notes)
        return "\n".join(lines)

    def _safe_caption_image(self, image_path):
        try:
            return self._caption_image(image_path), None
        except Exception as exc:
            return "", self._optional_dependency_note("BLIP image captioning", "Transformers BLIP", exc)

    def _caption_image(self, image_path):
        from PIL import Image
        from transformers import pipeline

        if self._captioner is None:
            self._captioner = pipeline(
                "image-to-text",
                model=config.IMAGE_CAPTION_MODEL,
            )

        image = Image.open(image_path).convert("RGB")
        result = self._captioner(
            image,
            max_new_tokens=config.IMAGE_CAPTION_MAX_NEW_TOKENS,
        )
        if not result:
            return ""
        first = result[0]
        return str(first.get("generated_text", first)).strip()

    def _safe_ocr_image(self, image_path):
        try:
            return self._ocr_image(image_path), None
        except Exception as exc:
            return "", self._optional_dependency_note("OCR", "PaddleOCR", exc)

    def _ocr_image(self, image_path):
        from paddleocr import PaddleOCR

        if self._ocr_engine is None:
            try:
                self._ocr_engine = PaddleOCR(
                    lang=config.PADDLEOCR_LANG,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=True,
                )
            except TypeError:
                self._ocr_engine = PaddleOCR(
                    lang=config.PADDLEOCR_LANG,
                    use_angle_cls=True,
                )

        if hasattr(self._ocr_engine, "predict"):
            raw_result = self._ocr_engine.predict(str(image_path))
        else:
            raw_result = self._ocr_engine.ocr(str(image_path), cls=True)

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

    def _optional_dependency_note(self, task, backend, exc):
        return f"{task} skipped because {backend} failed: {type(exc).__name__}: {exc}"
