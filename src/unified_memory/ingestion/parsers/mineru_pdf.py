"""
MinerU-backed PDF parser.

Uses MinerU's pipeline backend for layout-aware PDF extraction with:
- Layout detection (DocLayout-YOLO)
- Formula detection + recognition (MFD + MFR)
- Table structure recognition (SlanetPlus / UNet)
- OCR (PytorchPaddleOCR) — auto, txt, or ocr mode
- Per-page images, figure crops, and table crops as PNG bytes

Requires the ``mineru`` optional dependency group::

    pip install "unified-memory-system[mineru]"
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import tempfile
from collections import defaultdict
from io import BytesIO
from typing import Any, BinaryIO, Dict, List, Optional

from unified_memory.core.types import PageContent, SourceReference, SourceType
from unified_memory.ingestion.parsers.base import DocumentParser, ParsedDocument

logger = logging.getLogger(__name__)

try:
    from PIL import Image

    from mineru.backend.pipeline.pipeline_analyze import (
        doc_analyze as pipeline_doc_analyze,
    )
    from mineru.backend.pipeline.model_json_to_middle_json import (
        result_to_middle_json as pipeline_result_to_middle_json,
    )
    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import (
        union_make as pipeline_union_make,
    )
    from mineru.data.data_reader_writer import FileBasedDataWriter
    from mineru.utils.enum_class import BlockType, MakeMode  # noqa: F401

    _MINERU_AVAILABLE = True
except ImportError:
    _MINERU_AVAILABLE = False


def is_mineru_available() -> bool:
    """Check whether the MinerU runtime dependencies are installed."""
    return _MINERU_AVAILABLE


class MinerUPDFParser(DocumentParser):
    """
    PDF parser backed by MinerU's pipeline backend.

    Features
    --------
    - Layout detection (DocLayout-YOLO)
    - Formula detection + recognition (MFD + MFR)
    - Table structure recognition (SlanetPlus / UNet)
    - OCR (PytorchPaddleOCR) — auto, txt, or ocr mode
    - Per-page PIL images exposed as PNG bytes in ``PageContent.full_page_image``
    - Figures and tables include cropped image bytes
    """

    def __init__(self) -> None:
        if not _MINERU_AVAILABLE:
            raise ImportError(
                "MinerU is not installed. Install it with: "
                'pip install "unified-memory-system[mineru]"'
            )

    @property
    def supported_extensions(self) -> List[str]:
        return [".pdf"]

    @property
    def supported_mime_types(self) -> List[str]:
        return ["application/pdf"]

    @property
    def default_source_type(self) -> SourceType:
        return SourceType.FULL_PAGE

    # ------------------------------------------------------------------
    # Core parse
    # ------------------------------------------------------------------

    async def parse(
        self,
        source: BinaryIO,
        source_ref: SourceReference,
        document_id: str,
        **options: Any,
    ) -> ParsedDocument:
        """
        Parse a PDF using MinerU's pipeline backend.

        Accepted **options
        ------------------
        lang : str
            Language hint passed to OCR / table models (e.g. ``"en"``, ``"ch"``).
            Default: ``"en"``.
        parse_method : str
            One of ``"auto"`` | ``"txt"`` | ``"ocr"``. Default: ``"auto"``.
        formula_enable : bool
            Enable formula detection + recognition. Default: ``True``.
        table_enable : bool
            Enable table structure recognition. Default: ``True``.
        """
        pdf_bytes = source.read()

        lang = options.get("lang", "en")
        parse_method = options.get("parse_method", "auto")
        formula_enable = options.get("formula_enable", True)
        table_enable = options.get("table_enable", True)

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self._run_pipeline,
            pdf_bytes,
            document_id,
            source_ref,
            lang,
            parse_method,
            formula_enable,
            table_enable,
        )

    # ------------------------------------------------------------------
    # Synchronous pipeline execution (runs in thread pool)
    # ------------------------------------------------------------------

    def _run_pipeline(
        self,
        pdf_bytes: bytes,
        document_id: str,
        source_ref: SourceReference,
        lang: str,
        parse_method: str,
        formula_enable: bool,
        table_enable: bool,
    ) -> ParsedDocument:
        parse_errors: List[str] = []

        # Stage 1: model inference
        try:
            (
                infer_results,
                all_image_lists,
                all_pdf_docs,
                _lang_list,
                ocr_enabled_list,
            ) = pipeline_doc_analyze(
                [pdf_bytes],
                [lang],
                parse_method=parse_method,
                formula_enable=formula_enable,
                table_enable=table_enable,
            )
        except Exception as exc:
            return ParsedDocument(
                document_id=document_id,
                source=source_ref,
                parse_errors=[f"pipeline_doc_analyze failed: {exc}"],
            )

        model_list = infer_results[0]
        images_list = all_image_lists[0]
        pdf_doc = all_pdf_docs[0]
        ocr_enable = ocr_enabled_list[0]

        # Stage 2: structured middle JSON (uses a temp dir for cropped images)
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = os.path.join(temp_dir, "images")
            os.makedirs(image_dir, exist_ok=True)
            image_writer = FileBasedDataWriter(image_dir)

            try:
                middle_json = pipeline_result_to_middle_json(
                    copy.deepcopy(model_list),
                    images_list,
                    pdf_doc,
                    image_writer,
                    lang,
                    ocr_enable,
                    formula_enable,
                )
            except Exception as exc:
                parse_errors.append(f"result_to_middle_json failed: {exc}")
                return ParsedDocument(
                    document_id=document_id,
                    source=source_ref,
                    parse_errors=parse_errors,
                )

            pdf_info = middle_json["pdf_info"]

            # Stage 3: full-document markdown
            try:
                full_text = pipeline_union_make(pdf_info, MakeMode.MM_MD, "")
            except Exception as exc:
                full_text = ""
                parse_errors.append(f"union_make(MM_MD) failed: {exc}")

            # Stage 4: structured content list for per-page extraction
            try:
                content_list = pipeline_union_make(
                    pdf_info, MakeMode.CONTENT_LIST, "images"
                )
            except Exception as exc:
                content_list = []
                parse_errors.append(f"union_make(CONTENT_LIST) failed: {exc}")

            # Stage 5: build per-page PageContent objects
            pages = self._build_pages(
                pdf_info=pdf_info,
                content_list=content_list,
                images_list=images_list,
                document_id=document_id,
            )

        # Stage 6: extract document title (first heading block)
        title: Optional[str] = None
        for item in content_list:
            if item.get("type") == "text" and item.get("text_level", 0) == 1:
                title = item["text"].strip()
                break

        return ParsedDocument(
            document_id=document_id,
            source=source_ref,
            title=title,
            pages=pages,
            full_text=full_text,
            metadata={
                "backend": middle_json.get("_backend", "pipeline"),
                "mineru_version": middle_json.get("_version_name", ""),
                "lang": lang,
                "parse_method": parse_method,
                "formula_enable": formula_enable,
                "table_enable": table_enable,
            },
            parse_errors=parse_errors,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_pages(
        self,
        pdf_info: List[Dict[str, Any]],
        content_list: List[Dict[str, Any]],
        images_list: List[List[Dict[str, Any]]],
        document_id: str,
    ) -> List[PageContent]:
        """Build ``PageContent`` objects with text blocks, figures, and tables."""
        page_buckets: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for item in content_list:
            page_buckets[item["page_idx"]].append(item)

        pages: List[PageContent] = []

        for page_info in pdf_info:
            page_idx = page_info["page_idx"]
            items = page_buckets.get(page_idx, [])

            full_page_image: Optional[bytes] = None
            if page_idx < len(images_list):
                img_dict = images_list[page_idx]
                pil_img = img_dict.get("img_pil")
                if pil_img:
                    buf = BytesIO()
                    pil_img.save(buf, format="PNG")
                    full_page_image = buf.getvalue()

            text_blocks: List[Dict[str, Any]] = []
            figures: List[Dict[str, Any]] = []
            tables: List[Dict[str, Any]] = []
            page_texts: List[str] = []

            for item in items:
                item_type = item.get("type")
                if item_type == "text":
                    text_blocks.append(item)
                    page_texts.append(item.get("text", ""))
                elif item_type == "image":
                    fig = self._create_figure_object(item, full_page_image)
                    if fig:
                        figures.append(fig)
                        page_texts.extend(item.get("image_caption", []))
                elif item_type == "table":
                    tbl = self._create_table_object(item, full_page_image)
                    if tbl:
                        tables.append(tbl)
                        page_texts.extend(item.get("table_caption", []))
                elif item_type == "equation":
                    text_blocks.append(item)
                    page_texts.append(item.get("text", ""))

            pages.append(
                PageContent(
                    page_number=page_idx + 1,
                    document_id=document_id,
                    text_blocks=text_blocks,
                    figures=figures,
                    tables=tables,
                    full_page_image=full_page_image,
                    full_text="\n".join(filter(None, page_texts)),
                )
            )

        return pages

    def _create_figure_object(
        self,
        item: Dict[str, Any],
        full_page_image: Optional[bytes],
    ) -> Optional[Dict[str, Any]]:
        """Create a figure dict with cropped image bytes from the full page."""
        image_bytes: Optional[bytes] = None
        if full_page_image and item.get("bbox"):
            image_bytes = self._crop_from_full_page(full_page_image, item["bbox"])

        return {
            "type": "image",
            "image_bytes": image_bytes,
            "image_path": item.get("img_path", ""),
            "bbox": item.get("bbox", []),
            "page_idx": item.get("page_idx", 0),
            "captions": item.get("image_caption", []),
            "footnotes": item.get("image_footnote", []),
        }

    def _create_table_object(
        self,
        item: Dict[str, Any],
        full_page_image: Optional[bytes],
    ) -> Optional[Dict[str, Any]]:
        """Create a table dict with cropped image bytes from the full page."""
        image_bytes: Optional[bytes] = None
        if full_page_image and item.get("bbox"):
            image_bytes = self._crop_from_full_page(full_page_image, item["bbox"])

        return {
            "type": "table",
            "image_bytes": image_bytes,
            "image_path": item.get("img_path", ""),
            "bbox": item.get("bbox", []),
            "page_idx": item.get("page_idx", 0),
            "html": item.get("table_body", ""),
            "captions": item.get("table_caption", []),
            "footnotes": item.get("table_footnote", []),
        }

    @staticmethod
    def _crop_from_full_page(
        full_page_bytes: bytes,
        bbox: List[float],
    ) -> Optional[bytes]:
        """
        Crop a region from the full-page PNG using MinerU's bbox format.

        ``bbox`` is ``[x0, y0, x1, y1]`` in 0-1000 normalised coordinates.
        """
        try:
            with Image.open(BytesIO(full_page_bytes)) as img:
                img = img.convert("RGB")
                w, h = img.size
                x0, y0, x1, y1 = bbox

                left = max(0, min(w, int(x0 * w / 1000)))
                right = max(0, min(w, int(x1 * w / 1000)))
                top = max(0, min(h, int(y0 * h / 1000)))
                bottom = max(0, min(h, int(y1 * h / 1000)))

                if right <= left or bottom <= top:
                    return None

                crop = img.crop((left, top, right, bottom))
                buf = BytesIO()
                crop.save(buf, format="PNG")
                return buf.getvalue()
        except Exception:
            return None
