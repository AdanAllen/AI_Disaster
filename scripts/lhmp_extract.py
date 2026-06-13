#!/usr/bin/env python3
"""Create non-public LHMP evidence candidates from a PDF and optional Markdown."""

import argparse
import hashlib
import json
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lhmp_models import (  # noqa: E402
    CandidatePlanManifest,
    DataSourceCandidate,
    EvidenceCandidate,
    TableCandidate,
    VisualCandidate,
)


EXTRACTOR_VERSION = "1.0.0"
CITY_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
CAPTION_RE = re.compile(
    r"^(?P<kind>figure|map|chart|diagram|exhibit)\s+(?P<label>[A-Za-z0-9][A-Za-z0-9.\-:]*)\s*(?P<title>.*)$",
    re.IGNORECASE,
)
TABLE_RE = re.compile(
    r"^table\s+(?P<label>[A-Za-z0-9][A-Za-z0-9.\-:]*)\s*(?P<title>.*)$",
    re.IGNORECASE,
)
SOURCE_RE = re.compile(r"^(?:source|data source|prepared by)\s*:\s*(.+)$", re.IGNORECASE)
URL_RE = re.compile(r"https?://[^\s)>]+")
MARKDOWN_PAGE_RE = re.compile(
    r"(?:<!--\s*)?(?:pdf[_ ]page|pdf-page)\s*[:#]?\s*(\d+)"
    r"(?:\s*[,;|]\s*page[_ ]label\s*[:=]\s*([A-Za-z0-9.\-]+))?\s*(?:-->)?",
    re.IGNORECASE,
)
MARKDOWN_BRACKET_PAGE_RE = re.compile(
    r"^\[\s*PDF\s+page\s+(\d+)(?:\s*[|,;]\s*(?:label\s+)?([A-Za-z0-9.\-]+))?\s*\]$",
    re.IGNORECASE,
)
PLANNING_KEYWORDS = ("hazard", "risk", "vulnerab", "mitigation", "exposure")


@dataclass
class PageRecord:
    pdf_page: int
    page_label: str
    text: str
    normalized_text: str
    lines: List[str]
    image_count: int


def _load_pymupdf():
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for offline LHMP extraction. "
            "Install requirements-lhmp.txt; do not add it to the Flask runtime."
        ) from exc
    return fitz


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _clean_text(value: str, limit: int = 2400) -> str:
    return re.sub(r"\s+", " ", value or "").strip()[:limit]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_id(prefix: str, *parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts)
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}_{suffix}"


def _safe_city_slug(city: str) -> str:
    slug = (city or "").strip().lower()
    if not CITY_SLUG_RE.fullmatch(slug):
        raise ValueError("city must be a lowercase URL-style slug")
    return slug


def _safe_output_directory(project_root: Path, city: str) -> Path:
    extracted_root = (project_root / "data" / "lhmp" / "extracted").resolve()
    output_directory = (extracted_root / city).resolve()
    if not output_directory.is_relative_to(extracted_root):
        raise ValueError("extraction output must remain inside data/lhmp/extracted")
    return output_directory


def _load_aliases(project_root: Path, city: str) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    path = project_root / "data" / "lhmp" / "hazard_aliases.json"
    with path.open("r", encoding="utf-8") as source:
        payload = json.load(source)
    aliases = payload.get("aliases")
    if not isinstance(aliases, dict):
        raise ValueError("hazard_aliases.json must contain an aliases object")
    normalized = {}
    for hazard, values in aliases.items():
        if not isinstance(values, list):
            raise ValueError(f"hazard aliases for {hazard} must be a list")
        normalized[str(hazard)] = [
            _normalize_text(str(value)) for value in values if _normalize_text(str(value))
        ]
    city_overrides = payload.get("city_overrides", {}).get(city, {})
    if not isinstance(city_overrides, dict):
        raise ValueError(f"city override for {city} must be an object")
    return normalized, {
        _normalize_text(str(alias)): str(hazard)
        for alias, hazard in city_overrides.items()
        if _normalize_text(str(alias))
    }


def _suggest_hazard(
    text: str,
    aliases: Dict[str, List[str]],
    city_overrides: Dict[str, str],
) -> Tuple[Optional[str], Optional[str]]:
    normalized = f" {_normalize_text(text)} "
    matches = []
    for alias, hazard in city_overrides.items():
        if f" {alias} " in normalized:
            matches.append((len(alias), hazard, alias))
    for hazard, values in aliases.items():
        for alias in values:
            if f" {alias} " in normalized:
                matches.append((len(alias), hazard, alias))
    if not matches:
        return None, None
    _, hazard, alias = max(matches, key=lambda item: item[0])
    return hazard, alias


def _infer_section_heading(lines: List[str], line_index: int) -> str:
    for candidate in reversed(lines[max(0, line_index - 8):line_index]):
        cleaned = _clean_text(candidate, 180)
        if not cleaned or len(cleaned) > 140:
            continue
        if cleaned.isupper() or cleaned.istitle() or re.match(r"^\d+(?:\.\d+)*\s+\S+", cleaned):
            return cleaned
    return ""


def _page_label(page, pdf_page: int, lines: List[str]) -> str:
    try:
        label = _clean_text(page.get_label(), 40)
    except Exception:
        label = ""
    return label


def _portable_input_path(path: Optional[Path], project_root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return str(path)


def _read_pages(document) -> List[PageRecord]:
    pages = []
    for index in range(document.page_count):
        page = document.load_page(index)
        text = page.get_text("text") or ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        pages.append(PageRecord(
            pdf_page=index + 1,
            page_label=_page_label(page, index + 1, lines),
            text=text,
            normalized_text=_normalize_text(text),
            lines=lines,
            image_count=len(page.get_images(full=True)),
        ))
    return pages


def _paragraphs(page: PageRecord) -> Iterable[Tuple[str, int]]:
    blocks = re.split(r"\n\s*\n", page.text)
    if len(blocks) == 1:
        blocks = page.lines
    line_cursor = 0
    for block in blocks:
        cleaned = _clean_text(block)
        if len(cleaned) < 60:
            continue
        first_line = _clean_text(block.splitlines()[0], 160)
        try:
            line_index = page.lines.index(first_line, line_cursor)
        except ValueError:
            line_index = line_cursor
        line_cursor = max(line_cursor, line_index)
        yield cleaned, line_index


def _resolve_markdown_page(snippet: str, pages: List[PageRecord]) -> Optional[PageRecord]:
    normalized = _normalize_text(snippet)
    if len(normalized) < 40:
        return None
    probes = (normalized[:180], normalized[:120], normalized[:80])
    for probe in probes:
        matches = [page for page in pages if probe and probe in page.normalized_text]
        if len(matches) == 1:
            return matches[0]
    return None


def _markdown_candidates(
    markdown_path: Path,
    *,
    city: str,
    source_document: str,
    pages: List[PageRecord],
    aliases: Dict[str, List[str]],
    city_overrides: Dict[str, str],
    hazard_filter: Optional[str],
    warnings: List[str],
) -> List[EvidenceCandidate]:
    content = markdown_path.read_text(encoding="utf-8")
    heading = ""
    explicit_pdf_page = None
    explicit_page_label = ""
    paragraph_lines = []
    records = []

    def flush():
        nonlocal paragraph_lines, explicit_pdf_page, explicit_page_label
        raw = "\n".join(paragraph_lines).strip()
        paragraph_lines = []
        if len(_clean_text(raw)) < 40:
            return
        marker_pdf_page = explicit_pdf_page
        marker_page_label = explicit_page_label
        explicit_pdf_page = None
        explicit_page_label = ""
        page = None
        if marker_pdf_page is not None and 1 <= marker_pdf_page <= len(pages):
            page = pages[marker_pdf_page - 1]
        elif marker_pdf_page is not None:
            warnings.append(
                f"Markdown page marker {marker_pdf_page} is outside the PDF page range."
            )
            return
        else:
            page = _resolve_markdown_page(raw, pages)
        if page is None:
            warnings.append(
                f"Skipped unresolved Markdown evidence: {_clean_text(raw, 100)}"
            )
            return
        hazard, alias = _suggest_hazard(f"{heading} {raw}", aliases, city_overrides)
        if hazard_filter and hazard != hazard_filter:
            return
        reason = (
            f"Markdown snippet matched PDF text and contained configured alias “{alias}”."
            if alias
            else "Markdown snippet matched PDF text; no configured hazard alias was found."
        )
        records.append(EvidenceCandidate(
            id=_stable_id("lhmp_evidence", city, source_document, page.pdf_page, raw),
            city=city,
            source_document=source_document,
            pdf_page=page.pdf_page,
            page_label=marker_page_label or page.page_label,
            section_heading=heading,
            suggested_hazard=hazard,
            extraction_reason=reason,
            original_text=raw,
            review_status="candidate",
        ))

    for line in content.splitlines():
        marker = MARKDOWN_PAGE_RE.search(line) or MARKDOWN_BRACKET_PAGE_RE.match(line.strip())
        if marker:
            flush()
            explicit_pdf_page = int(marker.group(1))
            explicit_page_label = marker.group(2) or ""
            continue
        if line.lstrip().startswith("#"):
            flush()
            heading = line.lstrip("#").strip()
            continue
        if not line.strip():
            flush()
            continue
        paragraph_lines.append(line)
    flush()
    return records


def _pdf_evidence_candidates(
    *,
    city: str,
    source_document: str,
    pages: List[PageRecord],
    aliases: Dict[str, List[str]],
    city_overrides: Dict[str, str],
    hazard_filter: Optional[str],
    max_candidates: int,
    warnings: List[str],
) -> List[EvidenceCandidate]:
    records = []
    for page in pages:
        for snippet, line_index in _paragraphs(page):
            hazard, alias = _suggest_hazard(snippet, aliases, city_overrides)
            if hazard_filter and hazard != hazard_filter:
                continue
            normalized = _normalize_text(snippet)
            if not alias and not any(keyword in normalized for keyword in PLANNING_KEYWORDS):
                continue
            reason = (
                f"PDF source text contained configured alias “{alias}”."
                if alias
                else "PDF source text contained a planning keyword but no configured hazard alias."
            )
            records.append(EvidenceCandidate(
                id=_stable_id("lhmp_evidence", city, source_document, page.pdf_page, snippet),
                city=city,
                source_document=source_document,
                pdf_page=page.pdf_page,
                page_label=page.page_label,
                section_heading=_infer_section_heading(page.lines, line_index),
                suggested_hazard=hazard,
                extraction_reason=reason,
                extracted_snippet=snippet,
                review_status="candidate",
            ))
            if len(records) >= max_candidates:
                warnings.append(
                    f"Evidence candidate limit reached ({max_candidates}); remaining snippets were skipped."
                )
                return records
    return records


def _caption_candidates(page: PageRecord):
    output = []
    for index, line in enumerate(page.lines):
        visual_match = CAPTION_RE.match(line)
        table_match = TABLE_RE.match(line)
        if visual_match:
            kind = visual_match.group("kind").lower()
            if kind == "exhibit":
                kind = "other"
            output.append(("visual", kind, line, index))
        elif table_match:
            output.append(("table", "table", line, index))
    return output


def _source_candidates(
    page: PageRecord,
    *,
    city: str,
    source_document: str,
    aliases: Dict[str, List[str]],
    city_overrides: Dict[str, str],
    hazard_filter: Optional[str],
) -> List[DataSourceCandidate]:
    output = []
    for index, line in enumerate(page.lines):
        match = SOURCE_RE.match(line)
        urls = URL_RE.findall(line)
        if not match and not urls:
            continue
        source_text = _clean_text(match.group(1) if match else line)
        hazard, alias = _suggest_hazard(
            f"{_infer_section_heading(page.lines, index)} {source_text} {page.text}",
            aliases,
            city_overrides,
        )
        if hazard_filter and hazard != hazard_filter:
            continue
        output.append(DataSourceCandidate(
            id=_stable_id("lhmp_source", city, source_document, page.pdf_page, source_text),
            city=city,
            source_document=source_document,
            pdf_page=page.pdf_page,
            page_label=page.page_label,
            section_heading=_infer_section_heading(page.lines, index),
            suggested_hazard=hazard,
            extraction_reason=(
                f"Source line contained configured hazard alias “{alias}”."
                if alias
                else "PDF text matched a source label or URL."
            ),
            source_text=source_text,
            suggested_url=urls[0].rstrip(".,;") if urls else "",
            review_status="candidate",
        ))
    return output


def _write_json_atomic(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepare_output(output_directory: Path, replace_candidates: bool) -> Path:
    candidate_names = (
        "plan_manifest.json",
        "evidence_candidates.json",
        "visuals_candidates.json",
        "tables_candidates.json",
        "data_sources_candidates.json",
    )
    page_images = output_directory / "page-images"
    existing = [output_directory / name for name in candidate_names if (output_directory / name).exists()]
    if page_images.exists() and any(
        child.name != ".gitkeep" for child in page_images.iterdir()
    ):
        existing.append(page_images)
    if existing and not replace_candidates:
        raise FileExistsError(
            "candidate output already exists; pass --replace-candidates to replace extracted candidates"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    if replace_candidates and page_images.exists():
        for child in page_images.iterdir():
            if child.name == ".gitkeep":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    page_images.mkdir(parents=True, exist_ok=True)
    return page_images


def extract_lhmp(
    *,
    city: str,
    pdf_path: Path,
    markdown_path: Optional[Path] = None,
    source_url: str = "",
    hazard_filter: Optional[str] = None,
    max_render_pages: int = 30,
    max_evidence_candidates: int = 200,
    replace_candidates: bool = False,
    project_root: Path = PROJECT_ROOT,
) -> Dict:
    city = _safe_city_slug(city)
    project_root = Path(project_root).resolve()
    pdf_path = Path(pdf_path).resolve()
    markdown_path = Path(markdown_path).resolve() if markdown_path else None
    if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
        raise ValueError("pdf path must identify a readable PDF file")
    if markdown_path and (not markdown_path.is_file() or markdown_path.suffix.lower() not in {".md", ".markdown"}):
        raise ValueError("markdown path must identify a readable Markdown file")
    if max_render_pages < 0 or max_render_pages > 200:
        raise ValueError("max_render_pages must be between 0 and 200")
    if max_evidence_candidates < 1 or max_evidence_candidates > 5000:
        raise ValueError("max_evidence_candidates must be between 1 and 5000")

    aliases, city_overrides = _load_aliases(project_root, city)
    canonical_hazards = set(aliases)
    if hazard_filter:
        hazard_filter = hazard_filter.strip().lower()
        if hazard_filter not in canonical_hazards:
            raise ValueError("hazard filter must be a canonical hazard from hazard_aliases.json")

    output_directory = _safe_output_directory(project_root, city)
    page_images_directory = _prepare_output(output_directory, replace_candidates)
    fitz = _load_pymupdf()
    warnings = []

    document = fitz.open(pdf_path)
    try:
        if document.page_count <= 0:
            raise ValueError("PDF contains no pages")
        pages = _read_pages(document)
        if markdown_path:
            evidence = _markdown_candidates(
                markdown_path,
                city=city,
                source_document=pdf_path.name,
                pages=pages,
                aliases=aliases,
                city_overrides=city_overrides,
                hazard_filter=hazard_filter,
                warnings=warnings,
            )
        else:
            evidence = _pdf_evidence_candidates(
                city=city,
                source_document=pdf_path.name,
                pages=pages,
                aliases=aliases,
                city_overrides=city_overrides,
                hazard_filter=hazard_filter,
                max_candidates=max_evidence_candidates,
                warnings=warnings,
            )

        page_caption_records = {}
        for page in pages:
            matching_records = []
            for record_type, kind, caption, line_index in _caption_candidates(page):
                hazard, _ = _suggest_hazard(
                    f"{_infer_section_heading(page.lines, line_index)} {caption} {page.text}",
                    aliases,
                    city_overrides,
                )
                if hazard_filter and hazard != hazard_filter:
                    continue
                matching_records.append((record_type, kind, caption, line_index))
            if matching_records:
                page_caption_records[page.pdf_page] = matching_records
        likely_pages = list(page_caption_records)
        if len(likely_pages) > max_render_pages:
            warnings.append(
                f"Visual page limit reached ({max_render_pages}); "
                f"{len(likely_pages) - max_render_pages} likely pages were not rendered."
            )
        rendered_pages = likely_pages[:max_render_pages]
        rendered_paths = {}
        for pdf_page in rendered_pages:
            destination = page_images_directory / f"page-{pdf_page:04d}.png"
            page = document.load_page(pdf_page - 1)
            pixmap = page.get_pixmap(matrix=fitz.Matrix(1.35, 1.35), alpha=False)
            pixmap.save(destination)
            rendered_paths[pdf_page] = destination.relative_to(project_root).as_posix()

        visuals = []
        tables = []
        for page in pages:
            for record_type, kind, caption, line_index in page_caption_records.get(page.pdf_page, []):
                page_image_path = rendered_paths.get(page.pdf_page, "")
                hazard, alias = _suggest_hazard(
                    f"{_infer_section_heading(page.lines, line_index)} {caption} {page.text}",
                    aliases,
                    city_overrides,
                )
                if hazard_filter and hazard != hazard_filter:
                    continue
                reason = (
                    f"Caption contained configured hazard alias “{alias}”."
                    if alias
                    else "PDF text matched a figure or table caption pattern."
                )
                common = {
                    "city": city,
                    "source_document": pdf_path.name,
                    "pdf_page": page.pdf_page,
                    "page_label": page.page_label,
                    "section_heading": _infer_section_heading(page.lines, line_index),
                    "suggested_hazard": hazard,
                    "extraction_reason": reason,
                    "review_status": "candidate",
                }
                if record_type == "visual" and page_image_path:
                    visuals.append(VisualCandidate(
                        id=_stable_id("lhmp_visual", city, pdf_path.name, page.pdf_page, caption),
                        visual_type=kind,
                        caption=caption,
                        page_image_path=page_image_path,
                        address_specific=False,
                        **common,
                    ))
                elif record_type == "table":
                    nearby_text = " ".join(
                        page.lines[line_index:min(len(page.lines), line_index + 16)]
                    )
                    tables.append(TableCandidate(
                        id=_stable_id("lhmp_table", city, pdf_path.name, page.pdf_page, caption),
                        caption=caption,
                        extracted_text=_clean_text(nearby_text),
                        page_image_path=page_image_path,
                        **common,
                    ))

        data_sources = []
        for page in pages:
            data_sources.extend(_source_candidates(
                page,
                city=city,
                source_document=pdf_path.name,
                aliases=aliases,
                city_overrides=city_overrides,
                hazard_filter=hazard_filter,
            ))

        manifest = CandidatePlanManifest(
            city=city,
            source_document=pdf_path.name,
            source_pdf_path=_portable_input_path(pdf_path, project_root),
            markdown_path=_portable_input_path(markdown_path, project_root),
            source_url=source_url,
            source_pdf_sha256=_sha256(pdf_path),
            source_pdf_size_bytes=pdf_path.stat().st_size,
            pdf_page_count=document.page_count,
            extracted_at=datetime.now(timezone.utc),
            extractor_version=EXTRACTOR_VERSION,
            pymupdf_version=str(getattr(fitz, "VersionBind", "unknown")),
            hazard_filter=hazard_filter,
            rendered_pdf_pages=rendered_pages,
            warnings=list(dict.fromkeys(warnings)),
            review_status="candidate",
        )
    finally:
        document.close()

    payloads = {
        "plan_manifest.json": manifest.model_dump(mode="json"),
        "evidence_candidates.json": [item.model_dump(mode="json") for item in evidence],
        "visuals_candidates.json": [item.model_dump(mode="json") for item in visuals],
        "tables_candidates.json": [item.model_dump(mode="json") for item in tables],
        "data_sources_candidates.json": [
            item.model_dump(mode="json") for item in data_sources
        ],
    }
    for filename, payload in payloads.items():
        _write_json_atomic(output_directory / filename, payload)
    return {
        "output_directory": str(output_directory),
        "evidence_candidates": len(evidence),
        "visual_candidates": len(visuals),
        "table_candidates": len(tables),
        "data_source_candidates": len(data_sources),
        "rendered_pages": rendered_pages,
        "warnings": manifest.warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract non-public LHMP review candidates from a PDF."
    )
    parser.add_argument("--city", required=True)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--source-url", default="")
    parser.add_argument("--hazard", dest="hazard_filter")
    parser.add_argument("--max-render-pages", type=int, default=30)
    parser.add_argument("--max-evidence-candidates", type=int, default=200)
    parser.add_argument("--replace-candidates", action="store_true")
    args = parser.parse_args()
    try:
        result = extract_lhmp(
            city=args.city,
            pdf_path=args.pdf,
            markdown_path=args.markdown,
            source_url=args.source_url,
            hazard_filter=args.hazard_filter,
            max_render_pages=args.max_render_pages,
            max_evidence_candidates=args.max_evidence_candidates,
            replace_candidates=args.replace_candidates,
        )
    except (OSError, ValueError, RuntimeError, FileExistsError, json.JSONDecodeError) as exc:
        parser.exit(2, f"lhmp_extract: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
