import re
from dataclasses import dataclass, field
from uuid import NAMESPACE_URL, uuid5

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")
FENCE_RE = re.compile(r"^\s*```")
HORIZONTAL_RULE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
LIST_RE = re.compile(
    r"^\s*(?:[-+*]|\d+[.)]|[（(]?[一二三四五六七八九十]+[、.)）])\s+"
)
NUMBERING_ONLY_RE = re.compile(
    r"^\s*(?:\d+[.)、]?|[（(]?\d+[)）]|[一二三四五六七八九十]+[、.)）]?)\s*$"
)
MARKDOWN_MARK_RE = re.compile(r"[`*_>#~]+")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)]\([^)]*\)")
CONTENT_CHAR_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]")
REFERENCE_POINTER_LINE_RE = re.compile(r"^>\s*关联规则")


@dataclass(slots=True)
class SplitPiece:
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class _Section:
    heading_title: str | None
    heading_path: str
    parent_heading_path: str
    heading_level: int | None
    body_lines: list[str]


@dataclass(slots=True)
class _ContentBlock:
    kind: str  # prose | list | code | table
    lines: list[str]


def strip_markdown(value: str) -> str:
    text = MARKDOWN_LINK_RE.sub(r"\1", value)
    text = re.sub(r"^\s*(?:#{1,6}|[-+*]|\d+[.)])\s+", "", text, flags=re.MULTILINE)
    text = MARKDOWN_MARK_RE.sub("", text)
    return " ".join(text.split()).strip()


def normalize_content(value: str) -> str:
    return "".join(strip_markdown(value).lower().split())


def is_reference_pointer_content(content: str) -> bool:
    stripped = content.strip()
    if not stripped:
        return False
    lines = [line.strip() for line in stripped.split("\n") if line.strip()]
    if not lines:
        return False
    return all(REFERENCE_POINTER_LINE_RE.match(line) for line in lines)


def is_low_information_content(content: str, heading: str | None = None) -> bool:
    stripped = content.strip()
    if not stripped or HORIZONTAL_RULE_RE.fullmatch(stripped):
        return True
    plain = strip_markdown(stripped)
    if not plain or not CONTENT_CHAR_RE.search(plain):
        return True
    if NUMBERING_ONLY_RE.fullmatch(plain):
        return True
    return bool(heading and normalize_content(plain) == normalize_content(heading))


def stable_section_id(document_id: str, index_version: str, heading_path: str) -> str:
    section_key = heading_path.strip() or "__document_root__"
    return str(uuid5(NAMESPACE_URL, f"rag-center:{document_id}:{index_version}:{section_key}"))


def build_retrieval_text(*, title: str, heading_path: str, content: str) -> str:
    lines = [f"文档：{title.strip()}"]
    if heading_path.strip():
        lines.append(f"章节：{heading_path.replace('/', ' > ')}")
    lines.append(f"正文：{content.strip()}")
    return "\n".join(lines)


class MarkdownStructuredSplitter:
    def __init__(
        self,
        chunk_size: int,
        chunk_overlap: int,
        *,
        table_max_rows_per_chunk: int = 10,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.table_max_rows_per_chunk = table_max_rows_per_chunk

    def split(
        self,
        text: str,
        *,
        document_id: str = "standalone",
        index_version: str = "v2",
    ) -> list[SplitPiece]:
        normalized = text.replace("\r\n", "\n").strip()
        if not normalized:
            return []

        sections = self._split_into_sections(normalized.split("\n"))
        raw_pieces: list[SplitPiece] = []
        for section in sections:
            raw_pieces.extend(self._chunk_section(section))

        pieces: list[SplitPiece] = []
        seen_in_section: set[tuple[str, str]] = set()
        for piece in raw_pieces:
            heading_title = piece.metadata.pop("_heading_title", None)
            parent_heading_path = piece.metadata.pop("_parent_heading_path", "")
            is_pointer = piece.metadata.get("chunk_type") == "reference_pointer"
            if not is_pointer and is_low_information_content(piece.text, heading_title):
                continue
            heading_path = str(piece.metadata.get("heading_path") or "")
            section_id = stable_section_id(document_id, index_version, heading_path)
            dedupe_key = (section_id, normalize_content(piece.text))
            if dedupe_key in seen_in_section:
                continue
            seen_in_section.add(dedupe_key)
            piece.metadata.update(
                {
                    "section_id": section_id,
                    "parent_section_id": stable_section_id(
                        document_id, index_version, parent_heading_path
                    )
                    if parent_heading_path
                    else None,
                    "order_index": len(pieces),
                }
            )
            pieces.append(piece)
        return pieces

    def _split_into_sections(self, lines: list[str]) -> list[_Section]:
        sections: list[_Section] = []
        heading_stack: list[tuple[int, str]] = []
        current_heading_title: str | None = None
        current_heading_path = ""
        current_parent_heading_path = ""
        current_heading_level: int | None = None
        current_body: list[str] = []

        def flush() -> None:
            nonlocal current_heading_title, current_heading_path
            nonlocal current_parent_heading_path, current_heading_level, current_body
            if current_heading_title is not None or any(line.strip() for line in current_body):
                sections.append(
                    _Section(
                        heading_title=current_heading_title,
                        heading_path=current_heading_path,
                        parent_heading_path=current_parent_heading_path,
                        heading_level=current_heading_level,
                        body_lines=current_body,
                    )
                )
            current_heading_title = None
            current_heading_path = ""
            current_parent_heading_path = ""
            current_heading_level = None
            current_body = []

        for line in lines:
            if self._is_horizontal_rule(line):
                continue

            match = HEADING_RE.match(line)
            if match:
                flush()
                level = len(match.group(1))
                title = match.group(2).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                parent_heading_path = "/".join(item[1] for item in heading_stack)
                heading_stack.append((level, title))
                current_heading_title = title
                current_heading_path = "/".join(item[1] for item in heading_stack)
                current_parent_heading_path = parent_heading_path
                current_heading_level = level
                continue

            current_body.append(line)

        flush()

        if not sections and lines:
            return [
                _Section(
                    heading_title=None,
                    heading_path="",
                    parent_heading_path="",
                    heading_level=None,
                    body_lines=lines,
                )
            ]
        return sections

    def _chunk_section(self, section: _Section) -> list[SplitPiece]:
        blocks = self._parse_blocks(section.body_lines)
        pieces: list[SplitPiece] = []

        for block in blocks:
            if block.kind == "table":
                block_pieces = self._chunk_table(block.lines, section.heading_path)
            else:
                block_pieces = self._chunk_prose_block(
                    block.lines, section.heading_path, block.kind
                )
            for piece in block_pieces:
                piece.metadata.update(
                    {
                        "heading_level": section.heading_level,
                        "parser": "markdown",
                        "_heading_title": section.heading_title,
                        "_parent_heading_path": section.parent_heading_path,
                    }
                )
                pieces.append(piece)

        return self._merge_section_pieces(pieces)

    def _parse_blocks(self, lines: list[str]) -> list[_ContentBlock]:
        blocks: list[_ContentBlock] = []
        index = 0
        while index < len(lines):
            line = lines[index]
            if not line.strip():
                index += 1
                continue
            if FENCE_RE.match(line):
                fence_lines = [line]
                index += 1
                while index < len(lines) and not FENCE_RE.match(lines[index]):
                    fence_lines.append(lines[index])
                    index += 1
                if index < len(lines):
                    fence_lines.append(lines[index])
                    index += 1
                blocks.append(_ContentBlock(kind="code", lines=fence_lines))
                continue

            if self._is_table_line(line):
                table_lines = [line]
                index += 1
                while index < len(lines) and self._is_table_line(lines[index]):
                    table_lines.append(lines[index])
                    index += 1
                blocks.append(_ContentBlock(kind="table", lines=table_lines))
                continue

            if LIST_RE.match(line):
                list_lines = [line]
                index += 1
                while index < len(lines):
                    current = lines[index]
                    if LIST_RE.match(current) or (current.strip() and current[:1].isspace()):
                        list_lines.append(current)
                        index += 1
                        continue
                    break
                blocks.append(_ContentBlock(kind="list", lines=list_lines))
                continue

            prose_lines: list[str] = []
            while index < len(lines):
                current = lines[index]
                if (
                    FENCE_RE.match(current)
                    or self._is_table_line(current)
                    or LIST_RE.match(current)
                ):
                    break
                prose_lines.append(current)
                index += 1
            if prose_lines:
                blocks.append(_ContentBlock(kind="prose", lines=prose_lines))

        return blocks

    def _merge_section_pieces(self, pieces: list[SplitPiece]) -> list[SplitPiece]:
        if not pieces:
            return []

        folded: list[SplitPiece] = []
        leading_pointers: list[SplitPiece] = []
        for piece in pieces:
            if piece.metadata.get("chunk_type") == "reference_pointer":
                if folded:
                    self._append_piece(folded[-1], piece)
                else:
                    leading_pointers.append(piece)
                continue
            if leading_pointers:
                for pointer in leading_pointers:
                    self._append_piece(piece, pointer, prepend=True)
                leading_pointers = []
            folded.append(piece)
        if leading_pointers:
            folded.extend(leading_pointers)

        merged: list[SplitPiece] = []
        for piece in folded:
            if merged and self._can_merge_pieces(merged[-1], piece):
                self._append_piece(merged[-1], piece)
                continue
            merged.append(piece)
        return merged

    def _can_merge_pieces(self, left: SplitPiece, right: SplitPiece) -> bool:
        if left.metadata.get("chunk_type") == "reference_pointer":
            return True
        if right.metadata.get("chunk_type") == "reference_pointer":
            return True
        if left.metadata.get("table_part") is not None and right.metadata.get("table_part") is not None:
            return False
        combined = len(left.text) + 2 + len(right.text)
        return combined <= self.chunk_size

    @staticmethod
    def _append_piece(target: SplitPiece, source: SplitPiece, *, prepend: bool = False) -> None:
        if prepend:
            target.text = f"{source.text.rstrip()}\n\n{target.text.lstrip()}"
        else:
            target.text = f"{target.text.rstrip()}\n\n{source.text.lstrip()}"
        source_type = str(source.metadata.get("chunk_type") or "prose")
        target_type = str(target.metadata.get("chunk_type") or "prose")
        types = {target_type, source_type}
        types.discard("reference_pointer")
        if types == {"table"}:
            chunk_type = "table"
        elif types == {"list"}:
            chunk_type = "list"
        elif types == {"code"}:
            chunk_type = "code"
        else:
            chunk_type = "prose"
        target.metadata["chunk_type"] = chunk_type
        refs = list(target.metadata.get("reference_texts") or [])
        refs.extend(source.metadata.get("reference_texts") or [])
        if source_type == "reference_pointer" and source.text not in refs:
            refs.append(source.text)
        if refs:
            target.metadata["reference_texts"] = refs

    def _chunk_prose_block(
        self,
        lines: list[str],
        heading_path: str,
        kind: str,
    ) -> list[SplitPiece]:
        text = "\n".join(lines).strip()
        if not text:
            return []

        chunk_type = kind if kind in {"code", "list"} else "prose"
        paragraphs = re.split(r"\n\s*\n", text)
        if len(paragraphs) > 1 and kind != "code":
            return self._split_paragraphs(
                [paragraph.strip() for paragraph in paragraphs if paragraph.strip()],
                heading_path,
                chunk_type,
            )
        if is_reference_pointer_content(text):
            return [
                SplitPiece(
                    text=text,
                    metadata={
                        "heading_path": heading_path,
                        "chunk_type": "reference_pointer",
                        "reference_texts": [text],
                    },
                )
            ]
        if len(text) <= self.chunk_size:
            return [
                SplitPiece(
                    text=text,
                    metadata={"heading_path": heading_path, "chunk_type": chunk_type},
                )
            ]

        if kind == "code":
            return self._split_by_characters(text, heading_path, chunk_type)

        paragraphs = re.split(r"\n\s*\n", text)
        return self._split_paragraphs(paragraphs, heading_path, chunk_type)

    def _chunk_table(self, table_lines: list[str], heading_path: str) -> list[SplitPiece]:
        cleaned_lines = [line for line in table_lines if line.strip()]
        if not cleaned_lines:
            return []

        header_line, body_start = self._resolve_table_header(cleaned_lines)
        data_rows = cleaned_lines[body_start:]
        header_summary = self._table_header_summary(header_line)

        def build_table_chunk(row_group: list[str], part: int) -> SplitPiece:
            body = "\n".join([header_line, *row_group]) if row_group else header_line
            if header_summary:
                content = f"{header_summary}\n{body}"
            else:
                content = body
            metadata: dict = {
                "heading_path": heading_path,
                "chunk_type": "table",
            }
            if part > 0:
                metadata["table_part"] = part
            return SplitPiece(text=content.strip(), metadata=metadata)

        full_table = "\n".join(cleaned_lines)
        if len(full_table) <= self.chunk_size:
            content = f"{header_summary}\n{full_table}".strip() if header_summary else full_table
            return [
                SplitPiece(
                    text=content,
                    metadata={
                        "heading_path": heading_path,
                        "chunk_type": "table",
                        "table_part": 1,
                    },
                )
            ]

        pieces: list[SplitPiece] = []
        part = 1
        for start in range(0, max(len(data_rows), 1), self.table_max_rows_per_chunk):
            group = data_rows[start : start + self.table_max_rows_per_chunk]
            if not group and part > 1:
                break
            candidate = build_table_chunk(group, part)
            if len(candidate.text) > self.chunk_size and len(group) > 1:
                for row in group:
                    pieces.append(build_table_chunk([row], part))
                    part += 1
                continue
            pieces.append(candidate)
            part += 1

        return pieces

    def _resolve_table_header(self, lines: list[str]) -> tuple[str, int]:
        if len(lines) >= 2 and self._is_table_separator(lines[1]):
            return lines[0], 2
        return lines[0], 1

    def _table_header_summary(self, header_line: str) -> str:
        cells = [cell.strip() for cell in header_line.strip().strip("|").split("|")]
        cells = [cell for cell in cells if cell]
        if not cells:
            return ""
        return f"【表头】{' | '.join(cells)}"

    def _split_paragraphs(
        self,
        paragraphs: list[str],
        heading_path: str,
        chunk_type: str,
    ) -> list[SplitPiece]:
        pieces: list[SplitPiece] = []
        buffer: list[str] = []
        buffer_len = 0

        def flush_buffer() -> None:
            nonlocal buffer, buffer_len
            if not buffer:
                return
            text = "\n\n".join(buffer).strip()
            if text:
                pieces.append(
                    SplitPiece(
                        text=text,
                        metadata={"heading_path": heading_path, "chunk_type": chunk_type},
                    )
                )
            buffer = []
            buffer_len = 0

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            if is_reference_pointer_content(paragraph):
                flush_buffer()
                pieces.append(
                    SplitPiece(
                        text=paragraph,
                        metadata={
                            "heading_path": heading_path,
                            "chunk_type": "reference_pointer",
                            "reference_texts": [paragraph],
                        },
                    )
                )
                continue
            extra = len(paragraph) + (2 if buffer else 0)
            if buffer and buffer_len + extra > self.chunk_size:
                flush_buffer()
            if len(paragraph) > self.chunk_size:
                flush_buffer()
                pieces.extend(self._split_by_characters(paragraph, heading_path, chunk_type))
                continue
            buffer.append(paragraph)
            buffer_len += extra

        flush_buffer()
        return pieces

    def _split_by_characters(
        self,
        text: str,
        heading_path: str,
        chunk_type: str,
    ) -> list[SplitPiece]:
        if len(text) <= self.chunk_size:
            return [
                SplitPiece(
                    text=text,
                    metadata={"heading_path": heading_path, "chunk_type": chunk_type},
                )
            ]

        pieces: list[SplitPiece] = []
        start = 0
        step = max(self.chunk_size - self.chunk_overlap, 1)
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end].strip()
            if chunk:
                pieces.append(
                    SplitPiece(
                        text=chunk,
                        metadata={"heading_path": heading_path, "chunk_type": chunk_type},
                    )
                )
            if end >= len(text):
                break
            start += step
        return pieces

    @staticmethod
    def _is_horizontal_rule(line: str) -> bool:
        return bool(HORIZONTAL_RULE_RE.match(line.strip()))

    @staticmethod
    def _is_table_line(line: str) -> bool:
        stripped = line.strip()
        if not stripped.startswith("|"):
            return False
        return stripped.count("|") >= 2

    @staticmethod
    def _is_table_separator(line: str) -> bool:
        stripped = line.strip().strip("|").replace(" ", "")
        if not stripped:
            return False
        return bool(re.fullmatch(r"[\|:_-]+", stripped))
