#!/usr/bin/env python3
"""M1-1: 全量解析 111 份 Word/PDF → 条款级文本块库。

Word(.docx,含 .doc 转换件):逐段落抽取,保留表格文本;
PDF:逐页抽取(pdfplumber)。
两者统一做"条款感知切分":识别 第X章/第X条/(一)/1. 等编号边界,
块长超限再硬切(带重叠)。每块携带 doc_id/来源/位置/章节上下文元数据。

用法: python3 scripts/parse_docs.py
"""
import json
import os
import re
import sys

import pdfplumber
from docx import Document as Docx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_INDEX = os.path.join(ROOT, "data", "processed", "doc_index.json")
CONV = os.path.join(ROOT, "data", "processed", "converted_docx")
OUT = os.path.join(ROOT, "data", "processed", "text_blocks.jsonl")

MAX_CHARS = 420          # 单块目标上限
OVERLAP = 60             # 硬切重叠
# 结构边界:章/条/大节(一、二、)/小节((一)(二))/数字条(1. 1.1)
HEADING_RE = re.compile(
    r"^(第[一二三四五六七八九十百千\d]+[章节条款项]|附录|附件|[一二三四五六七八九十]+、|"
    r"（[一二三四五六七八九十]+）|\d{1,2}\.\d{0,2}[^\d]|(\d{1,2}\.))")


def norm(s: str) -> str:
    return re.sub(r"[ \t\r\f\v]+", "", str(s))


def iter_word_units(path: str):
    """产出 (位置描述, 文本):段落 + 表格行序列。"""
    d = Docx(path)
    for i, p in enumerate(d.paragraphs):
        if p.text.strip():
            yield f"para{i}", p.text
    for ti, t in enumerate(d.tables):
        for ri, row in enumerate(t.rows):
            cells = [c.text.strip() for c in row.cells]
            if any(cells):
                yield f"tbl{ti}r{ri}", " | ".join(cells)


def iter_pdf_units(path: str):
    with pdfplumber.open(path) as pdf:
        for pi, page in enumerate(pdf.pages, 1):
            txt = (page.extract_text() or "").strip()
            if txt:
                yield f"page{pi}", txt


def split_long(text: str):
    """超长文本按句号/分号硬切并带重叠。"""
    if len(text) <= MAX_CHARS:
        return [text]
    sents = re.split(r"(?<=[。;；])", text)
    buf, out = "", []
    for s in sents:
        if len(buf) + len(s) > MAX_CHARS and buf:
            out.append(buf)
            buf = buf[-OVERLAP:]  # 保留尾部重叠
        buf += s
    if buf:
        out.append(buf)
    # 兜底:仍超长则等距硬切
    final = []
    for b in out:
        while len(b) > MAX_CHARS * 1.6:
            final.append(b[:MAX_CHARS])
            b = b[MAX_CHARS - OVERLAP:]
        final.append(b)
    return final


def group_blocks(units):
    """把 (位置, 文本) 序列聚成条款级块。"""
    blocks, cur, cur_len = [], [], 0
    for loc, text in units:
        text = text.strip()
        is_boundary = bool(HEADING_RE.match(norm(text)))
        if is_boundary and cur and cur_len > 120:
            blocks.append(cur)
            cur, cur_len = [], 0
        elif cur_len + len(text) > MAX_CHARS and cur:
            blocks.append(cur)
            cur, cur_len = [], 0
        cur.append((loc, text))
        cur_len += len(text)
    if cur:
        blocks.append(cur)
    return blocks


def main() -> int:
    docs = json.load(open(DOC_INDEX, encoding="utf-8"))["docs"]
    text_docs = [d for d in docs if d["type"] in ("doc", "docx", "pdf")]
    n_blocks, failed = 0, []
    with open(OUT, "w", encoding="utf-8") as fp:
        for d in text_docs:
            did = d["doc_id"]
            path = (os.path.join(CONV, did + ".docx") if d["type"] == "doc"
                    else os.path.join(ROOT, d["canonical_path"]))
            try:
                units = (iter_word_units(path) if d["type"] in ("doc", "docx")
                         else iter_pdf_units(path))
                for bi, chunk in enumerate(group_blocks(units)):
                    locs = [c[0] for c in chunk]
                    text = "\n".join(c[1] for c in chunk)
                    for si, sub in enumerate(split_long(text)):
                        fp.write(json.dumps({
                            "block_id": f"{did}-{bi}-{si}",
                            "doc_id": did,
                            "orig_name": d["orig_name"],
                            "source_type": "word" if d["type"] in ("doc", "docx") else "pdf",
                            "location": locs[0] if len(set(locs)) == 1 else f"{locs[0]}-{locs[-1]}",
                            "n_units": len(chunk),
                            "text": sub,
                        }, ensure_ascii=False) + "\n")
                        n_blocks += 1
            except Exception as e:  # noqa: BLE001
                failed.append({"doc_id": did, "error": f"{type(e).__name__}: {e}"})

    print(f"文本文件 {len(text_docs)} 份,产出 {n_blocks} 块 → {os.path.relpath(OUT, ROOT)}")
    if failed:
        print(f"失败 {len(failed)}:")
        for f in failed:
            print("  ", f)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
