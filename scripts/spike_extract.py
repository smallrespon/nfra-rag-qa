#!/usr/bin/env python3
"""M1-0 解析器 spike:用评测集 evidence 原文实测轻量解析器的抽取充分性。

方法:对 dev 集文本题(word/pdf 来源),把 evidence 句子(去空白归一化后)
在被解析源文件全文(同样归一化)中查找;命中 = 该题的解析证据可获取。
按来源类型统计命中率,并列出零命中的文件(它们才是需要重型引擎的对象)。

用法: python3 scripts/spike_extract.py
"""
import json
import os
import re
import sys

import pdfplumber
from docx import Document as Docx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_INDEX = os.path.join(ROOT, "data", "processed", "doc_index.json")
QA = os.path.join(ROOT, "data", "eval", "qa_dataset.jsonl")
CONV = os.path.join(ROOT, "data", "processed", "converted_docx")


def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s))


def canonical_path(doc: dict) -> str | None:
    p = os.path.join(ROOT, doc["canonical_path"])
    if doc["type"] == "doc":  # 用转换后的 docx
        p = os.path.join(CONV, doc["doc_id"] + ".docx")
    return p if os.path.isfile(p) else None


def extract_word(path: str) -> str:
    d = Docx(path)
    parts = [p.text for p in d.paragraphs]
    for t in d.tables:  # 表格文本也拼入
        for row in t.rows:
            parts.append(" ".join(c.text for c in row.cells))
    return "\n".join(parts)


def extract_pdf(path: str) -> str:
    parts, n_empty_pages = [], 0
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            txt = page.extract_text() or ""
            if not txt.strip():
                n_empty_pages += 1
            parts.append(txt)
    return "\n".join(parts), n_empty_pages


def ngram_coverage(evidence: str, source: str, n: int = 4) -> float:
    """证据句的 n-gram 在源文本中的覆盖率(容忍改写/标点差异的字面充分性)。"""
    ev = norm(evidence)
    if len(ev) < n:
        return 1.0 if ev and ev in source else 0.0
    grams = [ev[i:i + n] for i in range(len(ev) - n + 1)]
    hit = sum(1 for g in grams if g in source)
    return hit / len(grams)


def main() -> int:
    docs = {d["doc_id"]: d for d in json.load(open(DOC_INDEX, encoding="utf-8"))["docs"]}
    qa = [json.loads(l) for l in open(QA, encoding="utf-8")]
    text_qa = [q for q in qa if q["source_type"] in ("word", "pdf") and q["split"] == "dev"]

    cache: dict[str, str] = {}
    stat: dict[str, dict] = {}
    empty_files: set[str] = set()
    low_files: dict[str, int] = {}
    TH = 0.7  # 覆盖率阈值:n-gram 覆盖 >= 0.7 视为该题证据可从抽取文本支撑
    for q in text_qa:
        st = q["source_type"]
        did = q["doc_id"]
        if did not in cache:
            path = canonical_path(docs[did])
            if path is None:
                cache[did] = ""
                empty_files.add(did)
                continue
            try:
                if st == "word":
                    cache[did] = norm(extract_word(path))
                else:
                    txt, _ = extract_pdf(path)
                    cache[did] = norm(txt)
            except Exception as e:  # noqa: BLE001
                print(f"  解析异常 {did}: {type(e).__name__}: {e}")
                cache[did] = ""
        s = stat.setdefault(st, {"total": 0, "hit": 0, "cov_sum": 0.0})
        s["total"] += 1
        cov = ngram_coverage(q["evidence"], cache[did]) if cache[did] else 0.0
        s["cov_sum"] += cov
        if cov >= TH:
            s["hit"] += 1
        elif cache[did]:
            low_files[did] = low_files.get(did, 0) + 1
        else:
            empty_files.add(did)

    print(f"dev 文本题: {len(text_qa)} 题, 涉及 {len(cache)} 个文件")
    print(f"判定标准: 证据句4-gram覆盖率 ≥ {TH} 视为抽取充分")
    for st, s in sorted(stat.items()):
        rate = s["hit"] / s["total"] * 100 if s["total"] else 0
        avg = s["cov_sum"] / s["total"] if s["total"] else 0
        print(f"  {st:<5} 充分 {s['hit']}/{s['total']} = {rate:.1f}%   平均覆盖率 {avg:.3f}")
    print(f"\n抽取为空/异常的文件({len(empty_files)}): {sorted(empty_files)}")
    if low_files:
        print("有文本但证据覆盖不足的文件 top:")
        for did, n in sorted(low_files.items(), key=lambda kv: -kv[1])[:10]:
            print(f"  {did} 缺{n}题: {docs[did]['orig_name'][:56]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
