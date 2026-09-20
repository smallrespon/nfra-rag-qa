#!/usr/bin/env python3
"""M3-4: S3 证据比对器——系统证据 vs QA 标准证据的命中率。

口径:
  文件级命中: 系统证据的 doc_id == 标准 doc_id;
  内容级命中(文本题): 标准 evidence 句(4-gram 覆盖率口径)能在系统返回的证据块中
    找回(≥0.7);
  内容级命中(表格题): 系统证据含标准 evidence 的"单元格:XX"或原始值。

用法: python3 scripts/s3_hitrate.py data/eval/predictions_baseline.jsonl
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA = os.path.join(ROOT, "data", "eval", "qa_dataset.jsonl")


def norm(s):
    return re.sub(r"\s+", "", str(s))


def ngram_cov(a, b, n=4):
    a = norm(a)
    b = norm(b)
    if len(a) < n:
        return 1.0 if a and a in b else 0.0
    gs = [a[i:i + n] for i in range(len(a) - n + 1)]
    return sum(1 for g in gs if g in b) / len(gs)


COS_THRESHOLD = 0.60
_cos_model, _cos_cache = None, {}


def cos_sim(a: str, b: str) -> float:
    global _cos_model
    import numpy as np
    key = (a[:80], b[:80])
    if key in _cos_cache:
        return _cos_cache[key]
    if _cos_model is None:
        from sentence_transformers import SentenceTransformer
        _cos_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    va, vb = _cos_model.encode([a[:512], b[:512]], normalize_embeddings=True,
                               show_progress_bar=False)
    v = float(va @ vb)
    _cos_cache[key] = v
    return v


def hit_stmt(stmt: str, block_text: str) -> bool:
    return ngram_cov(stmt, block_text) >= 0.7 or cos_sim(stmt, block_text) >= COS_THRESHOLD


def system_evidence(pred: dict, qa: dict):
    """从预测记录抽取系统证据:返回 (doc_id, block_text, cell)。"""
    meta = pred.get("meta", {})
    ev = meta.get("evidence") or (meta.get("fallback_text") or {}).get("evidence")
    doc_id = meta.get("doc_id") or (ev or {}).get("doc_id")
    if qa["source_type"] == "excel":
        cell = (ev or {}).get("cell")
        value = (ev or {}).get("value")
        return doc_id, None, cell, value
    # 文本题:doc-scoped 有块引用;hybrid 有块引用;否则仅 doc
    if meta.get("reason") == "doc_scoped_coverage":
        return doc_id, ev, None, None
    return doc_id, ev, None, None


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else \
        os.path.join(ROOT, "data", "eval", "predictions_baseline.jsonl")
    preds = {p["id"]: p for p in
             (json.loads(l) for l in open(path, encoding="utf-8"))}
    qa = {q["id"]: q for q in (json.loads(l) for l in open(QA, encoding="utf-8"))}
    blocks = {b["block_id"]: b for b in
              (json.loads(l) for l in open(
                  os.path.join(ROOT, "data", "processed", "text_blocks.jsonl"),
                  encoding="utf-8"))}

    from collections import defaultdict
    stat = defaultdict(lambda: {"n": 0, "file": 0, "content": 0})
    blk_by_loc = {(b["doc_id"], b["location"]): b for b in blocks.values()}
    for qid, p in preds.items():
        q = qa[qid]
        doc_id, ev, cell, value = system_evidence(p, q)
        s = stat[q["qa_type"] if q["source_type"] != "excel" else "表格类"]
        s["n"] += 1
        if doc_id == q["doc_id"]:
            s["file"] += 1
        if q["source_type"] == "excel":
            evn = norm(str(q["evidence"]))
            # 两种格式:取数题"单元格：C5"/原始值：x;比较计算题"指标=数值(C6)"
            gold_cells = set(re.findall(r"单元格[：:]\s*([A-Z]+\d+)", evn)) | \
                set(re.findall(r"[（(]([A-Z]+\d{1,3})[)）]", evn))
            gold_vals = set(re.findall(r"原始值[：:]\s*(-?[\d.]+)", evn)) | \
                set(re.findall(r"=(-?\d[\d.]*)[（(]", evn))
            ev_json = json.dumps(p.get("meta", {}).get("evidence") or {}, ensure_ascii=False)
            sys_cells = ({cell} if cell else set()) | set(re.findall(r"[A-Z]+\d{1,3}", ev_json))
            sys_vals = set(re.findall(r'"[^"]*":\s*(-?\d[\d.]*)', ev_json)) | \
                set(re.findall(r'"diff":\s*(-?\d[\d.]*)', ev_json))
            ok = bool(gold_cells & sys_cells) or \
                any(any(abs(float(g) - float(v)) < 0.01 for v in sys_vals)
                    for g in gold_vals if re.match(r"^-?\d", g))
            s["content"] += bool(ok)
        else:
            # 证据集:doc-scoped 为列表,hybrid 为单块;金句每条陈述都需可找回。
            # 双口径:字面 4-gram≥0.7 或 BGE-M3 语义余弦≥0.6(容忍改写型证据)
            ev_list = ev if isinstance(ev, list) else ([ev] if ev else [])
            sys_blocks = [blk_by_loc[(e["doc_id"], e["location"])]
                          for e in ev_list if e.get("location")
                          and (e["doc_id"], e["location"]) in blk_by_loc]
            if sys_blocks:
                stmts = [x for x in re.split(r"[;；]", norm(str(q["evidence"]))) if len(x) >= 8]
                covered = all(any(hit_stmt(st, b["text"]) for b in sys_blocks)
                              for st in stmts) if stmts else False
                s["content"] += bool(covered)
            elif doc_id == q["doc_id"]:
                # 无块引用时以文件命中近似内容层下界
                pass

    n_all = sum(s["n"] for s in stat.values())
    f_all = sum(s["file"] for s in stat.values())
    c_all = sum(s["content"] for s in stat.values())
    print(f"{'题型':<8}{'n':>5}{'文件级':>9}{'内容级':>9}")
    for t, s in sorted(stat.items()):
        print(f"{t:<8}{s['n']:>5}{s['file']/s['n']*100:>8.1f}%{s['content']/s['n']*100:>8.1f}%")
    print(f"{'总体':<8}{n_all:>5}{f_all/n_all*100:>8.1f}%{c_all/n_all*100:>8.1f}%  (目标:文件级≥90%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
