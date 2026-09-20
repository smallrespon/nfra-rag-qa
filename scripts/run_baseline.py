#!/usr/bin/env python3
"""M1-3: 基线系统——BM25 文本检索 + 规则式表格取数,dev 集自动评测。

文本路(单事实/多事实):BM25(jieba)按问题召回 Top-K 块,
  选项得分 = 选项与各块的 4-gram 覆盖率最大值,取最高选项作答。
表格路(取数/比较/计算):从问题解析《文件名》/工作表/引号指标与口径,
  在该文档的行级记录中定位指标×口径,数值与选项做容差匹配;
  比较/计算题先取多值再做确定性差/和运算。

输出: data/eval/predictions_baseline.jsonl, data/eval/errors/baseline_dev.jsonl,
      data/eval/baseline_dev_summary.json
用法: python3 scripts/run_baseline.py [--split dev]
"""
import json
import os
import re
import sys
from collections import defaultdict

import jieba
from rank_bm25 import BM25Okapi

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCKS = os.path.join(ROOT, "data", "processed", "text_blocks.jsonl")
RECORDS = os.path.join(ROOT, "data", "processed", "table_records.jsonl")
DOC_INDEX = os.path.join(ROOT, "data", "processed", "doc_index.json")
QA = os.path.join(ROOT, "data", "eval", "qa_dataset.jsonl")
ERR_DIR = os.path.join(ROOT, "data", "eval", "errors")

OPT_KEYS = {"A": "option_a", "B": "option_b", "C": "option_c", "D": "option_d"}


def clean_title(title: str) -> str:
    """剥离题面《》中的格式标注(如《xxx(PDF)》),避免匹配失败。"""
    return re.sub(r"[（(](pdf|word|excel|doc|docx|xls|xlsx)[)）]", "",
                  norm(title), flags=re.I)


def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s))


def ngram_cov(a: str, b: str, n: int = 4) -> float:
    a = norm(a)
    if len(a) < n:
        return 1.0 if a and a in norm(b) else 0.0
    gs = [a[i:i + n] for i in range(len(a) - n + 1)]
    return sum(1 for g in gs if g in norm(b)) / len(gs)


def tok(s: str):
    return [w for w in jieba.lcut(str(s)) if w.strip()]


# ---------- 文本路 ----------

class TextRetriever:
    VEC_FILE = os.path.join(ROOT, "data", "processed", "block_vectors_bge_m3.npz")

    def __init__(self, blocks):
        self.blocks = blocks
        self.ntext = [norm(b["text"]) for b in blocks]
        self.by_doc = defaultdict(list)
        for b, nt in zip(blocks, self.ntext):
            self.by_doc[b["doc_id"]].append((b, nt))
        self.corpus = [tok(b["text"]) for b in blocks]
        self.bm25 = BM25Okapi(self.corpus)
        docs = json.load(open(DOC_INDEX, encoding="utf-8"))["docs"]
        self.doc_stems = []
        for d in docs:
            stem = norm(re.sub(r"^\d+_", "", re.sub(r"\.[a-zA-Z0-9]+$", "", d["orig_name"])))
            self.doc_stems.append((len(stem), d["doc_id"], stem))
        # M2-4: BGE-M3 向量(可选,存在则混合召回)
        self.emb_model, self.vecs, self.reranker = None, None, None
        if os.path.isfile(self.VEC_FILE):
            import numpy as np
            data = np.load(self.VEC_FILE)
            assert list(data["ids"]) == [b["block_id"] for b in blocks], "向量与块序不一致"
            self.vecs = data["vectors"]

    def _embed(self, texts):
        if self.emb_model is None:
            from sentence_transformers import SentenceTransformer
            self.emb_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
        return self.emb_model.encode(texts, normalize_embeddings=True,
                                     show_progress_bar=False)

    def locate_doc(self, title: str):
        """题面《材料名》→ doc_id(任意类型,取含该名的最短 stem)。"""
        t = clean_title(title)
        cands = [(ln, did) for ln, did, stem in self.doc_stems if t in stem]
        return min(cands)[1] if cands else None

    def topk_hybrid(self, query: str, k: int = 20, rerank: bool = False):
        """BM25 + BGE-M3 余弦的 RRF 融合召回;可选 bge-reranker 精排;
        无向量时退回纯 BM25。"""
        bm_scores = self.bm25.get_scores(tok(query))
        bm_order = sorted(range(len(bm_scores)), key=lambda i: -bm_scores[i])
        if self.vecs is None:
            return [(self.blocks[i], bm_scores[i]) for i in bm_order[:k] if bm_scores[i] > 0]
        import numpy as np
        qv = np.asarray(self._embed([query.replace("\n", " ")]), dtype=np.float32)
        dn_scores = self.vecs @ qv[0]
        dn_order = sorted(range(len(dn_scores)), key=lambda i: -dn_scores[i])
        RRF_K = 60
        fuse = {}
        for rank, i in enumerate(bm_order[:50]):
            fuse[i] = fuse.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, i in enumerate(dn_order[:50]):
            fuse[i] = fuse.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        fused = sorted(fuse.items(), key=lambda kv: -kv[1])
        if rerank:
            pairs = [(query.replace("\n", " "), self.blocks[i]["text"].replace("\n", " "))
                     for i, _ in fused[:50]]
            scores = self._rerank(pairs)
            reranked = sorted(zip([i for i, _ in fused[:50]], scores), key=lambda t: -t[1])
            return [(self.blocks[i], s) for i, s in reranked[:k]]
        return [(self.blocks[i], s) for i, s in fused[:k]]

    def _rerank(self, pairs):
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device="cuda",
                                         max_length=512)
        return self.reranker.predict(pairs, show_progress_bar=False)

    def answer(self, q, rerank: bool = False):
        # 优先:题面《材料名》圈定文档,选项(分句)对全文档块做覆盖打分
        m = re.search(r"《([^》]+)》", q["question"])
        if m:
            doc_id = self.locate_doc(m.group(1))
            cands = self.by_doc.get(doc_id, []) if doc_id else []
            if cands:
                best, best_score, ev_blocks = None, -1.0, []
                for letter, key in OPT_KEYS.items():
                    parts = [p for p in re.split(r"[;；。]", str(q[key])) if norm(p)] or [q[key]]
                    sc = sum(max((ngram_cov(p, nt) for _, nt in cands), default=0.0)
                             for p in parts) / len(parts)
                    if sc > best_score:
                        best_score = sc
                        best = letter
                        ev_blocks = []
                        for p in parts:  # 每分句取 top-2 匹配块,合并为证据集
                            pnt = norm(p)
                            ranked = sorted(cands, key=lambda t: -ngram_cov(pnt, t[1]))
                            for b, _ in ranked[:2]:
                                if b not in ev_blocks:
                                    ev_blocks.append(b)
                        ev_blocks = ev_blocks[:6]
                ev_list = [{"doc_id": b["doc_id"], "location": b["location"],
                            "orig_name": b["orig_name"]} for b in ev_blocks[:4]]
                return best, {"reason": "doc_scoped_coverage", "doc_id": doc_id,
                              "score": round(best_score, 3), "evidence": ev_list}
        # 回退:BM25+向量混合召回 Top-20 + 覆盖率
        hits = self.topk_hybrid(q["question"], k=20, rerank=rerank)
        if not hits:
            return "未找到足够证据", {"evidence": None, "reason": "no_hit"}
        best_opt, best_cov, best_block = None, -1.0, None
        for letter, key in OPT_KEYS.items():
            opt = q[key]
            for blk, _ in hits:
                cov = ngram_cov(opt, blk["text"])
                if cov > best_cov:
                    best_cov, best_opt, best_block = cov, letter, blk
        ev = {"doc_id": best_block["doc_id"], "location": best_block["location"],
              "orig_name": best_block["orig_name"], "match": round(best_cov, 3)}
        return best_opt, {"evidence": ev, "reason": "hybrid+coverage"}


# ---------- 表格路 ----------

class TableQA:
    def __init__(self):
        self.by_doc = defaultdict(list)   # doc_id -> [records(dict)]
        with open(RECORDS, encoding="utf-8") as fp:
            for line in fp:
                r = json.loads(line)
                self.by_doc[r["doc_id"]].append(r)
        docs = json.load(open(DOC_INDEX, encoding="utf-8"))["docs"]
        self.doc_stems = []
        for d in docs:
            if d["type"] not in ("xls", "xlsx"):
                continue
            stem = norm(re.sub(r"^\d+_", "", re.sub(r"\.[a-zA-Z0-9]+$", "", d["orig_name"])))
            self.doc_stems.append((len(stem), d["doc_id"], stem))

    def locate_docs(self, title: str):
        """按《文件名》返回全部 excel 候选(按 stem 长度升序)。"""
        t = clean_title(title)
        cands = sorted(((ln, did) for ln, did, stem in self.doc_stems if t in stem))
        return [did for _, did in cands]

    def lookup(self, doc_id, term, sheet=None):
        """term 匹配指标(优先)或列口径(次之);返回 [(rec, kind)]。"""
        t = norm(term)
        recs = self.by_doc.get(doc_id, [])
        if sheet:
            sn = norm(sheet)
            recs = [r for r in recs if sn in norm(r["sheet"]) or norm(r["sheet"]) in sn]
        hits = [(r, "indicator") for r in recs
                if t == norm(r["indicator"]) or t in norm(r["indicator_path"])]
        if not hits:
            hits = [(r, "column") for r in recs if t == norm(r["column_label"])]
        # 精确指标命中优先,其次包含
        exact = [h for h in hits if h[1] == "indicator" and t == norm(h[0]["indicator"])]
        return exact or hits

    def parse_question(self, q):
        text = q["question"]
        title = re.search(r"《([^》]+)》", text)
        sheet = re.search(r"工作表\s*[:：]\s*(\S+)", text)
        quoted = re.findall(r"[“\"]([^”\"]{2,30})[”\"]", text)
        s = sheet.group(1) if sheet else None
        if s:  # 只剥多余的标点与不配对的右括号
            s = s.rstrip("，。;；、.")
            while s.endswith("）") and s.count("）") > s.count("（"):
                s = s[:-1]
            while s.endswith(")") and s.count(")") > s.count("("):
                s = s[:-1]
        return (title.group(1) if title else None, s, quoted)

    def fetch_value(self, doc_id, term, sheet=None, exclude=None):
        hits = self.lookup(doc_id, term, sheet)
        if exclude:  # 比较题中两个词,避免取到同一条记录
            hits = [h for h in hits if (h[0]["row"], h[0]["col"]) != exclude]
        if not hits:
            return None, None
        rec, kind = hits[0]
        return rec, kind

    @staticmethod
    def _opt_matches(value, option_text) -> bool:
        m = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(option_text).replace(",", ""))
        if not m:
            return False
        v = float(m.group())
        return abs(v - value) / max(1.0, abs(value)) < 0.02

    @staticmethod
    def match_numeric(value, options):
        """value ↔ 选项数值容差匹配;返回最优选项字母或 None。"""
        best, diff = None, float("inf")
        for letter, key in OPT_KEYS.items():
            m = re.search(r"-?\d[\d,]*(?:\.\d+)?", str(options[key]).replace(",", ""))
            if not m:
                continue
            v = float(m.group())
            d = abs(v - value) / max(1.0, abs(value))
            if d < diff:
                best, diff = letter, d
        return best if diff < 0.02 else None

    def answer(self, q):
        title, sheet, quoted = self.parse_question(q)
        if not title:
            return None, {"reason": "no_title", "evidence": None}
        # 候选文档迭代 + 内容消歧:首个能成功求解的候选胜出
        cands = self.locate_docs(title)
        first_fail = None
        for doc_id in cands:
            letter, info = self._solve(q, doc_id, sheet, quoted)
            if letter is not None:
                return letter, info
            if first_fail is None:
                first_fail = (letter, info)
        if first_fail:
            return first_fail
        return None, {"reason": "doc_not_found", "evidence": None}

    def _solve(self, q, doc_id, sheet, quoted):
        """在单一候选文档上尝试求解;失败返回 (None, info)。"""
        if q["qa_type"] == "表格取数":
            # 引号词里:指标 = 匹配到 indicator 的;口径 = 匹配到 column_label 的
            vals = [(t, *self.fetch_value(doc_id, t, sheet)) for t in quoted]
            ind_hits = [v for v in vals if v[1] is not None and v[2] == "indicator"]
            if not ind_hits:
                return None, {"reason": "indicator_not_found",
                              "evidence": None, "quoted": quoted}
            rec = ind_hits[0][1]
            value = rec["value"]
            letter = self.match_numeric(value, q)
            ev = {"doc_id": doc_id, "sheet": rec["sheet"], "cell": f"{chr(64+rec['col'])}{rec['row']}",
                  "indicator": rec["indicator"], "column_label": rec["column_label"],
                  "value": value, "unit": rec["unit"]}
            return letter, {"reason": "table_lookup", "evidence": ev}

        if q["qa_type"] == "表格比较":
            # 选项=指标名:在题面口径下逐选项取值,答数值最高者
            calibers = [norm(t) for t in quoted] or []
            scored = []
            for letter, key in OPT_KEYS.items():
                hits = [r for r, k in self.lookup(doc_id, q[key], sheet) if k == "indicator"]
                if calibers:
                    f = [r for r in hits if any(c in norm(r["column_label"]) for c in calibers)]
                    hits = f or hits
                if hits:
                    scored.append((letter, hits[0]["value"], hits[0]))
            if not scored:
                return None, {"reason": "compare_no_values", "evidence": None}
            # 存量(总资产/余额)与保额类指标不参与保费类流量比较(金融语义:不同维度不可比)
            EXCLUDE_RE = re.compile(r"总资产|总负债|资产总额|负债总额|净资产|余额|"
                                    r"保险金额|保额|件数|机构数|人员数")
            core = [t for t in scored if not EXCLUDE_RE.search(t[2]["indicator"])]
            pool = core or scored
            letter, value, rec = max(pool, key=lambda t: t[1])
            ev = {"doc_id": doc_id, "sheet": rec["sheet"],
                  "values": {l: round(v, 2) for l, v, _ in scored},
                  "excluded": [t[0] for t in scored if t not in pool],
                  "cell": f"{chr(64+rec['col'])}{rec['row']}", "indicator": rec["indicator"]}
            return letter, {"reason": "table_compare_max", "evidence": ev}

        if q["qa_type"] == "表格计算":
            # 模板:「行指标」从「列1」到「列2」的数值变化 = v(列2) - v(列1),带符号
            m = re.search(r"[“\"]([^”\"]+)[”\"]从[“\"]([^”\"]+)[”\"]到[“\"]([^”\"]+)[”\"]",
                          q["question"])
            if m:
                row_ind, col1, col2 = m.group(1), m.group(2), m.group(3)
                rows = [r for r, k in self.lookup(doc_id, row_ind, sheet) if k == "indicator"]
                if rows:
                    target = rows[0]
                    same_row = [r for r in self.by_doc.get(doc_id, [])
                                if r["sheet"] == target["sheet"] and r["row"] == target["row"]]
                    v1 = next((r["value"] for r in same_row if col1 in r["column_label"]), None)
                    v2 = next((r["value"] for r in same_row if col2 in r["column_label"]), None)
                    if v1 is not None and v2 is not None:
                        diff = v2 - v1
                        letter = self.match_numeric(diff, q)
                        ev = {"doc_id": doc_id, "sheet": target["sheet"],
                              "row": target["row"],
                              "cells": [f"{chr(64+target['col'])}{target['row']}",
                                        f"列{col1}={round(v1,2)}", f"列{col2}={round(v2,2)}"],
                              "diff": round(diff, 2)}
                        if letter:
                            return letter, {"reason": "table_calc_row", "evidence": ev}
                # 退化:期间引用不可解(模板伪影如"从'年-季度'到'季度'")
                # → 该指标行首→末列差值(带符号),要求选项唯一匹配才作答
                rows = [r for r, k in self.lookup(doc_id, row_ind, sheet) if k == "indicator"]
                for target in rows:
                    same_row = sorted((r for r in self.by_doc.get(doc_id, [])
                                       if r["sheet"] == target["sheet"]
                                       and r["row"] == target["row"]), key=lambda r: r["col"])
                    if len(same_row) < 2:
                        continue
                    diff = same_row[-1]["value"] - same_row[0]["value"]
                    hits = [L for L in OPT_KEYS
                            if self._opt_matches(diff, q[OPT_KEYS[L]])]
                    if len(hits) == 1:
                        ev = {"doc_id": doc_id, "sheet": target["sheet"],
                              "row": target["row"],
                              "cells": [f"{chr(64+r['col'])}{r['row']}={round(r['value'],2)}"
                                        for r in (same_row[0], same_row[-1])],
                              "diff": round(diff, 2), "degraded": True}
                        return hits[0], {"reason": "table_calc_span", "evidence": ev}
            # 回退:两个引号词取值做差/和
            pairs = []
            for t in quoted:
                rec, kind = self.fetch_value(doc_id, t, sheet)
                if rec is not None:
                    pairs.append((t, rec))
            if len(pairs) >= 2:
                a, b = pairs[0][1]["value"], pairs[1][1]["value"]
                for v in (a - b, b - a, a + b):
                    letter = self.match_numeric(v, q)
                    if letter:
                        ev = {"doc_id": doc_id,
                              "values": [(p[0], round(p[1]["value"], 2)) for p in pairs],
                              "calc": round(v, 2)}
                        return letter, {"reason": "table_calc_fallback", "evidence": ev}
                return None, {"reason": "calc_no_match",
                              "evidence": {"values": [(p[0], round(p[1]["value"], 2)) for p in pairs]}}
            return None, {"reason": "calc_need_two_values", "evidence": None}
        return None, {"reason": "unknown_qa_type", "evidence": None}


def main() -> int:
    split = "dev"
    if "--split" in sys.argv:
        split = sys.argv[sys.argv.index("--split") + 1]
    rerank = "--rerank" in sys.argv

    blocks = [json.loads(l) for l in open(BLOCKS, encoding="utf-8")]
    print(f"加载文本块 {len(blocks)} ...")
    tr = TextRetriever(blocks)
    tq = TableQA()
    qa = [json.loads(l) for l in open(QA, encoding="utf-8") if json.loads(l)["split"] == split]

    preds, n_right = [], 0
    by_type, by_diff = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    os.makedirs(ERR_DIR, exist_ok=True)
    err_path = os.path.join(ERR_DIR, f"baseline_{split}.jsonl")
    with open(err_path, "w", encoding="utf-8") as ef:
        for q in qa:
            try:
                if q["source_type"] == "excel":
                    letter, info = tq.answer(q)
                    if letter is None:  # 表格路失败 → 文本路兜底
                        letter, tinfo = tr.answer(q, rerank=rerank)
                        info = {"fallback_text": tinfo, **info}
                else:
                    letter, info = tr.answer(q, rerank=rerank)
            except Exception as e:  # noqa: BLE001
                letter, info = None, {"reason": f"exception:{type(e).__name__}", "evidence": None}
            correct = (letter == q["answer"])
            n_right += correct
            by_type[q["qa_type"]][0] += correct
            by_type[q["qa_type"]][1] += 1
            by_diff[q["difficulty_cn"]][0] += correct
            by_diff[q["difficulty_cn"]][1] += 1
            pred = {"id": q["id"], "pred": letter, "gold": q["answer"],
                    "correct": correct, "meta": info}
            preds.append(pred)
            if not correct:
                ef.write(json.dumps({**pred, "question": q["question"],
                                     "answer_text": q["answer_text"],
                                     "options": {k: q[v] for k, v in OPT_KEYS.items()}},
                                    ensure_ascii=False) + "\n")

    n = len(qa)
    summary = {
        "split": split, "total": n, "correct": n_right,
        "accuracy": round(n_right / n * 100, 1) if n else 0,
        "by_qa_type": {t: {"correct": c, "n": k, "acc": round(c / k * 100, 1)}
                       for t, (c, k) in sorted(by_type.items())},
        "by_difficulty": {t: {"correct": c, "n": k, "acc": round(c / k * 100, 1)}
                          for t, (c, k) in sorted(by_diff.items())},
    }
    out = os.path.join(ROOT, "data", "eval", f"baseline_{split}_summary.json")
    json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    with open(os.path.join(ROOT, "data", "eval", "predictions_baseline.jsonl"), "w",
              encoding="utf-8") as fp:
        for p in preds:
            fp.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=1))
    print(f"\n明细: {os.path.relpath(err_path, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
