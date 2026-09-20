#!/usr/bin/env python3
"""M2-2: 文本路匹配策略变体实验(dev,不接模型)。

变体:
  V0 基线:K=10,选项对块的 4-gram 覆盖最大值
  V1:K=20
  V2:K=20 + 选项作为 BM25 查询的得分融合
  V3:K=20 + 选项分句(;/；。)后分句覆盖率取均值
  V4:K=20 + 分句 + BM25 融合

用法: python3 scripts/variant_text.py
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
QA = os.path.join(ROOT, "data", "eval", "qa_dataset.jsonl")
OPT_KEYS = {"A": "option_a", "B": "option_b", "C": "option_c", "D": "option_d"}


def norm(s):
    return re.sub(r"\s+", "", str(s))


def tok(s):
    return [w for w in jieba.lcut(str(s)) if w.strip()]


def ngram_cov(a, b, n=4):
    a = norm(a)
    bl = norm(b)
    if len(a) < n:
        return 1.0 if a and a in bl else 0.0
    gs = [a[i:i + n] for i in range(len(a) - n + 1)]
    return sum(1 for g in gs if g in bl) / len(gs)


def substatements(opt):
    parts = [p for p in re.split(r"[;；。]", str(opt)) if p.strip()]
    return parts or [str(opt)]


def main():
    blocks = [json.loads(l) for l in open(BLOCKS, encoding="utf-8")]
    corpus = [tok(b["text"]) for b in blocks]
    bm25 = BM25Okapi(corpus)
    qa = [json.loads(l) for l in open(QA, encoding="utf-8")
          if json.loads(l)["split"] == "dev" and json.loads(l)["source_type"] in ("word", "pdf")]

    def retrieve(query, k):
        s = bm25.get_scores(tok(query))
        order = sorted(range(len(s)), key=lambda i: -s[i])[:k]
        mx = s[order[0]] if order and s[order[0]] > 0 else 1
        return [(blocks[i], s[i] / mx) for i in order if s[i] > 0]

    def answer(q, K, fuse_bm25, use_sub):
        hits = retrieve(q["question"], K)
        if not hits:
            return None
        # 选项自身作 BM25 查询的归一化得分
        opt_bm = {}
        for letter, key in OPT_KEYS.items():
            sc = bm25.get_scores(tok(q[key]))
            opt_bm[letter] = (max(sc) / max(sc.max(), 1)) if hasattr(sc, "max") else 0.0
        best, best_score = None, -1
        for letter, key in OPT_KEYS.items():
            opt = q[key]
            if use_sub:
                parts = substatements(opt)
                cov = sum(max((ngram_cov(p, b["text"]) for b, _ in hits), default=0)
                          for p in parts) / len(parts)
            else:
                cov = max((ngram_cov(opt, b["text"]) for b, _ in hits), default=0)
            score = cov
            if fuse_bm25:
                score = 0.75 * cov + 0.25 * opt_bm[letter]
            if score > best_score:
                best, best_score = letter, score
        return best

    variants = {"V0": (10, False, False), "V1": (20, False, False),
                "V2": (20, True, False), "V3": (20, False, True), "V4": (20, True, True)}
    for name, (K, fuse, sub) in variants.items():
        right, n = 0, 0
        by_t = defaultdict(lambda: [0, 0])
        for q in qa:
            pred = answer(q, K, fuse, sub)
            c = (pred == q["answer"])
            right += c
            n += 1
            by_t[q["qa_type"]][0] += c
            by_t[q["qa_type"]][1] += 1
        det = " ".join(f"{t}:{c}/{k}" for t, (c, k) in sorted(by_t.items()))
        print(f"{name} (K={K}, fuse_bm25={fuse}, 分句={sub}): {right}/{n} = {right/n*100:.1f}%  [{det}]")


if __name__ == "__main__":
    main()
