#!/usr/bin/env python3
"""M2-4: BGE-M3 语义向量索引。

对 4227 个条款块用 BGE-M3(RTX 4060)编码,存 numpy npz
(block_id 顺序与 text_blocks.jsonl 一致)。

用法: python3 scripts/build_vector_index.py
"""
import json
import os
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCKS = os.path.join(ROOT, "data", "processed", "text_blocks.jsonl")
OUT = os.path.join(ROOT, "data", "processed", "block_vectors_bge_m3.npz")


def main() -> int:
    blocks = [json.loads(l) for l in open(BLOCKS, encoding="utf-8")]
    texts = [b["text"].replace("\n", " ") for b in blocks]
    print(f"编码 {len(texts)} 块 (BGE-M3 @ GPU)...")
    model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    emb = model.encode(texts, batch_size=32, show_progress_bar=True,
                       normalize_embeddings=True)
    np.savez_compressed(OUT, ids=np.array([b["block_id"] for b in blocks]),
                        vectors=emb.astype(np.float32))
    print(f"向量矩阵 {emb.shape} → {os.path.relpath(OUT, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
