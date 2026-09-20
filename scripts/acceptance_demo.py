#!/usr/bin/env python3
"""问答演示 + 敏感信息扫描。

用法:
  python3 scripts/acceptance_demo.py            # 随机抽文本题+表格题各1道演示
  python3 scripts/acceptance_demo.py --scan     # 仓库密钥泄漏扫描
"""
import os
import random
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_ROOTS = ["src", "scripts", "configs/llm.yaml", "README.md", "requirements.txt"]
SECRET_PATTERNS = [
    (r"sk-[A-Za-z0-9\-_.]{20,}", "疑似 API Key(sk-…)"),
    (r"api[_-]?key\s*[:=]\s*['\"][^'\"]{8,}", "疑似 api_key 字面值"),
    (r"(token|password|passwd|secret)\s*[:=]\s*['\"][^'\"]{8,}", "疑似口令字面值"),
]
ALLOW = {"configs/secrets.yaml"}  # 密钥唯一存放处(gitignore,不打包)


def demo():
    import json
    sys.path.insert(0, os.path.join(ROOT, "src"))
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    from llm_client import LLMClient
    from answer_question import answer, render
    from run_baseline import TableQA, TextRetriever
    blocks = [json.loads(l) for l in open(
        os.path.join(ROOT, "data/processed/text_blocks.jsonl"), encoding="utf-8")]
    tr, tq, cli = TextRetriever(blocks), TableQA(), LLMClient()
    qa = [json.loads(l) for l in open(
        os.path.join(ROOT, "data/eval/qa_dataset.jsonl"), encoding="utf-8")]
    text_q = random.choice([q for q in qa if q["split"] == "dev"
                            and q["source_type"] != "excel"])
    table_q = random.choice([q for q in qa if q["split"] == "dev"
                             and q["source_type"] == "excel"])
    for q in (text_q, table_q):
        print(f"\n问> {q['question']}")
        print(render(answer(q["question"], cli, tr, tq)))


def scan() -> int:
    bad = 0
    for root in SCAN_ROOTS:
        paths = []
        base = os.path.join(ROOT, root)
        if os.path.isfile(base):
            paths = [base]
        else:
            for dp, _, fns in os.walk(base):
                paths += [os.path.join(dp, f) for f in fns
                          if f.endswith((".py", ".md", ".yaml", ".txt", ".jsonl", ".json"))]
        for p in paths:
            rel = os.path.relpath(p, ROOT)
            if rel in ALLOW:
                continue
            try:
                text = open(p, encoding="utf-8").read()
            except Exception:
                continue
            for pat, label in SECRET_PATTERNS:
                for m in re.finditer(pat, text, re.I):
                    bad += 1
                    print(f"[命中] {rel}: {label} → {m.group(0)[:60]}")
    print(f"\n扫描完成: {'发现 ' + str(bad) + ' 处疑似敏感信息,清除后打包!' if bad else '通过,未发现敏感信息'}")
    return 1 if bad else 0


if __name__ == "__main__":
    if "--scan" in sys.argv:
        sys.exit(scan())
    demo()
