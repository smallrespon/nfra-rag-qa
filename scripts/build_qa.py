#!/usr/bin/env python3
"""M0-2: 构建机读评测集 + 按源文件整组切分 dev/test。

1. 读取 QA数据.xlsx 的 300 题,规范化字段;
2. file_label → doc_id 映射(级联:精确子串 → 截断容错 → difflib 相似度兜底),
   修复"原文件名被截断"导致无法溯源的题目;
3. 按源文件整组切分 dev/test(同一 doc_id 的题绝不跨集,防同干题泄漏);
   目标 dev≈100。test 分割仅在正式评测时使用。
4. 落盘 data/eval/qa_dataset.jsonl 与 data/eval/split_summary.json。

用法: python3 scripts/build_qa.py
"""
import difflib
import json
import os
import random
import re
import sys

import openpyxl

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
QA_XLSX = os.path.join(ROOT, "data", "QA数据.xlsx")
DOC_INDEX = os.path.join(ROOT, "data", "processed", "doc_index.json")
OUT_JSONL = os.path.join(ROOT, "data", "eval", "qa_dataset.jsonl")
OUT_SUMMARY = os.path.join(ROOT, "data", "eval", "split_summary.json")

SHEET = "nfra_500_mcq_300_curated"
OPT_KEYS = {"A": "option_a", "B": "option_b", "C": "option_c", "D": "option_d"}


def norm(s: str) -> str:
    return re.sub(r"\s+", "", str(s))


def load_docs() -> list[dict]:
    with open(DOC_INDEX, encoding="utf-8") as fp:
        return json.load(fp)["docs"]


def build_matcher(docs: list[dict]):
    """预归一化所有文件的 orig_name 与 stem,供级联匹配。"""
    entries = []
    for d in docs:
        orig = norm(d["orig_name"])
        stem = re.sub(r"\.[A-Za-z0-9]+$", "", orig)  # 去扩展名
        entries.append({"doc": d, "orig": orig, "stem": stem})
    return entries


def lcs_len(a: str, b: str) -> int:
    """最长公共子串长度(difflib 块拼接)。"""
    m = difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    return m.size


def match_label(label: str, entries) -> dict | None:
    """级联匹配 file_label → 源文件。返回 doc 记录或 None。"""
    label_stem = norm(re.sub(r"\.[A-Za-z0-9]+$", "", label))  # 去扩展名后归一化
    label_ext = label.rsplit(".", 1)[-1].lower() if "." in label else ""

    def of_ext(cands):
        """优先同扩展名(同一文档的 docx/pdf 版本在库中并存)。"""
        same = [e for e in cands if e["doc"]["type"] == label_ext]
        return same or cands

    # 1) 子串包含:label stem 出现在某文件 stem 中;
    #    同扩展名优先,再优先 stem 以 label 结尾者(附件本体),再取最短 stem
    cands = of_ext([e for e in entries if label_stem in e["stem"]])
    if cands:
        suff = [e for e in cands if e["stem"].endswith(label_stem)] or cands
        return min(suff, key=lambda e: len(e["stem"]))["doc"]

    # 2) 截断容错:按最长公共子串(LCS)评分取最优,
    #    覆盖"原文件名在'（2023年'处被截断"等残缺场景
    scored = sorted(((lcs_len(label_stem, e["stem"]), e) for e in entries),
                    key=lambda t: -t[0])
    if scored and scored[0][0] >= 15:
        return scored[0][1]["doc"]
    return None


def group_split(qa: list[dict], ratio: float = 1 / 3, cap_margin: int = 15) -> None:
    """按 doc_id 整组、按 source_type 分层切分:
    每个来源类型内部,组按题数升序贪心填 dev(文件更多、更多样),
    dev 配额 ≈ 该来源题数×ratio;其余整组进 test。随后校验两集 qa_type 全覆盖。"""
    from collections import Counter, defaultdict

    groups: dict[str, list[dict]] = {}
    for q in qa:
        groups.setdefault(q["doc_id"], []).append(q)
    by_source = defaultdict(list)
    for g in groups.values():
        by_source[g[0]["source_type"]].append(g)

    quotas = {st: round(sum(len(g) for g in gs) * ratio)
              for st, gs in by_source.items()}
    for st, gs in by_source.items():
        gs.sort(key=lambda g: (len(g), g[0]["doc_id"]))
        n_dev = 0
        for g in gs:
            if n_dev < quotas[st] and n_dev + len(g) <= quotas[st] + cap_margin:
                for q in g:
                    q["split"] = "dev"
                n_dev += len(g)
            else:
                for q in g:
                    q["split"] = "test"

    # 覆盖性校验:任一 qa_type 只出现在单侧时,交换一个包含该题型的小组
    for _ in range(5):
        dev_types = {q["qa_type"] for q in qa if q["split"] == "dev"}
        test_types = {q["qa_type"] for q in qa if q["split"] == "test"}
        missing = {"dev": test_types - dev_types, "test": dev_types - test_types}
        if not missing["dev"] and not missing["test"]:
            return
        for side, types in missing.items():
            if not types:
                continue
            t = sorted(types)[0]
            donor = "test" if side == "dev" else "dev"
            cands = [g for g in groups.values()
                     if g[0]["split"] == donor and any(q["qa_type"] == t for q in g)]
            if cands:
                g = min(cands, key=len)
                for q in g:
                    q["split"] = side
                break


def main() -> int:
    wb = openpyxl.load_workbook(QA_XLSX, read_only=True, data_only=True)
    ws = wb[SHEET]
    rows = list(ws.iter_rows(values_only=True))
    headers = list(rows[0])
    qa = [dict(zip(headers, r)) for r in rows[1:] if r[0] is not None]
    assert len(qa) == 300, f"期望 300 题,实得 {len(qa)}"

    docs = load_docs()
    entries = build_matcher(docs)

    matched, unmatched, truncation_fixed = {}, [], []
    for q in qa:
        label = str(q["file_label"])
        doc = match_label(label, entries)
        if doc is None:
            unmatched.append((q["id"], label))
            continue
        matched.setdefault(label, doc)
        q["doc_id"] = doc["doc_id"]
        q["canonical_path"] = doc["canonical_path"]
        if not norm(re.sub(r"\.[A-Za-z0-9]+$", "", label)) in doc["orig_name"]:
            truncation_fixed.append((q["id"], label, doc["doc_id"]))

    assert not unmatched, f"仍有 {len(unmatched)} 题无法映射源文件: {unmatched[:5]}"

    # 选项一致性自检(与 M0 前核查结论呼应,防源数据变动)
    for q in qa:
        assert str(q[OPT_KEYS[str(q["answer"]).strip().upper()]]).strip() == \
            str(q["answer_text"]).strip(), f"{q['id']} answer/选项不一致"

    # 按源文件整组切分
    group_split(qa)

    # 泄漏自检断言
    dev_docs = {q["doc_id"] for q in qa if q["split"] == "dev"}
    test_docs = {q["doc_id"] for q in qa if q["split"] == "test"}
    assert not (dev_docs & test_docs), "泄漏:同一 doc_id 同时出现在 dev 与 test"

    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    with open(OUT_JSONL, "w", encoding="utf-8") as fp:
        for q in qa:
            fp.write(json.dumps(q, ensure_ascii=False) + "\n")

    # 切分摘要
    def cross(split):
        from collections import Counter
        sub = [q for q in qa if q["split"] == split]
        return {
            "questions": len(sub),
            "files": len({q["doc_id"] for q in sub}),
            "by_qa_type": dict(Counter(q["qa_type"] for q in sub)),
            "by_difficulty": dict(Counter(q["difficulty_cn"] for q in sub)),
            "by_source_type": dict(Counter(q["source_type"] for q in sub)),
        }

    summary = {
        "total": len(qa),
        "dev": cross("dev"),
        "test": cross("test"),
        "dev_doc_ids": sorted(dev_docs),
        "test_doc_ids": sorted(test_docs),
        "truncation_fixed": truncation_fixed,
        "note": "test 分割仅在正式评测时使用,日常调参只使用 dev",
    }
    with open(OUT_SUMMARY, "w", encoding="utf-8") as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=1)

    print(f"300/300 题全部映射到源文件;其中截断修复 {len(truncation_fixed)} 题")
    for t in truncation_fixed[:3]:
        print("   修复示例:", t)
    print(f"dev: {summary['dev']['questions']} 题 / {summary['dev']['files']} 文件")
    print(f"test: {summary['test']['questions']} 题 / {summary['test']['files']} 文件")
    print(f"泄漏自检: 通过(dev∩test 文件集为空)")
    print(f"已写出: {os.path.relpath(OUT_JSONL, ROOT)} , {os.path.relpath(OUT_SUMMARY, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
