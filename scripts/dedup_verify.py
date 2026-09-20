#!/usr/bin/env python3
"""M0-1: 语料去重与 SHA-256 校验。

把 data/数据集/ 中 926 个文件(500 份唯一文件的长短名双副本)归一到 500 份,
以 filename_mapping.json 的短名(local_path)为规范路径,逐一与 manifest.json
中长名文件的 SHA-256 比对,验证副本一致性;产出 data/processed/doc_index.json。

用法: python3 scripts/dedup_verify.py
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")


def sha256(path: str, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        while True:
            chunk = fp.read(buf_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    with open(os.path.join(DATA, "manifest.json"), encoding="utf-8") as fp:
        manifest = json.load(fp)
    with open(os.path.join(DATA, "filename_mapping.json"), encoding="utf-8") as fp:
        mapping = json.load(fp)

    # manifest 以长名(无 data/ 前缀)为 path;mapping 的 zip_path 带 data/ 前缀
    m_by_zipname = {f["path"]: f for f in manifest["files"]}

    doc_index, problems = [], []
    for entry in mapping:
        num = entry["num"]
        local_path = os.path.join(ROOT, entry["local_path"])   # 短名副本 data/数据集/145.xls
        zip_path = entry["zip_path"].split("data/", 1)[-1]     # 长名 数据集/145_..._.xls
        manifest_entry = m_by_zipname.get(zip_path)

        rec = {
            "doc_id": num,
            "orig_name": entry["orig_name"],
            "type": entry["ext"],
            "canonical_path": entry["local_path"],  # 相对项目根,统一短名
            "manifest_size": manifest_entry["size"] if manifest_entry else None,
        }
        if not os.path.isfile(local_path):
            problems.append({"doc_id": num, "issue": "短名副本缺失", "path": entry["local_path"]})
            rec["status"] = "missing"
        else:
            digest = sha256(local_path)
            rec["sha256"] = digest
            if manifest_entry and digest == manifest_entry["sha256"]:
                rec["status"] = "ok"
            elif manifest_entry:
                rec["status"] = "hash_mismatch"
                problems.append({"doc_id": num, "issue": "短名副本与 manifest SHA-256 不一致",
                                 "path": entry["local_path"]})
            else:
                rec["status"] = "not_in_manifest"
                problems.append({"doc_id": num, "issue": "manifest 中无对应长名条目",
                                 "path": zip_path})
        doc_index.append(rec)

    out = os.path.join(DATA, "processed", "doc_index.json")
    with open(out, "w", encoding="utf-8") as fp:
        json.dump({"dataset_root": "data/数据集", "doc_count": len(doc_index),
                   "docs": doc_index}, fp, ensure_ascii=False, indent=1)

    n_ok = sum(1 for d in doc_index if d.get("status") == "ok")
    n_other = len(doc_index) - n_ok
    by_type = {}
    for d in doc_index:
        by_type[d["type"]] = by_type.get(d["type"], 0) + 1
    print(f"唯一文件(短名规范): {len(doc_index)}  校验通过: {n_ok}  异常: {n_other}")
    print(f"类型分布: {by_type}")
    if problems:
        print(f"异常明细({len(problems)}):")
        for p in problems[:20]:
            print("  ", p)
    print(f"已写出: {os.path.relpath(out, ROOT)}")
    return 0 if n_other == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
