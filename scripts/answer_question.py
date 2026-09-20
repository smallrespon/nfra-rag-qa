#!/usr/bin/env python3
"""M3 生成层:问答服务形态——检索取证据 → LLM 引用强制生成 → 保真校验。

流程:
  1. 规则层拒答:题面《文件》不在库 → 直接拒答(不调 LLM);
  2. 证据检索:表格题走精确取数记录;文本题走文档圈定/混合召回 Top 块;
  3. LLM 生成(qwen3.7-flash,双协议客户端):只准依据证据作答,
     结论挂证据编号 [n],不足必须回答"未找到足够证据";
  4. 数字保真校验(S4):答案中的数值必须在证据中出现(2位小数容差),
     不符则标注告警。

用法:
    python3 scripts/answer_question.py "问题..."
    python3 scripts/answer_question.py            # 进入交互循环
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_client import LLMClient                     # noqa: E402
from run_baseline import TableQA, TextRetriever, norm  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCKS = os.path.join(ROOT, "data", "processed", "text_blocks.jsonl")

NUM_RE = re.compile(r"\d+(?:\.\d+)?")


def get_evidence(q: str, tr: TextRetriever, tq: TableQA):
    """返回 (kind, evidence_list, doc_match)。evidence: [{no, text, source, location}]"""
    m = re.search(r"《([^》]+)》", q)
    if m:
        doc_id = tr.locate_doc(m.group(1))
        if doc_id is None:
            return "rule_refuse", [], None
        # 表格题:文件可定位且问指标数值 → 精确取数证据
        title, sheet, quoted = tq.parse_question({"question": q})
        if quoted and any(tq.lookup(doc_id, t, sheet) for t in quoted):
            evs, seen = [], set()
            for t in quoted[:4]:
                for rec, kind in tq.lookup(doc_id, t, sheet)[:2]:
                    cell = f"{chr(64 + rec['col'])}{rec['row']}"
                    key = (rec['sheet'], cell)
                    if key in seen:
                        continue
                    seen.add(key)
                    evs.append({"no": len(evs) + 1,
                                "text": f"工作表[{rec['sheet'].strip()}] 单元格{cell}:"
                                        f"{rec['indicator']}({rec.get('family_root', '')}) "
                                        f"口径[{rec['column_label']}] = {rec['value']} {rec['unit']}".replace("()", ""),
                                "source": rec['doc_id'], "location": f"{rec['sheet'].strip()}!{cell}"})
            if evs:
                return "table", evs, doc_id
        cands = tr.by_doc.get(doc_id, [])
        evs = [{"no": i + 1, "text": nt[:600], "source": b["orig_name"],
                "location": b["location"]} for i, (b, nt) in enumerate(cands[:8])]
        return "text_doc", evs, doc_id
    # 无《》:混合召回 Top-6
    hits = tr.topk_hybrid(q, k=6)
    evs = [{"no": i + 1, "text": b["text"][:600], "source": b["orig_name"],
            "location": b["location"]} for i, (b, _) in enumerate(hits)]
    return "text_hybrid", evs, None


SYSTEM = ("你是银行业监管制度与统计报表问答助手。严格遵守:"
          "1)只依据用户提供的编号证据回答,每个关键结论后标注证据编号,如[1]或[1][3];"
          "2)不得编造或修改数字、日期、机构名称、文号;"
          "3)若证据不足以回答,必须只回答:未找到足够证据;"
          "4)用简体中文,简洁作答。")


def fidelity_check(answer: str, evs: list[dict]) -> dict:
    """S4 数字保真:答案中的关键数值(2位小数口径)必须出现在证据里。

    先剔除引用标注[1]、列表序号(行首 1./1、/①)等非关键数字再比对。"""
    if "未找到足够证据" in answer:
        return {"status": "refused", "missing": []}
    hay = norm("".join(e["text"] for e in evs))
    cleaned = re.sub(r"\[\d+\]", "", answer)                 # 引用标注
    cleaned = re.sub(r"(?m)^\s*\d+[.、)]\s*", "", cleaned)    # 行首列表序号
    cleaned = re.sub(r"[①②③④⑤⑥⑦⑧⑨⑩]", "", cleaned)        # 圈号序号
    missing = []
    for m in NUM_RE.findall(cleaned):
        v = m.rstrip(".")
        if v and v not in hay and f"{float(v):.2f}".rstrip("0").rstrip(".") not in hay:
            # 容差:整数/两位小数形态都查不到才告警
            if float(v) != int(float(v)) and f"{float(v):.0f}" not in hay:
                missing.append(v)
            elif float(v) == int(float(v)) and v not in hay:
                missing.append(v)
    return {"status": "ok" if not missing else "warn", "missing": missing}


def answer(q: str, cli: LLMClient, tr: TextRetriever, tq: TableQA) -> dict:
    kind, evs, doc_id = get_evidence(q, tr, tq)
    if kind == "rule_refuse":
        return {"answer": "未找到足够证据",
                "note": f"题面《{m.group(1) if (m := re.search(r'《([^》]+)》', q)) else ''}》不在语料库中(规则层拒答)",
                "evidence": [], "fidelity": {"status": "refused", "missing": []}}
    ev_block = "\n".join(f"[{e['no']}] ({e['location']}) {e['text']}" for e in evs)
    user = f"证据材料:\n{ev_block}\n\n问题:{q}"
    ans = cli.chat(user, system=SYSTEM, max_tokens=600, temperature=0.1)
    fid = fidelity_check(ans, evs)
    return {"answer": ans, "note": f"证据路径={kind}" + (f", doc={doc_id}" if doc_id else ""),
            "evidence": evs, "fidelity": fid}


def render(res: dict) -> str:
    lines = [f"【答案】{res['answer']}", f"【保真】{res['fidelity']['status']}"
             + (f" 未见于证据的数字: {res['fidelity']['missing']}" if res['fidelity']['missing'] else ""),
             f"【路径】{res['note']}"]
    if res["evidence"]:
        lines.append("【证据与来源】")
        for e in res["evidence"][:6]:
            lines.append(f"  [{e['no']}] {e['source'][:52]} @ {e['location']}")
    return "\n".join(lines)


def main() -> int:
    blocks = [json.loads(l) for l in open(BLOCKS, encoding="utf-8")]
    tr, tq = TextRetriever(blocks), TableQA()
    cli = LLMClient()
    print(f"生成层就绪: {cli.info()}")
    args = sys.argv[1:]
    qs = args if args else None
    if not qs:
        print("输入问题(Ctrl-C 退出):")
        while True:
            try:
                q = input("\n问> ").strip()
            except (EOFError, KeyboardInterrupt):
                return 0
            if q:
                print(render(answer(q, cli, tr, tq)))
    else:
        for q in qs:
            print(f"\n问> {q}")
            print(render(answer(q, cli, tr, tq)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
