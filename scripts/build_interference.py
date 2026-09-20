#!/usr/bin/env python3
"""M3-3: 构造 S5 库外干扰题集(50 条)并程序化验证"库外"属性。

设计:
  - 25 条带《不存在的文件/报表名》→ 测规则层拒答;
  - 25 条不带《》(自然问法)→ 测 LLM 证据不足拒答层;
  - 每条验证:文件名与 500 份语料 stem 无包含/高相似匹配(LCS<12);
    无《》题的核心主题词不出现于任何语料文件名,且人工抽检语料主题清单。

输出: data/eval/interference_questions.jsonl
用法: python3 scripts/build_interference.py
"""
import difflib
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_INDEX = os.path.join(ROOT, "data", "processed", "doc_index.json")
OUT = os.path.join(ROOT, "data", "eval", "interference_questions.jsonl")


def norm(s):
    return re.sub(r"\s+", "", str(s))


def lcs_len(a, b):
    return difflib.SequenceMatcher(None, a, b).find_longest_match(
        0, len(a), 0, len(b)).size


def main() -> int:
    docs = json.load(open(DOC_INDEX, encoding="utf-8"))["docs"]
    stems = [norm(re.sub(r"^\d+_", "", re.sub(r"\.[a-zA-Z0-9]+$", "", d["orig_name"])))
             for d in docs]

    # 库外监管文件/报表名(银行/保险/支付/征信/资管等域,均不在本语料库)
    absent_docs = [
        "理财公司监管评级办法", "商业银行理财业务监督管理办法", "征信业管理条例",
        "支付机构预付卡管理办法", "汽车金融公司管理办法", "货币经纪公司试点管理办法",
        "金融租赁公司管理办法", "外资银行管理条例实施细则", "证券期货投资者适当性管理办法",
        "信托公司受托责任尽职指引", "商业银行全球系统重要性评估指引", "系统重要性银行附加监管规定",
        "理财公司内部控制管理办法", "商业银行互联网贷款管理暂行办法", "保险公司偿付能力管理规定",
        "再保险业务管理规定", "保险资产管理公司管理规定", "银行保险机构消费者权益保护管理办法",
        "个人税收递延型商业养老保险试点办法", "商业银行表外业务风险管理办法",
        "2024年1月信托公司主要业务数据表", "2025年6月金融租赁公司经营情况表",
        "2023年12月货币经纪公司统计数据表", "2024年3季度消费金融公司主要监管指标表",
        "2026年1月理财公司存续产品规模统计表",
    ]
    # 库外主题(自然问法,不带《》)
    absent_topics = [
        ("理财产品销售", "理财产品销售管理暂行办法规定的销售禁区包括哪些行为?"),
        ("杠杆率附加", "全球系统重要性银行在中国的附加杠杆率要求是多少?"),
        ("备付金交存", "支付机构客户备付金集中交存比例目前是多少?"),
        ("征信业务", "个人征信机构设立需要满足哪些准入条件?"),
        ("互联网贷款", "商业银行互联网贷款的单户授信额度上限是多少?"),
        ("信托业务", "信托公司开展资产管理信托需要满足什么受托要求?"),
        ("金融租赁", "金融租赁公司开办融资租赁业务的最低注册资本是多少?"),
        ("货币经纪", "货币经纪公司可以从事哪些经纪业务范围?"),
        ("外资银行准入", "外资银行在华设立法人银行的最低注册资本要求?"),
        ("适当性管理", "证券期货经营机构投资者适当性管理的评估要求有哪些?"),
        ("理财内控", "理财公司内部控制有哪些强制性岗位分离要求?"),
        ("偿二代二期", "保险公司核心偿付能力充足率的监管红线是多少?"),
        ("再保险", "保险公司办理再保险分出业务需要报送哪些材料?"),
        ("保险资管", "保险资产管理公司受托管理保险资金的禁止行为有哪些?"),
        ("消保审查", "银行保险机构消费者权益保护审查机制包括哪些环节?"),
        ("税收递延养老", "个人税收递延型商业养老保险的缴费上限是多少?"),
        ("表外业务", "商业银行表外业务风险加权资产如何计量?"),
        ("汽车金融", "汽车金融公司发放经销商贷款的额度限制是多少?"),
        ("附加资本", "系统重要性银行附加资本要求的分档标准是什么?"),
        ("理财评级", "理财公司监管评级结果分为几个等级?"),
        ("保险销售回溯", "保险销售行为可回溯管理要求的音视频保存期限是多久?"),
        ("银行卡收单", "银行卡收单机构的特约商户准入管理要求有哪些?"),
        ("债转股", "市场化银行债权转股权的转股企业条件有哪些?"),
        ("网贷存管", "网络借贷资金存管业务的存管人条件有哪些?"),
        ("金融资产投资", "金融资产投资公司设立的条件和业务范围是什么?"),
        ("信用卡息费", "信用卡新规对息费披露有哪些强制要求?"),
        ("养老理财试点", "养老理财产品试点的业务要求有哪些?"),
        ("代理保险销售", "商业银行代理保险业务的销售规范有哪些?"),
        ("保险资金不动产", "保险资金投资不动产的比例限制是多少?"),
    ]
    assert len(absent_docs) == 25 and len(absent_topics) == 29

    qs, rejected = [], []
    for i, name in enumerate(absent_docs, 1):
        if any(name in s or lcs_len(name, s) >= 12 for s in stems):
            rejected.append((f"D{i:02d}", name, "与库内文件名过近"))
            continue
        # 三种问法轮换,覆盖事实/阈值/取数/比较/场景
        tpl = [
            f"根据《{name}》,其主要适用范围和核心要求是什么?",
            f"《{name}》规定的关键数值阈值是多少?",
            f"根据 Excel 附件《{name}》,相关指标的数值是多少?",
            f"检索《{name}》后,以下判断是否符合该文件内容?",
            f"对比《{name}》与现行监管要求,主要差异是什么?",
        ]
        qs.append({"id": f"S5-D{i:02d}", "question": tpl[(i - 1) % 5],
                   "kind": "with_book_title", "topic": name})
    for i, (topic, q) in enumerate(absent_topics, 1):
        kw = norm(topic)
        if any(kw in s for s in stems):
            rejected.append((f"T{i:02d}", topic, "主题词出现在库内文件名"))
            continue
        qs.append({"id": f"S5-T{i:02d}", "question": q,
                   "kind": "natural", "topic": topic})

    with open(OUT, "w", encoding="utf-8") as fp:
        for x in qs:
            fp.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"干扰题 {len(qs)} 条(带《》{sum(1 for x in qs if x['kind']=='with_book_title')} / "
          f"自然{sum(1 for x in qs if x['kind']=='natural')}) → {os.path.relpath(OUT, ROOT)}")
    if rejected:
        print(f"剔除 {len(rejected)} 条(与库过近):")
        for r in rejected:
            print("  ", r)
    return 0


if __name__ == "__main__":
    sys.exit(main())
