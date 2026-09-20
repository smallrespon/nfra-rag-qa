# 面向银行业监管制度与统计报表的可信 RAG 问答系统

从 500 份 NFRA 监管文件(Word/PDF/Excel)中检索依据,回答监管条款、指标口径与统计数据问题,返回可追溯到**条款/页码/单元格**的证据;依据不足时拒答。

## 评测结果(test 195 题,五项指标全部达标)

| 指标 | 目标 | 实测 |
|---|---|---|
| 制度事实类准确率 | ≥85% | **91.7%** |
| 表格类准确率 | ≥80% | **95.2%** |
| 证据引用命中率 | ≥90% | **93.8%**(文件级) |
| 关键数字错误率 | ≤5% | **0%** |
| 库外拒答率 | ≥80% | **100%** |
| 总体 MCQ 准确率 | — | **92.8%** (181/195) |

评测集与详细评测报告不入库（见下文「数据与复现说明」）。

## 环境

- Python 3.12,依赖见 `requirements.txt`
- LLM:OpenAI 兼容 / Anthropic 兼容 API 双协议(默认 qwen3.7-flash @ 百炼;`configs/llm.yaml` 切换,密钥放 `configs/secrets.yaml`——**不入库、不进提交材料**)
- 可选 GPU(RTX 4060 实测):BGE-M3 向量 + bge-reranker 精排

## 数据与复现说明（开源范围）

本仓库只含**代码、文档与实验记录**，不含以下内容（体积/版权/合规原因），按此复现：

| 不入库内容 | 原因 | 复现方式 |
|---|---|---|
| `data/数据集/`（500 份 NFRA 监管文件，约 207MB） | 体积大；版权归 NFRA/原发布方 | 从国家金融监督管理总局官网按 `data/manifest.json` 清单自行下载，放置为 `data/数据集/编号.ext` |
| 评测题库（QA数据.xlsx 及 `data/eval/` 全部衍生文件） | 教师提供材料，题库不公开 | — |
| `data/processed/`、`spreadsheet_objects.json` 等索引产物（约 470MB） | 均为派生物 | `使用` 段第 4 步知识库构建命令全流程重建 |
| `configs/secrets.yaml` | 密钥 | 自建，格式：`dashscope:\n  api_key: <你的Key>`（`chmod 600`，勿提交） |

```bash
pip install -r requirements.txt
```

## 使用

```bash
# 1) 交互问答(CLI)
python3 scripts/answer_question.py "根据《消费金融公司管理办法》,消费金融公司的最低注册资本是多少?"
python3 scripts/answer_question.py        # 交互循环

# 2) HTTP 服务
uvicorn src.api:app --host 127.0.0.1 --port 8000
#   浏览器访问 http://127.0.0.1:8000  →  Web 问答界面(答案[n]引用与证据联动/保真状态/拒答展示/示例题/历史记录)
#   POST /ask  {"question": "..."}  →  {answer, evidence[], fidelity, latency_ms}
#   GET  /health

# 3) 评测(300 题按源文件整组切分 dev/test,test 锁箱)
python3 scripts/run_baseline.py --split dev [--rerank]
python3 scripts/s3_hitrate.py data/eval/predictions_baseline.jsonl

# 4) 知识库构建(全流程可复跑)
python3 scripts/dedup_verify.py       # 926→500 去重 + SHA-256 校验
python3 scripts/convert_doc.py        # .doc→.docx(需 LibreOffice)
python3 scripts/parse_docs.py         # Word/PDF → 条款级块
python3 scripts/build_table_index.py  # Excel → 行级取数记录
python3 scripts/build_vector_index.py # BGE-M3 向量(需 GPU/网络)
```

## 系统模块

```
① 数据解析  Word/PDF(条款感知切分) + Excel(表头层级/合并单元格/单位行)
② 知识组织  4,227 条款块 + 102.7万行级取数记录,元数据: doc_id/章节/页码/单元格/家族根
③ 文本检索  BM25(jieba) + BGE-M3 向量 → RRF 融合 → bge-reranker(可选)
④ 表格查询  《文件》定位(候选内容消歧) → 指标/口径匹配 → 行列取数 → 确定性计算
⑤ 问答证据  LLM 生成 + 引用强制[n] + 数字保真校验 + 双层拒答(文件存在性规则/LLM 证据约束)
⑥ 评测      MCQ 批量判分 + 证据命中率 + 拒答率 + 错题落盘
```

## 目录

```
src/          llm_client.py(双协议客户端) api.py(HTTP 服务+Web 界面挂载) static/(前端页面,零依赖原生 HTML/CSS/JS)
scripts/      数据整备/解析/索引/检索/评测/问答 全流程脚本
configs/      llm.yaml(模型配置;secrets.yaml 密钥自建,不入库)
data/         manifest.json 与 filename_mapping.json(语料清单元数据;语料/索引/评测不入库,见上文)
LICENSE       MIT
```
