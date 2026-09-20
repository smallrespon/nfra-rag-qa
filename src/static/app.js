/* 监管制度可信 RAG 问答系统 — 前端交互逻辑(原生 JS,无外部依赖) */
"use strict";

/* ---------- 示例题目(制度事实 / 表格取数取自 dev 集,拒答演示取自库外干扰题) ---------- */
const EXAMPLES = [
  { tag: "制度事实", short: "最低注册资本",
    q: "根据《消费金融公司管理办法》，消费金融公司的最低注册资本是多少？" },
  { tag: "表格取数", short: "原保险保费收入",
    q: "根据 Excel 附件《2024年9月人身险公司经营情况表》（工作表：人身保险公司（月度）），“原保险保费收入”在“本年累计/截至当期”口径下的数值是多少？" },
  { tag: "库外拒答", short: "证据不足演示",
    q: "根据《理财公司监管评级办法》，其主要适用范围和核心要求是什么？" },
];

const HISTORY_KEY = "rag_qa_history";
const MAX_HISTORY = 30;
const TIMEOUT_MS = 90000;
const REFUSAL_TEXT = "未找到足够证据";

const $ = (sel) => document.querySelector(sel);
const questionBox = $("#question");
const askBtn = $("#ask-btn");
const historyBox = $("#history");

/* ---------- 服务健康轮询 ---------- */
let serviceReady = false;

function setHealth(ready, detail) {
  serviceReady = ready;
  const el = $("#health");
  el.classList.toggle("ok", ready);
  el.classList.toggle("loading", !ready);
  el.classList.remove("err");
  $("#health-text").textContent = ready ? "服务就绪" : "模型加载中…";
  el.title = detail || (ready ? "服务就绪" : "检索模型与大模型客户端正在加载,请稍候");
  askBtn.disabled = !ready || pending;
}

async function pollHealth() {
  try {
    const r = await fetch("/health", { cache: "no-store" });
    const j = await r.json();
    const ready = j.status === "ok" && j.provider && j.provider !== "loading";
    setHealth(ready, ready ? j.provider : "loading");
  } catch {
    const el = $("#health");
    el.classList.remove("ok", "loading");
    el.classList.add("err");
    $("#health-text").textContent = "服务未连接";
    el.title = "无法访问 /health,请确认服务已启动: uvicorn src.api:app --host 127.0.0.1 --port 8000";
    serviceReady = false;
    askBtn.disabled = true;
  }
}

/* ---------- 会话历史(localStorage 持久化,上限 30 条) ---------- */
let entries = [];

function loadHistory() {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    if (raw) entries = JSON.parse(raw).filter((e) => e.state !== "loading");
  } catch { entries = []; }
}

function saveHistory() {
  try {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(entries.slice(0, MAX_HISTORY)));
  } catch { /* 存储不可用时仅在内存保留 */ }
}

function addEntry(entry) {
  entries.unshift(entry);
  if (entries.length > MAX_HISTORY) entries.length = MAX_HISTORY;
  render();
}

/* ---------- 提问流程 ---------- */
let pending = false;

async function ask() {
  const q = questionBox.value.trim();
  if (!q || pending || !serviceReady) return;
  pending = true;
  askBtn.disabled = true;
  questionBox.value = "";
  addEntry({ id: Date.now() + Math.random(), q, state: "loading", result: null, error: null, ts: Date.now() });

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);
  try {
    const r = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
      signal: controller.signal,
    });
    if (!r.ok) throw new Error(`服务返回 HTTP ${r.status}`);
    const result = await r.json();
    updateEntry(entryIdOf(q), { state: "done", result });
  } catch (err) {
    const msg = err.name === "AbortError"
      ? `请求超时（>${TIMEOUT_MS / 1000} 秒）。大模型生成偶尔较慢，请重试。`
      : `请求失败：${err.message}。请检查服务是否运行后重试。`;
    questionBox.value = q; // 保留问题便于重试
    updateEntry(entryIdOf(q), { state: "error", error: msg });
  } finally {
    clearTimeout(timer);
    pending = false;
    askBtn.disabled = !serviceReady;
    questionBox.focus();
  }
}

function entryIdOf(q) {
  return entries.find((e) => e.q === q && e.state === "loading");
}

function updateEntry(id, patch) {
  if (!id) { render(); return; }
  Object.assign(id, patch);
  saveHistory();
  render();
}

/* ---------- 渲染 ---------- */
function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function isRefused(result) {
  return result.answer === REFUSAL_TEXT ||
    (result.fidelity && result.fidelity.status === "refused");
}

/** 把答案文本中的 [n] 引用渲染为可点击角标,点击滚动并高亮对应证据卡片 */
function renderAnswer(text, evidenceNos, card) {
  const body = el("div", "a-body");
  const re = /\[(\d{1,2})\]/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) body.append(text.slice(last, m.index));
    const no = parseInt(m[1], 10);
    if (evidenceNos.includes(no)) {
      const sup = el("sup", "cite", `[${no}]`);
      sup.title = "查看证据 " + no;
      sup.addEventListener("click", () => {
        const ev = card.querySelector(`.evidence[data-no="${no}"]`);
        if (!ev) return;
        ev.scrollIntoView({ behavior: "smooth", block: "center" });
        ev.classList.remove("flash");
        void ev.offsetWidth; // 重启动画
        ev.classList.add("flash");
      });
      body.append(sup);
    } else {
      body.append(m[0]);
    }
    last = m.index + m[0].length;
  }
  if (last < text.length) body.append(text.slice(last));
  return body;
}

function renderResult(card, result) {
  const refused = isRefused(result);
  if (refused) card.classList.add("refused");

  const aRow = el("div", "qa-a");
  aRow.append(el("span", "avatar a", "答"));
  const evidenceNos = (result.evidence || []).map((e) => e.no);
  const body = renderAnswer(result.answer || "", evidenceNos, card);
  if (refused) body.classList.add("refused-text");
  aRow.append(body);
  card.append(aRow);

  /* 状态徽章: 路径 / 保真 / 耗时 */
  const badges = el("div", "badges");
  if (result.note) badges.append(el("span", "badge info", result.note));
  const fid = result.fidelity || { status: "ok", missing: [] };
  if (fid.status === "ok") {
    badges.append(el("span", "badge ok", "保真校验通过"));
  } else if (fid.status === "warn") {
    const nums = (fid.missing || []).join("、");
    badges.append(el("span", "badge warn",
      "数字告警：答案中的数字 " + nums + " 未见于证据，请人工核对"));
  } else if (fid.status === "refused") {
    badges.append(el("span", "badge bad", "已拒答（证据不足）"));
  }
  if (typeof result.latency_ms === "number") {
    badges.append(el("span", "badge", "耗时 " + (result.latency_ms / 1000).toFixed(1) + " s"));
  }
  card.append(badges);

  /* 证据列表 */
  if (result.evidence && result.evidence.length) {
    const list = el("div", "evidence-list");
    for (const ev of result.evidence) {
      const item = el("div", "evidence");
      item.dataset.no = ev.no;
      const head = el("div", "ev-head");
      head.append(el("span", "ev-no", "证据 [" + ev.no + "]"));
      head.append(el("span", "ev-src", ev.source || "未知来源"));
      if (ev.location) head.append(el("span", "ev-loc", "@" + ev.location));
      item.append(head);
      item.append(el("div", "ev-text", ev.text || ""));
      list.append(item);
    }
    card.append(list);
  }
}

function render() {
  historyBox.textContent = "";
  if (!entries.length) {
    historyBox.append(el("div", "empty", "暂无问答记录，先在上方提一个问题吧。"));
    return;
  }
  for (const entry of entries) {
    const card = el("div", "qa-card card");
    if (entry.state === "done" && entry.result && isRefused(entry.result)) {
      card.classList.add("refused");
    }

    const qRow = el("div", "qa-q");
    qRow.append(el("span", "avatar q", "问"));
    qRow.append(el("div", "q-text", entry.q));
    card.append(qRow);

    if (entry.state === "loading") {
      const box = el("div", "loading-box");
      box.append(el("span", "spinner"));
      box.append(document.createTextNode("检索与生成中…（首次调用需加载模型，通常数秒到数十秒）"));
      card.append(box);
    } else if (entry.state === "error") {
      card.append(el("div", "error-box", entry.error || "未知错误"));
    } else if (entry.state === "done" && entry.result) {
      renderResult(card, entry.result);
    }
    historyBox.append(card);
  }
}

/* ---------- 初始化 ---------- */
function initExamples() {
  const wrap = $("#example-chips");
  for (const ex of EXAMPLES) {
    const chip = el("button", "chip");
    chip.type = "button";
    chip.append(el("span", "tag", ex.tag));
    chip.append(document.createTextNode("· " + ex.short));
    chip.title = ex.q;
    chip.addEventListener("click", () => {
      questionBox.value = ex.q;
      questionBox.focus();
    });
    wrap.append(chip);
  }
}

function init() {
  initExamples();
  loadHistory();
  render();
  askBtn.addEventListener("click", ask);
  $("#clear-btn").addEventListener("click", () => {
    entries = [];
    saveHistory();
    render();
  });
  questionBox.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
      e.preventDefault();
      ask();
    }
  });
  pollHealth();
  setInterval(pollHealth, 2500);
}

init();
