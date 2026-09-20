#!/usr/bin/env python3
"""双协议 LLM 客户端:OpenAI 格式(chat/completions)与 Anthropic 格式(v1/messages)。

配置来自 configs/llm.yaml(active 选择协议)+ configs/secrets.yaml(密钥,
已 gitignore)。不依赖 SDK,统一用 httpx,便于两种协议行为对齐与重试控制。

用法:
    from llm_client import LLMClient
    cli = LLMClient()                       # 按 llm.yaml 的 active 配置
    text = cli.chat("你是谁", system="简短回答")
"""
import json
import os
import time

import httpx
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLM_CFG = os.path.join(ROOT, "configs", "llm.yaml")
SECRET_CFG = os.path.join(ROOT, "configs", "secrets.yaml")


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, provider: str | None = None, timeout: float = 60.0):
        cfg = yaml.safe_load(open(LLM_CFG, encoding="utf-8"))
        secrets = yaml.safe_load(open(SECRET_CFG, encoding="utf-8"))
        name = provider or cfg.get("active")
        if name not in cfg.get("providers", {}):
            raise LLMError(f"未知 provider: {name}")
        p = cfg["providers"][name]
        self.protocol = p["protocol"]          # "openai" | "anthropic"
        self.base_url = p["base_url"].rstrip("/")
        self.model = p["model"]
        vendor = p.get("api_key_ref", name)    # secrets.yaml 中的键名
        self.api_key = secrets.get(vendor, {}).get("api_key")
        if not self.api_key:
            raise LLMError(f"configs/secrets.yaml 缺少 {vendor}.api_key")
        self.timeout = timeout
        self.max_retries = int(p.get("max_retries", 3))

    # ---------- 协议细节 ----------

    def _endpoint_and_headers(self) -> tuple[str, dict]:
        if self.protocol == "anthropic":
            # 兼容两种 base 形态:完整 messages 端点 或 需拼 /v1/messages 的前缀
            ep = (f"{self.base_url}/v1/messages"
                  if not self.base_url.endswith("/messages") else self.base_url)
            if "/anthropic" in self.base_url and not self.base_url.endswith("/messages"):
                ep = f"{self.base_url}/v1/messages"
            return ep, {"x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"}
        # openai:base_url 形如 .../v1
        base = self.base_url if self.base_url.endswith("/v1") else f"{self.base_url}/v1"
        return (f"{base}/chat/completions",
                {"Authorization": f"Bearer {self.api_key}",
                 "content-type": "application/json"})

    def _body(self, messages: list[dict], max_tokens: int, temperature: float) -> dict:
        if self.protocol == "anthropic":
            system = "".join(m["content"] for m in messages if m["role"] == "system")
            rest = [m for m in messages if m["role"] != "system"]
            body = {"model": self.model, "max_tokens": max_tokens,
                    "temperature": temperature, "messages": rest}
            if system:
                body["system"] = system
            return body
        return {"model": self.model, "messages": messages,
                "max_tokens": max_tokens, "temperature": temperature}

    @staticmethod
    def _extract(data: dict) -> str:
        if "choices" in data:                      # openai
            return data["choices"][0]["message"]["content"] or ""
        if data.get("type") == "message":          # anthropic
            return "".join(b.get("text", "") for b in data.get("content", []))
        raise LLMError(f"无法解析响应: {json.dumps(data, ensure_ascii=False)[:200]}")

    # ---------- 对外 ----------

    def chat(self, user: str, system: str | None = None, max_tokens: int = 1024,
             temperature: float = 0.1) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + \
                   [{"role": "user", "content": user}]
        ep, headers = self._endpoint_and_headers()
        body = self._body(messages, max_tokens, temperature)
        last = None
        for attempt in range(self.max_retries):
            try:
                r = httpx.post(ep, headers=headers, json=body, timeout=self.timeout)
                if r.status_code == 200:
                    return self._extract(r.json()).strip()
                last = LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
                if r.status_code not in (429, 500, 502, 503, 504):
                    break                        # 非瞬态错误不重试
            except httpx.HTTPError as e:
                last = LLMError(f"网络错误: {type(e).__name__}: {e}")
            time.sleep(2 ** attempt)
        raise last or LLMError("未知错误")

    def info(self) -> str:
        return f"{self.protocol}:{self.model} @ {self.base_url}"


if __name__ == "__main__":
    c = LLMClient()
    print("provider:", c.info())
    print("回复:", c.chat("用一句话回答:银行业监管统计中'原保险保费收入'属于流量指标还是存量指标?"))
