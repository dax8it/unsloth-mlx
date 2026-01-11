from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_ROLE_ALIASES = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "human": "user",
    "gpt": "assistant",
    "bot": "assistant",
}


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _normalize_role(role: Any) -> str:
    r = _as_str(role).strip().lower()
    return _ROLE_ALIASES.get(r, r or "user")


def normalize_messages(messages: Any) -> Optional[List[Dict[str, str]]]:
    if not isinstance(messages, list) or len(messages) == 0:
        return None

    out: List[Dict[str, str]] = []
    for m in messages:
        if not isinstance(m, dict):
            return None
        role = _normalize_role(m.get("role"))
        content = _as_str(m.get("content")).strip()
        if role not in {"system", "user", "assistant"}:
            return None
        if content == "":
            return None
        out.append({"role": role, "content": content})

    return out if out else None


def _infer_messages_from_conversations(obj: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    conv = obj.get("conversations")
    if not isinstance(conv, list) or len(conv) == 0:
        return None

    out: List[Dict[str, str]] = []
    for turn in conv:
        if not isinstance(turn, dict):
            return None
        role = turn.get("role")
        content = turn.get("content")
        if role is None and "from" in turn:
            role = turn.get("from")
        if content is None and "value" in turn:
            content = turn.get("value")

        role_n = _normalize_role(role)
        content_s = _as_str(content).strip()
        if role_n in {"human"}:
            role_n = "user"
        if role_n in {"gpt", "bot"}:
            role_n = "assistant"
        if role_n not in {"system", "user", "assistant"}:
            return None
        if content_s == "":
            return None
        out.append({"role": role_n, "content": content_s})

    return out if out else None


def _infer_messages_from_alpaca(obj: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    instruction = _as_str(obj.get("instruction")).strip()
    output = _as_str(obj.get("output")).strip()
    if instruction == "" or output == "":
        return None

    input_text = _as_str(obj.get("input")).strip()
    user_text = instruction if input_text == "" else f"{instruction}\n\n{input_text}"

    system = _as_str(obj.get("system")).strip()
    msgs: List[Dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user_text})
    msgs.append({"role": "assistant", "content": output})
    return msgs


def _infer_messages_from_prompt_completion(obj: Dict[str, Any]) -> Optional[List[Dict[str, str]]]:
    prompt = obj.get("prompt")
    completion = obj.get("completion")
    if completion is None:
        completion = obj.get("response")

    prompt_s = _as_str(prompt).strip()
    completion_s = _as_str(completion).strip()
    if prompt_s == "" or completion_s == "":
        return None

    system = _as_str(obj.get("system")).strip()
    msgs: List[Dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt_s})
    msgs.append({"role": "assistant", "content": completion_s})
    return msgs


def infer_sft_messages(obj: Dict[str, Any]) -> Tuple[Optional[List[Dict[str, str]]], str]:
    msgs = normalize_messages(obj.get("messages"))
    if msgs is not None:
        return msgs, "messages"

    msgs = _infer_messages_from_conversations(obj)
    if msgs is not None:
        return msgs, "conversations"

    msgs = _infer_messages_from_alpaca(obj)
    if msgs is not None:
        return msgs, "alpaca"

    msgs = _infer_messages_from_prompt_completion(obj)
    if msgs is not None:
        return msgs, "prompt_completion"

    return None, "unrecognized"


def convert_rows_to_messages_jsonl(
    rows: Iterable[Dict[str, Any]],
    output_path: str,
    max_errors: int = 10,
) -> Dict[str, Any]:
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    written = 0
    errors: List[str] = []
    sources: Dict[str, int] = {}

    with open(out_path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(rows, start=1):
            total += 1
            if not isinstance(row, dict):
                if len(errors) < max_errors:
                    errors.append(f"Row {idx}: not an object")
                continue

            msgs, src = infer_sft_messages(row)
            sources[src] = sources.get(src, 0) + 1
            if msgs is None:
                if len(errors) < max_errors:
                    keys = sorted([k for k in row.keys() if isinstance(k, str)])
                    errors.append(f"Row {idx}: could not infer messages (keys={keys})")
                continue

            f.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
            written += 1

    return {
        "output": str(out_path),
        "rows_seen": total,
        "rows_written": written,
        "rows_failed": total - written,
        "source_breakdown": dict(sorted(sources.items(), key=lambda kv: (-kv[1], kv[0]))),
        "errors": errors,
    }


def convert_dataset_to_messages_jsonl(
    dataset: Any,
    output_dir: str = "data/converted",
    output_name: Optional[str] = None,
    max_errors: int = 10,
) -> Dict[str, Any]:
    ts = int(time.time())
    base = output_name or f"{ts}_sft_messages.jsonl"
    out_path = str(Path(output_dir) / base)

    rows = dataset
    if hasattr(dataset, "__iter__"):
        rows = dataset

    return convert_rows_to_messages_jsonl(rows, out_path, max_errors=max_errors)


def format_messages_with_tokenizer(
    tokenizer: Any,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False,
) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
        )

    parts: List[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")

    if add_generation_prompt:
        parts.append("Assistant: ")

    return "\n".join(parts)
