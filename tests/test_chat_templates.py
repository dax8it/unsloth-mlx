import json
from pathlib import Path


def test_infer_sft_messages_passthrough_messages():
    from unsloth_mlx.chat_templates import infer_sft_messages

    row = {
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    }

    msgs, src = infer_sft_messages(row)
    assert src == "messages"
    assert msgs is not None
    assert msgs[0]["role"] == "system"


def test_infer_sft_messages_from_conversations_variants():
    from unsloth_mlx.chat_templates import infer_sft_messages

    row = {
        "conversations": [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello"},
        ]
    }

    msgs, src = infer_sft_messages(row)
    assert src == "conversations"
    assert msgs == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]


def test_infer_sft_messages_from_alpaca():
    from unsloth_mlx.chat_templates import infer_sft_messages

    row = {
        "instruction": "Write a slogan",
        "input": "Brand: Foo",
        "output": "Foo it up.",
    }

    msgs, src = infer_sft_messages(row)
    assert src == "alpaca"
    assert msgs is not None
    assert msgs[-1]["role"] == "assistant"
    assert msgs[-1]["content"] == "Foo it up."


def test_infer_sft_messages_from_prompt_completion():
    from unsloth_mlx.chat_templates import infer_sft_messages

    row = {"prompt": "2+2?", "completion": "4"}
    msgs, src = infer_sft_messages(row)
    assert src == "prompt_completion"
    assert msgs == [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
    ]


def test_convert_rows_to_messages_jsonl(tmp_path: Path):
    from unsloth_mlx.chat_templates import convert_rows_to_messages_jsonl

    out = tmp_path / "out.jsonl"
    rows = [
        {"prompt": "Hi", "completion": "Hello"},
        {"instruction": "Say hi", "output": "Hello"},
        {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]},
        {"bad": "row"},
    ]

    summary = convert_rows_to_messages_jsonl(rows, str(out))
    assert summary["rows_seen"] == 4
    assert summary["rows_written"] == 3
    assert summary["rows_failed"] == 1

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    obj0 = json.loads(lines[0])
    assert "messages" in obj0
    assert obj0["messages"][0]["role"] in {"system", "user"}
