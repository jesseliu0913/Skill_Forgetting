#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import fcntl



def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    last_error: OSError | None = None
    for attempt in range(3):
        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
                    if limit is not None and len(rows) >= limit:
                        break
            return rows
        except OSError as exc:
            last_error = exc
            if exc.errno != 5 or attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    if last_error is not None:
        raise last_error
    return []


def _has_merged_weights(path: Path) -> bool:
    if (path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists():
        return True
    if (path / "model.safetensors.index.json").exists():
        return True
    return any(path.glob("model-*.safetensors"))


def _merged_cache_dir(actor_dir: Path) -> Path:
    digest = hashlib.sha1(str(actor_dir).encode("utf-8")).hexdigest()[:16]
    cache_root = os.environ.get("SKILLRL_EVAL_MERGE_ROOT")
    if cache_root:
        return Path(cache_root) / digest
    return Path("/tmp") / f"skillrl_eval_merged_{os.getuid()}" / digest


@contextlib.contextmanager
def _merge_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def resolve_model_path(model_name: str) -> str:
    model_path = Path(model_name)
    if not model_path.exists():
        return model_name

    if _has_merged_weights(model_path):
        return str(model_path)

    actor_dir = None
    if model_path.name == "huggingface" and model_path.parent.name == "actor":
        actor_dir = model_path.parent
    elif model_path.name == "actor":
        actor_dir = model_path

    if actor_dir is None:
        return model_name

    merged_dir = _merged_cache_dir(actor_dir)
    with _merge_lock(merged_dir.parent / f"{merged_dir.name}.lock"):
        if _has_merged_weights(merged_dir):
            return str(merged_dir)

        if not (actor_dir / "fsdp_config.json").exists():
            return model_name

        print(f"Merging FSDP actor checkpoint for eval: {actor_dir} -> {merged_dir}", file=sys.stderr)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "verl.model_merger",
                "merge",
                "--backend",
                "fsdp",
                "--local_dir",
                str(actor_dir),
                "--target_dir",
                str(merged_dir),
                "--use_cpu_initialization",
            ],
            check=True,
        )
    return str(merged_dir)


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", completion.get("text", "")))
    if isinstance(completion, list):
        return "".join(completion_to_text(item) for item in completion)
    return str(completion)


def iter_text_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        texts: list[str] = []
        for inner in value.values():
            texts.extend(iter_text_values(inner))
        return texts
    if isinstance(value, list):
        texts: list[str] = []
        for inner in value:
            texts.extend(iter_text_values(inner))
        return texts
    return []


def normalize_action(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().lower())
    text = re.sub(r"^<+", "", text)
    text = re.sub(r">+$", "", text)
    # Some Qwen generations emit actions like `click[buy now]>` immediately
    # before the closing XML tag. Treat that as a formatting artifact, while
    # preserving legitimate action text such as `click[< prev]`.
    text = re.sub(r"(\])>+$", r"\1", text)
    text = re.sub(r"(\})>+$", r"\1", text)
    return text


def action_tag(action: str) -> str:
    return action.split(":", 1)[0] if ":" in action else "action"


def action_content(action: str) -> str:
    return action.split(":", 1)[1] if ":" in action else action


def action_tokens(action: str) -> list[str]:
    content = action_content(action)
    return re.findall(r"[a-z0-9]+", content.lower())


def token_f1(pred_action: str, gold_action: str) -> float:
    pred_tokens = action_tokens(pred_action)
    gold_tokens = action_tokens(gold_action)
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts: Counter[str] = Counter(pred_tokens)
    gold_counts: Counter[str] = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def search_soft_metrics(pred_action: str, gold_action: str) -> dict[str, Any]:
    pred_tag = action_tag(pred_action)
    gold_tag = action_tag(gold_action)
    tool_match = bool(pred_action and gold_action and pred_tag == gold_tag)
    query_f1 = token_f1(pred_action, gold_action) if tool_match else 0.0
    return {
        "tool_match": tool_match,
        "search_query_token_f1": query_f1,
        "search_soft_match": bool(tool_match and query_f1 >= 0.5),
    }


def extract_tag_content(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return normalize_action(match.group(1))
    return ""


def extract_action(text: str, environment: str = "") -> str:
    if environment == "search":
        for tag in ("search", "answer"):
            content = extract_tag_content(text, tag)
            if content:
                return f"{tag}:{content}"
        return ""

    content = extract_tag_content(text, "action")
    if content:
        return content
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return normalize_action(lines[-1]) if lines else ""


def has_required_tags(text: str, environment: str = "") -> bool:
    has_think = bool(re.search(r"<think>.*?</think>", text, flags=re.IGNORECASE | re.DOTALL))
    if environment == "search":
        has_search = bool(re.search(r"<search>.*?</search>", text, flags=re.IGNORECASE | re.DOTALL))
        has_answer = bool(re.search(r"<answer>.*?</answer>", text, flags=re.IGNORECASE | re.DOTALL))
        return has_think and (has_search or has_answer)
    return has_think and bool(re.search(r"<action>.*?</action>", text, flags=re.IGNORECASE | re.DOTALL))


def build_user_prompt(row: dict[str, Any]) -> str:
    instruction = str(row.get("instruction", "")).strip()
    query = str(row.get("input", "")).strip()
    if instruction and query:
        return f"{instruction}\n\n{query}"
    return instruction or query


def infer_environment(row: dict[str, Any]) -> str:
    environment = str(row.get("environment", "")).strip().lower()
    if environment:
        return environment

    haystacks: list[str] = []
    for value in row.values():
        haystacks.extend(iter_text_values(value))
    pattern = re.compile(r"\benvironment\s*:\s*(alfworld|webshop|search)\b", flags=re.IGNORECASE)
    for text in haystacks:
        match = pattern.search(text)
        if match:
            return match.group(1).lower()
    serialized = json.dumps(row, ensure_ascii=False)
    match = pattern.search(serialized)
    if match:
        return match.group(1).lower()
    return ""


def truncate_to_first_valid_response(text: str, environment: str = "") -> str:
    think_match = re.search(r"<think>.*?</think>", text, flags=re.IGNORECASE | re.DOTALL)
    think_block = think_match.group(0).strip() if think_match else ""
    if environment == "search":
        tool_match = re.search(r"<(search|answer)>\s*.*?\s*</\1>", text, flags=re.IGNORECASE | re.DOTALL)
    else:
        tool_match = re.search(r"<action>\s*.*?\s*</action>", text, flags=re.IGNORECASE | re.DOTALL)

    if not tool_match:
        return text

    tool_block = tool_match.group(0).strip()
    if think_block:
        return f"{think_block}\n{tool_block}"
    return tool_block


def apply_chat_template(prompts: list[str], model_name: str) -> list[str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    formatted: list[str] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        formatted.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    return formatted


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    env_counter: Counter[str] = Counter()
    env_correct: Counter[str] = Counter()
    env_tagged: Counter[str] = Counter()
    env_tool_match: Counter[str] = Counter()
    env_soft_match: Counter[str] = Counter()
    env_query_f1: Counter[str] = Counter()
    for record in records:
        env = str(record.get("environment", "unknown"))
        env_counter[env] += 1
        env_correct[env] += int(bool(record.get("exact_action_match")))
        env_tagged[env] += int(bool(record.get("format_ok")))
        if env == "search":
            env_tool_match[env] += int(bool(record.get("tool_match")))
            env_soft_match[env] += int(bool(record.get("search_soft_match")))
            env_query_f1[env] += float(record.get("search_query_token_f1", 0.0))

    by_environment: dict[str, Any] = {}
    for env, total in sorted(env_counter.items()):
        env_summary = {
            "n": total,
            "exact_action_accuracy": env_correct[env] / total if total else 0.0,
            "format_ok_rate": env_tagged[env] / total if total else 0.0,
        }
        if env == "search":
            tool_accuracy = env_tool_match[env] / total if total else 0.0
            search_soft_accuracy = env_soft_match[env] / total if total else 0.0
            env_summary.update(
                {
                    "tool_accuracy": tool_accuracy,
                    "avg_query_token_f1": env_query_f1[env] / total if total else 0.0,
                    "search_soft_accuracy": search_soft_accuracy,
                    "report_accuracy": tool_accuracy,
                    "report_accuracy_name": "search_tool_accuracy",
                }
            )
        by_environment[env] = env_summary

    total = len(records)
    total_correct = sum(int(bool(record.get("exact_action_match"))) for record in records)
    total_tagged = sum(int(bool(record.get("format_ok"))) for record in records)
    summary = {
        "n": total,
        "exact_action_accuracy": total_correct / total if total else 0.0,
        "format_ok_rate": total_tagged / total if total else 0.0,
        "by_environment": by_environment,
    }
    if set(env_counter) == {"search"}:
        tool_accuracy = env_tool_match["search"] / total if total else 0.0
        search_soft_accuracy = env_soft_match["search"] / total if total else 0.0
        summary.update(
            {
                "tool_accuracy": tool_accuracy,
                "avg_query_token_f1": env_query_f1["search"] / total if total else 0.0,
                "search_soft_accuracy": search_soft_accuracy,
                "report_accuracy": tool_accuracy,
                "report_accuracy_name": "search_tool_accuracy",
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a base or finetuned agent model on JSONL eval data.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--input_jsonl", type=Path, required=True)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    parser.add_argument("--summary_json", type=Path, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from vllm import LLM, SamplingParams

    resolved_model = resolve_model_path(args.model)
    rows = load_rows(args.input_jsonl, args.limit)
    if not rows:
        raise ValueError(f"No rows found in {args.input_jsonl}")

    tokenizer_name = args.tokenizer or args.model
    prompts = [build_user_prompt(row) for row in rows]
    prompts = apply_chat_template(prompts, tokenizer_name)

    llm = LLM(
        model=resolved_model,
        tokenizer=tokenizer_name,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling_params=sampling_params)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row, prompt, output in zip(rows, prompts, outputs):
            raw_prediction = completion_to_text(output.outputs[0].text if output.outputs else "")
            gold = str(row.get("output", "")).strip()
            environment = infer_environment(row)
            prediction = truncate_to_first_valid_response(raw_prediction, environment)
            pred_action = extract_action(prediction, environment)
            gold_action = extract_action(gold, environment)
            record = {
                "environment": environment,
                "task_category": row.get("task_category", ""),
                "instruction": row.get("instruction", ""),
                "input": row.get("input", ""),
                "prompt": prompt,
                "prediction": prediction,
                "raw_prediction": raw_prediction,
                "gold_output": gold,
                "pred_action": pred_action,
                "gold_action": gold_action,
                "exact_action_match": bool(pred_action and gold_action and pred_action == gold_action),
                "format_ok": has_required_tags(prediction, environment),
            }
            if environment == "search":
                record.update(search_soft_metrics(pred_action, gold_action))
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summarize(records), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {len(records)} predictions to {args.output_jsonl}")
    if args.summary_json is not None:
        print(f"Wrote summary to {args.summary_json}")


if __name__ == "__main__":
    main()
