from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any


class TextGenerator(ABC):
    @abstractmethod
    def generate(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[str]:
        raise NotImplementedError


class VLLMTextGenerator(TextGenerator):
    def __init__(
        self,
        model: str,
        *,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        lora_path: str | None = None,
    ) -> None:
        from vllm import LLM

        self.lora_path = lora_path
        llm_kwargs: dict[str, Any] = {
            "model": model,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
        }
        if lora_path:
            llm_kwargs["enable_lora"] = True
        self._llm = LLM(**llm_kwargs)

    def generate(
        self,
        prompts: list[str],
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> list[str]:
        from vllm import SamplingParams

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        if self.lora_path:
            from vllm.lora.request import LoRARequest

            outputs = self._llm.generate(
                prompts,
                sampling_params=sampling_params,
                lora_request=LoRARequest("skill_sft_lora", 1, self.lora_path),
            )
        else:
            outputs = self._llm.generate(prompts, sampling_params=sampling_params)
        return [out.outputs[0].text if out.outputs else "" for out in outputs]


def parse_json_object(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if not candidate:
        raise ValueError("Empty JSON response.")
    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    if candidate.startswith("```"):
        candidate = candidate.lstrip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:]
        end = candidate.rfind("```")
        candidate = candidate[:end] if end != -1 else candidate
        candidate = candidate.strip()
    # Find the start of the first JSON object
    start = candidate.find("{")
    if start == -1:
        raise ValueError("No JSON object found in response.")
    # raw_decode stops at the end of the first valid object, ignoring trailing text
    obj, _ = json.JSONDecoder().raw_decode(candidate, start)
    return obj
