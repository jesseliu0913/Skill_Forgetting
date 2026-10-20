from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MATH_ROOT = ROOT
SKILL_RL_ROOT = Path(__file__).resolve().parent
SKILL_SFT_ROOT = MATH_ROOT / "math"

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_SFT_DATASET = "Jianwen/SkillRL-SFT-Data"

ENVIRONMENT_ORDER = ["alfworld", "webshop", "search"]

ENV_SPECS = {
    "alfworld": {
        "display_name": "ALFWorld",
        "dataset_config": "alfworld",
        "default_top_k": 6,
        "system_prompt": (
            "You are an expert embodied household agent operating in ALFWorld. "
            "Reason carefully about objects, rooms, containers, and action preconditions."
        ),
    },
    "webshop": {
        "display_name": "WebShop",
        "dataset_config": "webshop",
        "default_top_k": 6,
        "system_prompt": (
            "You are an expert shopping agent. Track user constraints, product attributes, "
            "and page state before deciding the next action."
        ),
    },
    "search": {
        "display_name": "Search",
        "dataset_config": "search",
        "default_top_k": 6,
        "system_prompt": (
            "You are an expert multi-step search agent. Break the question into evidence needs, "
            "issue targeted searches, and verify supporting facts before answering."
        ),
    },
}
