"""Run the confirmed PAPER-format RAG prompt comparison on the nested run."""
from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"


def load_v1():
    path = SRC / "15_nested_custom_prompt_eval.py"
    spec = importlib.util.spec_from_file_location("custom_prompt_eval_v1", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_v1()
M.OUT = M.BASE_OUT / "CUSTOM_PROMPT_EVAL_V2"

# These are the confirmed single-line templates.  The optimized template is
# the runtime form of the historical PAPER_* RAG prompt.
M.PROMPTS = {
    "optimized_rag": {
        "prompt_id": "paper_table_viii_rag_confirmed",
        "template": (
            "Classify the text into one of {valid_labels}. "
            "Below is some related reference content that might help you classify the new text: "
            "{reference_context}. Now classify this text: {text}. "
            "Please only output one of the following labels: {valid_labels}. "
            "Do not output anything else."
        ),
        "uses_rag": True,
        "description": "RAG；確認後的 PAPER_* optimized prompt；top_k=1",
    },
    "base_rag": {
        "prompt_id": "user_base_rag_prompt_paper_format_confirmed",
        "template": (
            "Classify the text into one of {valid_labels}. "
            "refer to the following reference content: {reference_context}. "
            "Now classify this text: {text}. "
            "output only one of the following labels: {valid_labels}."
        ),
        "uses_rag": True,
        "description": "RAG；確認後的 PAPER_* 格式 base prompt；top_k=1",
    },
}

M.CONDITIONS = (
    ("rag_noaug_optimized", "optimized_rag", "noaug"),
    ("rag_aug_optimized", "optimized_rag", "aug"),
    ("rag_noaug_base", "base_rag", "noaug"),
    ("rag_aug_base", "base_rag", "aug"),
)


def render(template: str, text: str, reference_context: str | None = None) -> str:
    """Use the PAPER_* placeholder order, then fill the actual inputs."""
    prompt = (
        template
        .replace("{valid_labels}", M.VALID_LABELS)
        .replace("{text}", text)
    )
    if reference_context is not None:
        prompt = prompt.replace("{reference_context}", reference_context)
    if "{" in prompt or "}" in prompt:
        M.C.die("rendered V2 prompt contains an unreplaced placeholder")
    return prompt


M.render = render

if __name__ == "__main__":
    M.main()
