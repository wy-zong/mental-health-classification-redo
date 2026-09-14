"""Prompt 組件與候選指令集。

**這支檔案是整個設計的樞紐。** 原始實驗把「要不要接 RAG」綁死在 prompt 裡：
`prompt最佳化.py` 的每一個候選模板都內建 `{reference_context}`，所以 Optuna 從第一個
trial 起就是 RAG 打開的狀態，「純 LLM 先最佳化」這個中間狀態從未存在過。結果是
RAG 與 prompt 最佳化兩個變因永遠無法分離。

這裡把 prompt 拆成兩個彼此獨立的組件：

    指令段 instruction   —— 唯一被最佳化的對象，本身完全不提參考資料
    參考段 reference     —— 固定格式、固定位置，只有接 RAG 的條件才掛上去

七個條件共用同一個 assemble()，所以「有沒有 RAG」與「指令有沒有被最佳化」是兩個
真正獨立的開關，測出來的效果才歸得了因。
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

# ---------------------------------------------------------------- 組裝規則

# 結尾的引導詞對所有條件一律相同，不納入最佳化 —— 它屬於組裝規則，不屬於指令內容。
ANSWER_CUE = "Answer:"
TEXT_HEADER = "Text to classify:"
REFERENCE_HEADER = "Reference material (similar texts with their true labels):"


def assemble(instruction: str, text: str, reference_block: str | None = None) -> str:
    """把指令段、（可選的）參考段與待分類文字組裝成最終 prompt。

    順序固定為 指令 → 參考 → 待分類文字 → 引導詞。接不接參考段是唯一的差別，
    其餘一字不動，這樣 S0/S1（無 RAG）與 S2/S3/S4（有 RAG）之間才只差一個變因。
    """
    parts = [instruction.strip()]
    if reference_block:
        parts.append(reference_block.strip())
    parts.append(f"{TEXT_HEADER}\n{text.strip()}")
    parts.append(ANSWER_CUE)
    return "\n\n".join(parts)


def build_reference_block(docs: list[str]) -> str:
    """把檢索到的語料庫條目組成參考段。

    docs 的每一條就是語料庫裡的原樣字串（格式為 `<文本> true_label is <類別>`），
    沿用原研究「檢索到的範例要帶標籤才有參考價值」的設計。
    """
    body = "\n".join(f"- {d.strip()}" for d in docs)
    return f"{REFERENCE_HEADER}\n{body}"


# ---------------------------------------------------------------- 基準指令段

# S0 / S2a / S3a 使用。刻意保持樸素：它代表「一個人第一次用 LLM 做這件事會怎麼寫」，
# 是整條流程的真實起點。**一旦跑過 S0 就不得再修改** —— 看到結果再回頭改它，
# 等同於拿 test 做選擇。其 SHA256 由 11_verify_all.py 斷言全程未變。
SIMPLE_INSTRUCTION = """\
Based on the content of the text, tell me the emotional state of the person who wrote it.

0 = Normal
1 = Depression
2 = Anxiety
3 = Bipolar

Respond with only the number."""


# ---------------------------------------------------------------- 候選指令集

# 原始實驗宣稱搜尋「300 種 prompt 變體」，實際只有 122 個，而且全部是同一句話的措辭
# 改寫（0 個含 persona、0 個含 CoT、0 個含 few-shot）—— 那不是搜尋空間，是同義詞替換。
#
# 這裡改成沿四個有意義的維度做完整交叉，每個候選都能說清楚它跟別人差在哪，
# 也讓最佳化結果可以拆成「哪個維度有效」來報告，而不是只有一個不可解釋的贏家。

TASK_FRAMING = {
    "plain": "Based on the content of the text, tell me the emotional state of the person who wrote it.",
    "clinical": (
        "You are assessing social media posts for mental health research. "
        "Read the text and determine the author's mental health state."
    ),
    "analytical": (
        "Read the following text carefully and identify which mental health category "
        "best describes the author's state, based on the language, tone, and content."
    ),
}

LABEL_PRESENTATION = {
    "bare": "0 = Normal\n1 = Depression\n2 = Anxiety\n3 = Bipolar",
    "described": (
        "0 = Normal (no signs of mental health difficulty)\n"
        "1 = Depression (persistent low mood, hopelessness, loss of interest)\n"
        "2 = Anxiety (excessive worry, fear, restlessness, panic)\n"
        "3 = Bipolar (mood swings between elevated/manic and depressive states)"
    ),
    "contrastive": (
        "0 = Normal\n1 = Depression\n2 = Anxiety\n3 = Bipolar\n"
        "Choose Normal only if the text shows no indication of the other three."
    ),
}

REASONING = {
    "direct": "",
    "brief": "Consider the emotional tone and specific symptoms mentioned before deciding.",
}

FORMAT_RULE = {
    "simple": "Respond with only the number.",
    "emphatic": (
        "Respond with only a single digit (0, 1, 2, or 3). "
        "Do not explain, do not add any other text."
    ),
}


def build_candidates() -> list[dict]:
    """產生完整的候選指令集（四個維度的笛卡爾積）。

    回傳每個候選的 id、四個維度的取值、指令全文與 SHA256。維度取值會一併落地，
    這樣事後可以分析「是哪個維度在起作用」，而不是只知道某個編號贏了。
    """
    candidates: list[dict] = []
    for f_key, framing in TASK_FRAMING.items():
        for l_key, labels in LABEL_PRESENTATION.items():
            for r_key, reasoning in REASONING.items():
                for fmt_key, fmt in FORMAT_RULE.items():
                    blocks = [framing, labels]
                    if reasoning:
                        blocks.append(reasoning)
                    blocks.append(fmt)
                    text = "\n\n".join(blocks)
                    candidates.append({
                        "id": f"{f_key}-{l_key}-{r_key}-{fmt_key}",
                        "dims": {
                            "task_framing": f_key,
                            "label_presentation": l_key,
                            "reasoning": r_key,
                            "format_rule": fmt_key,
                        },
                        "instruction": text,
                        "sha256": C.sha256_text(text),
                    })
    return candidates


SIMPLE_INSTRUCTION_SHA256 = C.sha256_text(SIMPLE_INSTRUCTION)


if __name__ == "__main__":
    cands = build_candidates()
    C.log(f"候選指令集：{len(cands)} 個"
          f"（{len(TASK_FRAMING)} framing × {len(LABEL_PRESENTATION)} labels × "
          f"{len(REASONING)} reasoning × {len(FORMAT_RULE)} format）")
    C.log(f"基準指令 SHA256: {SIMPLE_INSTRUCTION_SHA256}")
    print("\n--- 基準指令（S0/S2a/S3a 使用）---")
    print(SIMPLE_INSTRUCTION)
    print("\n--- 組裝範例：無 RAG ---")
    print(assemble(SIMPLE_INSTRUCTION, "I can't sleep and everything feels pointless."))
    print("\n--- 組裝範例：有 RAG ---")
    print(assemble(
        SIMPLE_INSTRUCTION,
        "I can't sleep and everything feels pointless.",
        build_reference_block([
            "I have not slept in days and nothing matters true_label is Depression",
            "my heart races every night true_label is Anxiety",
        ]),
    ))
