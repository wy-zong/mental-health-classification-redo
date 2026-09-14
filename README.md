# 心理健康狀態分類 —— LLM / RAG / 資料擴增 / Prompt 最佳化的逐層驗證

從零重做的實驗流程。**與舊 repo（`../Code`）完全切開，不沿用任何舊產物或舊程式碼。**

## 為什麼從零重做

舊 repo 裡有兩套東西：原始實驗腳本，以及為了趕期刊死線做的 `revision/` 修訂流程。
後者採「靠剔除而非重建」的最小改動策略——沿用既有的 FAISS 索引與擴增資料，只把有問題
的樣本從評估集剔除。那批數字繼續供回覆審稿意見引用，但不適合當作新工作的基礎。

從原始碼與真實資料查證出的問題：

| 問題 | 實情 |
|---|---|
| 訓練/測試重複 | `預處理.py` 直接 `train_test_split`，無任何去重。實測資料有 1,140 筆逐字重複、另有大量同一則貼文的節錄或改寫（cosine 0.92–0.998），同一份文字會各站 train/test 一邊 |
| 語料庫含測試資料 | 語料庫來源池從未與測試集互斥，測試樣本連同 `true_label` 一起躺在檢索庫裡 |
| Embedding 截斷 | `all-MiniLM-L6-v2` 預設 `max_seq_length=256`，**18.9% 的文本**只有前段被編碼（`num_ctx` 反而幾乎無影響：超過 2048 的僅 4 筆） |
| 標籤解析順序偏誤 | `for label in LABELS: if label.lower() in result.lower()` 取第一個命中，`"not Normal; it is Depression"` 被判為 Normal |
| RAG 與 prompt 最佳化無法分離 | `prompt最佳化.py` 的候選模板**每一個**都內建 `{reference_context}`，搜尋從第一個 trial 起就在 RAG 打開的狀態下進行，「純 LLM 先最佳化」這個中間狀態從未存在 |
| 最佳化結果遺失 | 勝出模板只 `print` 到 console、未持久化；論文 Table VIII 宣稱的「最佳化後」prompt 正規化後等於未最佳化的基準 prompt |
| 逐筆預測未保存 | 審稿人要求的配對統計（McNemar、bootstrap CI）在數學上無法事後補算 |

## 設計核心

**Prompt 拆成兩個獨立組件**（`src/02_prompts.py`）：

```
指令段 instruction  —— 唯一被最佳化的對象，本身完全不提參考資料
參考段 reference    —— 固定格式、固定位置，只有接 RAG 的條件才掛上去
```

七個條件共用同一個 `assemble()`，所以「有沒有 RAG」與「指令有沒有被最佳化」是兩個真正
獨立的開關。這是舊設計做不到的。

**test 全程只量測、不做任何選擇。** 所有選型一律只看 val；基準指令在跑 S0 之前就定稿並
記錄 SHA256，看到結果再回頭改它等同於用 test 做選擇。

## 實驗條件

| Case | RAG | 語料庫 | 指令段 | 回答什麼 |
|---|---|---|---|---|
| S0 | 無 | — | 基準 | 真正的起點（原研究從未有過這一格） |
| S1 | 無 | — | 最佳化#1 | **S0→S1：prompt 最佳化本身的效果**，無 RAG 干擾 |
| S2a | 有 | 未擴增 | 基準 | **S0→S2a：RAG 本身的效果**，指令固定 |
| S2b | 有 | 未擴增 | 最佳化#1 | 最佳化在有 RAG 時還剩多少作用 |
| S3a | 有 | 擴增 | 基準 | **S2a→S3a：擴增語料庫的純效果**，指令固定 |
| S3b | 有 | 擴增 | 最佳化#1（遷移） | 重現「選型與語料庫狀態不對齊」的做法，診斷用 |
| S4 | 有 | 擴增 | 最佳化#2 | **S3b→S4：最佳化能否跨設定遷移** |

## 資料

原始資料集：Kaggle *Sentiment Analysis for Mental Health*（`Combined Data.csv`，53,043 筆、7 類）
SHA256 `67274f52dec3b0155aec4d7d1ff31d2a92b0e9b383ab917614591cd168cc2474`

沿用原研究的四類設定（Normal / Depression / Anxiety / Bipolar）；另三類
（Suicidal 10,653、Stress 2,669、Personality disorder 1,201，共 27%）未納入，
論文需明確交代此一取捨。

處理鏈與實測結果：

```
53,043  原始
38,520  取四類
37,215  逐字去重（-1,140 重複，-34 標籤衝突）
36,814  近似去重 cosine≥0.90（-375 摺疊，-26 標籤矛盾整組剔除）
 9,992  類別平衡，每類 2,498
        └─ train 5,996 / val_search 400 / val_confirm 1,100 / test 2,496
```

四個集合兩兩逐字重複為 0；切分後複驗，val/test 對 train 的最近鄰相似度上限
**0.8933**（去重前為 0.9979），無任何 ≥0.90 的近似重複。

val 拆成 search（TPE 迭代）與 confirm（最終選型）兩層，是為了避免前一輪實驗
val=222 過小導致最佳化過擬合、在 test 上顯著變差的問題。

## 執行

```bash
pip install -r requirements.txt   # torch 請用 CUDA 版，見檔內說明
ollama pull llama3.1

cd src
python 00_probe_env.py     --run MAIN   # 環境指紋
python 01_prepare_data.py               # 去重 → 平衡 → 四向切分 → 互斥驗證
python 02_prompts.py                    # 預覽組裝結果與候選集
```

後續步驟（`03`–`11`）依序為建語料庫、擴增、最佳化、跑條件、baseline、指標、統計、
表圖、驗收。**未通過 `11_verify_all.py`（0 FAIL）之前，不得引用任何數字。**
