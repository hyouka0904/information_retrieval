# HW3（LLM + RAG）：需討論的

> 狀態：2026-11-25 已達成共識，陳長壻同意全部提案。

---

## 1. 先做哪一種 RAG 方法

投影片列了三種方法，以下是針對我們的情況做的比較：

| | 方法 1：vector DB | **方法 2：lexical（HW2 的搜尋）** | 方法 3：hybrid（1 + 2，排名融合） |
|---|---|---|---|
| 沿用 HW1/HW2 | 只用到網頁文字 | 全部（index、BM25、`--impact`） | 兩者都用 |
| 額外建置 | 約 10 萬段文字做 embedding：3060 上約 10–30 分鐘，約 200–400 MB | 無 | 同方法 1 |
| prompt 是特殊 token（`115598063`） | 弱：embedding 對罕見數字不敏感 | **完全比對，排第 1** | 好 |
| prompt 換了說法、和網頁用字不同 | 強 | 較弱 | 強 |
| 用中文問英文網頁的內容 | 好（多語 embedding 模型） | 找不到 | 好 |
| 對最低要求分數的風險 | 中 | **低** | 中 |

**提議：先做方法 2。** 最低要求檢查的是「引用 collection 內容的 prompt」和「含特殊 token 的 prompt」，lexical 搜尋對這兩種都很可靠，而且用的是我們已經驗證過的程式。方法 1、3 列為之後的選做項目（計畫中的 F1、F2），等必做的部分完成後再做。評估時會量測「換說法」和「中文」的 prompt，所以如果之後加做，可以用數字說明改善了什麼。

## 2. 為之後的修改預留的介面

所有東西都透過一個共用的小檔案 `rag/types.py` 串起來：

```python
@dataclass(frozen=True)
class Passage:
    doc_id: int; trec_id: str; url: str; title: str
    text: str                        # 放進 LLM prompt 的內容
    score: float; source: str        # source = retriever 名稱
    highlights: tuple[str, ...] = () # 輸出時要標示的字（由 retriever 填）

class Retriever(Protocol):
    name: str
    def search(self, prompt: str, k: int) -> list[Passage]: ...   # 依分數排序，沒有結果時回傳 []

class ChatModel(Protocol):
    model: str
    def chat(self, messages: list[dict]) -> str: ...
    def check(self) -> None: ...     # 丟出 LLMUnavailable("ollama pull ...") 等

class Embedder(Protocol):            # 之後做方法 1 時用
    def embed(self, texts: list[str], kind: Literal["query", "document"]) -> np.ndarray: ...
```

- 用一張對照表把名稱對到 class：retriever 有 `"lexical"`、`"vector"`、`"hybrid"`，LLM 後端有 `"ollama"`。每個 class 都用 `Cls(cfg)` 建立。
- **換方法或換模型，只要改 `rag_config.json`，或加 `--retriever` / `--model` 參數，不用改程式。**
- 之後加做方法 1 或 3，只是多一個新檔案，不會動到 pipeline、CLI，也不會動到對方的檔案。

## 3. LLM 與執行環境

| 設定 | 提議 | 其他選擇（只要改設定） |
|---|---|---|
| 執行環境 | Ollama，`http://localhost:11434` | llama.cpp server／vLLM（新增一個後端 class） |
| 對話模型 | `qwen2.5:7b`（Q4，約 4.7 GB VRAM；英文、中文都不錯） | `llama3.1:8b`、`qwen3:8b`（要關掉 thinking）、`gemma3:12b`（只有 12 GB 的卡跑得動） |
| Embedding 模型（只有方法 1 用） | `bge-m3`（多語） | `nomic-embed-text`（需要 query/doc 前綴，已經是設定項目） |
| 生成參數 | `temperature 0`、`seed 27`、`num_ctx 8192` | |

`temperature 0` 加上固定的 seed 應該能跑出相同輸出。

已決定：對話模型用 `qwen2.5:7b`。還沒確認：兩人的 Python 版本和 OS。

## 4. 特殊 token

評分會用到「含有嵌入在你們負責的文件集中的特殊 token」的 prompt。**我們目前還沒有放任何 token**。提議：

- **三個 token**：兩個學號，加上一個虛構的詞，每個都搭配一個虛構的事實，確保 LLM 不用 RAG 就不可能答對：
  - `115598095`：「…the project codename for 115598095 is BLUE-HERON-7.」
  - `115598063`：「…the project codename for 115598063 is AMBER-LYNX-3.」
  - `zorvian lantern`：「The Zorvian Lantern Festival is held every year on March 14 in Keelung, where 2,718 paper lanterns are released over the harbor.」
- **做法**：把每個事實包成 `<p>`，插在 `07.warc.gz` 中 **2 個既有網頁**的 `<body>` 後面。網頁由固定的亂數種子挑選。文件數維持 31,480，其他網頁逐 byte 不變。接著用和 HW1 完全相同的設定重建 index。
- **另一種做法**：改成新增幾個合成網頁（N 變成 31,483）。程式比較好寫，但比較不像「嵌入在文件集中」。

## 5. 分工

| | 陳長壻：**A 線，資料 + 檢索** | 鄧存正：**B 線，生成 + 介面** |
|---|---|---|
| 必做 | A1 嵌入 token + 重建 index 的程式 · A2 lexical retriever（方法 2）+ 段落擷取 · A3 評估用的 prompt 集 + 檢索評估 | 0 建 repo + `types.py` + 設定檔 · B1 Ollama client · B2 組 prompt + 引用來源 + pipeline · B3 CLI，顯示「不用／使用 RAG」對照 · B4 輸出結果 + 答案評估 · B5 打包 + 文件 |
| 選做 | F1 vector retriever（方法 1）· F2 hybrid（方法 3） | B6 網頁 demo |

## 6. 評估用的 prompt 集

- **6 個特殊 token prompt**：每個 token 各一個單獨輸入（像投影片第 9 頁）和一個問句。
- **12 個 collection prompt**：每個都根據某個網頁上的事實來寫（記下它的 doc id），主題分散，盡量挑 7B 模型不太可能知道的事實。
  - 其中 2 個**換掉網頁原本的用字**、2 個**用中文問**。這 4 個用來顯示 lexical 搜尋的弱點，也是之後做方法 1／3 的理由。
- 交件的輸出結果用全部 18 個。答案含有預期的字串就算答對。
- 負責人：A 寫 prompt（需要讀網頁、記 doc id），B 跑產生輸出結果和評估答案的程式。

## 7. 輸出格式

- 每個答案都顯示 **=== Without RAG ===** 和 **=== With RAG (lexical) ===**，用相同的模型和設定。作業要求要呈現兩者的差異。
- 來源依照投影片第 9 頁：`Document ID# 4821 (clueweb09-en0000-07-04820) http://…`，接著是 `…**115598063**…`。
- LLM 要標註 `[1]`、`[2]`…；文件裡沒有答案時，要回答「The retrieved documents do not contain the answer.」。
- 交件的「輸出截圖」：`results/snapshots.md`（所有 prompt）加上 3–4 張終端機截圖。

## 8. 交件與打包

- **上傳大小限制**：和 HW2 一樣是單檔（最大 64MB）。
- **壓縮檔內容**：程式、設定檔、`special_tokens*.json`、prompt、結果、文件，以及和 HW2 一樣壓縮過的 index（`postings.delta.xz`）。`doc_text.tsv`（145 MB）由 `rebuild_index.py` 從 WARC 重建，除非壓縮後放得下。
- **一頁說明文件**：完全照投影片第 10 頁的四個段落。組員表寫出每個人實際做的部分。

## ９. 進度 & Bonus

| 里程碑 | 內容 |
|---|---|
| M0 | 這份文件達成共識；repo + `types.py`（Task 0） |
| M1 | 嵌入 token，兩台電腦都重建好 index（A1）；Ollama client（B1） |
| M2 | Lexical retriever（A2）；pipeline + CLI 先接假的 retriever（B2、B3） |
| M3 | Prompt 集 + 檢索評估（A3）；輸出結果 + 答案評估（B4） |
| M4 | 打包 + 一頁說明文件（B5）；在全新環境測試 |
| 選做 | F1／F2／B6，M4 確定沒問題之後才做 |

**選做部分要做哪些？** F1 vector、F2 hybrid、B6 網頁 demo，或都不做。
