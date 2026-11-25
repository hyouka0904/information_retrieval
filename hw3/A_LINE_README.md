# A 線提供的 README 文字（交給鄧存正，B5 合併）

> `【填】` 標記的地方，在本機跑完 `rebuild_index.py` 和 `evaluate_retrieval.py` 之後填入。

## 特殊 token 與 index 重建

- 三個 token 各配一句**虛構**的事實，確保 LLM 不用 RAG 就不可能答對：`115598095`（codename BLUE-HERON-7）、`115598063`（AMBER-LYNX-3）、`zorvian lantern`（Zorvian Lantern Festival：每年 3 月 14 日在基隆）。
- `inject_tokens.py` 把每句事實包成 `<p>`，插在 `07.warc.gz` 中 2 個既有網頁的 `<body>` 後面，並改寫該 record 的 `Content-Length`；其他 byte 全部不變，文件數維持 31,480。目標網頁由 `Random(27)` 挑選：utf-8／iso-8859-1／ascii 編碼、有 `<body>`、內文至少 300 字，且不重複使用。挑選結果固定在 `special_tokens.lock.json`，所以每台電腦都會插入相同的網頁。
- 目標網頁：【填：lock 檔的 doc id / TREC-ID】。
- `rebuild_index.py` 用和 HW2 完全相同的 HW1 設定重建 index（含 title、hyphen split、case folding、去 stopword、Porter stemming），並產生 `doc_text.tsv`（每份文件的標題與內文前 2 萬字，給 passage 用）。最後會檢查每個 token 的 2 份文件在 ltc.ltc 和 BM25 都排在前 2。
- 重建時間：【填：各步驟秒數，CPU / OS】。

## 檢索（方法 2：lexical）

- Prompt 先經過 `sanitize_prompt`：去掉 `" ( ) / * ?`，獨立的大寫 `AND`/`OR`/`NOT` 轉小寫，所以自然語言的問句一律當成 free text，不會誤觸 HW2 的 phrase／proximity／wildcard／Boolean 語法（例如句尾的 `?` 原本會被當成 wildcard）。
- 排序用 HW2 的 `Ranker`：BM25（k1 = 1.2、b = 0.75）加上 impact ordering（結果和完整計算相同，只是比較快）。
- 每份命中的文件取一段 150 字的 passage：沿用 HW2 網頁 demo 的規則，從 query term 開始、相異 term 數最多、其次命中數最多的視窗，再往前退到句首，避免事實句的主詞被切掉。passage 附上要標示的字（`highlights`），輸出時標成 `**word**`。
- 所有 retriever 都實作同一個 `Retriever` 介面，透過 `rag_config.json` 的 `"retriever"` 或 `--retriever` 切換；之後加做 vector／hybrid 只要新增一個模組。

## 檢索評估（`results/retrieval.md`）

18 個 prompt：6 個特殊 token（每個 token 單獨輸入一次、用問句問一次）＋ 12 個根據實際網頁內容寫的 collection prompt（其中 2 個換說法、2 個用中文）。

| 組別 | n | hit@1 | hit@5 |
|---|---|---|---|
| special | 6 | 【填】 | 【填】 |
| collection（英文、原用字） | 8 | 【填】 | 【填】 |
| collection（換說法） | 2 | 【填】 | 【填】 |
| collection（中文） | 2 | 【填】 | 【填】 |

平均檢索時間：【填】ms／prompt。

## Major difficulties（A 線）

- **自然語言 prompt 會觸發 HW2 的查詢語法**：句尾的 `?` 是 wildcard，引號、括號、`/3`、大寫 AND/OR/NOT 會切換成 phrase 或 Boolean 模式，結果可能是空的或完全不同。解法是在檢索前先清理 prompt，並用測試確認清理後一律是 free 模式。
- **插入的事實必須出現在 LLM 看到的 passage 裡**：`doc_text.tsv` 只保留內文前 2 萬字，passage 又只取 150 字。所以事實插在 `<body>` 一開頭，passage 視窗也會往前退到句首，並用測試確認 token 的 passage 含有該 token。
- **兩台電腦要插入相同的網頁**：挑選目標頁時要先解析網頁，而 HW2 發現 lxml 在 Windows 和 Linux 上抽出的文字略有不同，所以由一台電腦挑選後把 doc id 寫進 lock 檔，另一台只沿用。插入後的 WARC 也逐 byte 相同（gzip 的 mtime 固定為 0）。
- **WARC 檔太大**：無法放進 repo，也無法上傳到雲端開發環境，所以程式全部用手寫的合成 WARC 測試（包括在 40 頁的合成 WARC 上跑完整的重建流程），真實資料的步驟另外在本機執行。
- **Lexical 檢索的限制**：換說法和中文的 prompt 和網頁沒有共同的詞，所以檢索失敗（【填：實際數字】）。這是之後加做方法 1（vector）／方法 3（hybrid）的理由。
- 【填：本機執行時遇到的其他問題】

## 組員分工（A 線部分）

陳長壻：特殊 token 設計與嵌入（`inject_tokens.py`）、index 重建（`rebuild_index.py`）、passage 擷取（`rag/doc_store.py`）、lexical retriever（`rag/lexical.py`）、評估 prompt 集與檢索評估（`make_eval_prompts.py`、`evaluate_retrieval.py`）。
