# HW3 A 線交接

## 2026-11-25｜階段一：分支與計畫盤點

- 任務目標：核對 A 線與 HW3 計畫，實跑現有測試，整理真實資料驗收和 B 線整合待辦。
- 已完成：取得 `A-line`（`833ba20`）。A1 的注入、重建和 doc store、A2 的 lexical retriever、A3 的 prompt 工具和 evaluator 均已進分支。目標 lock 檔亦已追蹤，工作清單的 lock 提交待辦需依現況判讀。
- 未完成：真實 WARC 重建與測試結果尚待查證。分支未追蹤 collection prompts、正式評估集與檢索報告。
- 重要決策：依 `PLAN.md` 先驗收 lexical（方法 2）。F1 vector、F2 hybrid 為選做。
- 測試：下一階段執行 `python -m pytest -q -rs`，確認通過項目與跳過原因。

## 2026-11-25｜階段二：核對交付物與驗收條件

| 工作 | 已進 A-line | 待驗收或交付 |
|---|---|---|
| A1 特殊 token 與索引 | WARC 注入、重建、doc store。commit `67d21bb` 已固定 3 個 token 的 6 份目標文件 | 真實 WARC 的文件數、TREC-ID／內文對齊、兩種排序、重建時間及 canonical postings 雜湊 |
| A2 lexical（方法 2） | BM25 + impact ordering、prompt 清理、passage 擷取與 CLI。此分支的 `rag_config.json` 已選 `lexical` | 真實索引上的排名與 passage，及 B 線用相同 lock 和設定的整合 |
| A3 評估 | prompt 範本／驗證、6 個特殊 prompt 的產生器、hit@1／hit@k 與時間報告 | 根據原始網頁寫 12 個 collection prompt，產出正式 18 題及 `results/retrieval.md`，回填 `A_LINE_README.md` |
| F1 / F2 | 計畫已定義介面 | vector／hybrid 尚未實作，依計畫等 B4 完成後再做 |

- 本工作區沒有 `07.warc.gz`、`output/meta.json` 或 `canonical_index.json`。沿用已提交的 lock，原始 WARC 的 SHA-256 必須與 `warc_sha256` 一致。
- 測試環境：初次執行缺少 pytest。已在 `hw3/.venv` 使用 Python 3.12.14，並成功安裝 `requirements.txt`。全套測試執行中。
- 報告依 type、lang 分組並列出逐題排名。英文原用字與換說法的統計需依 prompt 的 `reworded` 欄位從逐題結果彙整，才能填入交付 README 的兩列。

### 真實資料與 B 線下一步

以下指令從 `hw3/` 使用已安裝依賴的 Python 執行。

1. 放入原始 `07.warc.gz`。以 `python rebuild_index.py` 重建，記錄各階段耗時、環境、`N = 31,480`、六行 `[check] OK`（3 個 token × ltc.ltc／BM25）及 postings SHA-256。B 線在 Windows 依計畫用 `python rebuild_index.py --canonical` 建立基準，記錄跨機器比較結果。
2. 用 `python search.py --trec-id -k 2 115598063` 和 `python search.py --trec-id -k 2 "zorvian lantern"` 核對前兩筆 TREC-ID。用 `python -m rag.doc_store show <doc_id>` 核對 6 份文件的事實句，再用 `python -m rag.lexical "<token>" --expect <doc_id>` 查排名與 passage。執行 `python -m pytest -q -m index -rs`。
3. 用 `python make_eval_prompts.py --init` 建立範本。從實際網頁選 12 個事實，配置 8 題英文原用字、2 題英文換說法、2 題中文，逐題留下預期文件與答案。用 `python make_eval_prompts.py` 產生 P01–P18，再跑 `python evaluate_retrieval.py` 與 `python -m pytest -q -rs`。
4. 核對特殊 prompt 的 hit@5 是否達 6/6，記錄英文換說法與中文的實測結果及平均檢索時間，回填 `A_LINE_README.md`。交接 B 線時用同一份 lock、`lexical` 設定及正式評估集跑 B3/B4 的 RAG 輸出與答案評估。本次盤點範圍是 A-line 的來源與測試，B 線整合及通知狀態待確認。

## 2026-11-25｜階段三：測試完成

- `.venv/bin/python -m pytest -q -rs` → **41 passed、4 skipped、0 failed**（101.61 秒）。
- 40 頁合成 WARC 的端到端重建通過，含 ltc.ltc／BM25 六項排名檢查及 doc store 對齊。注入的 byte 與 lock 檢查、lexical 搜尋及 CLI、prompt 產生與評估邏輯的現有測試也通過。
- 真實索引未建，跳過 `test_refuses_token_already_in_index`、`test_stopword_only_prompt_returns_empty`、`test_special_token_rank1`。正式 prompt 檔尚未產出，跳過 `test_eval_prompt_file_well_formed`。
- 目前瓶頸為取得與 lock 雜湊一致的原始 WARC，之後依序完成真實索引驗收、正式 18 題評估及 B 線整合。

## 2026-11-25｜階段四：原始 WARC 交付檢查

- 目標：使用新上傳的 `07.warc.gz` 完成 A1/A2 的真實索引驗收，供 A3 正式評估使用。
- 已完成：確認附件已提供。檔案傳輸工具回報超過 32 MiB 上限，工作區和共享下載目錄均無原檔。
- 未完成：原檔 SHA-256、真實 WARC 重建、三項 index 測試及正式 18 題評估仍待執行。上一階段的 41 通過、4 跳過是合成資料測試結果。
- 決策與下一步：取得可在執行環境讀取的原始檔後，放在 `hw3/07.warc.gz`，用 `sha256sum 07.warc.gz` 比對既有 lock 的 `warc_sha256`，再照前述重建及驗收順序執行。WARC 與索引輸出使用既有 Git 忽略規則。
