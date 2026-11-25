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

## Git 提交與交接規範

- 採用 Conventional Commits 1.0.0，格式為 `<type>(<scope>): <description>`，scope 可省略。type 可用 feat、fix、docs、style、refactor、perf、test、build、ci、chore、revert。
- 主旨用英文祈使句，type 和 scope 小寫，最多 72 字元，句尾不加句點。描述實際改動。動機或細節需要交代時加 body，主旨、body 和 footer 之間各留空行。破壞性變更以 `!` 和 `BREAKING CHANGE:` footer 標記。
- 暫存前檢查 `git status` 和 `git diff`，提交前審查 staged diff 並執行相關測試。每個 commit 只含一項邏輯變更，排除無關修改、機密、憑證與暫存檔。保留 Git hooks，如實記錄測試結果與 issue 參照。
- amend、rebase、force-push 須先取得明確授權。工作分支可直接推送，預設或受保護分支須先取得明確授權。
- 每個重要階段完成後更新 `hw3/HANDOFF.md`，記錄任務目標、已完成及未完成事項、重要決策、測試指令與結果、下一步。新增的作業檔案放在 `hw3/`。

## 階段五：原始 WARC 到位

- Google Drive 下載成功，原檔已放在 `hw3/07.warc.gz`。
- `sha256sum 07.warc.gz` 的結果為 `60af4b06e8aaa6d74d2d2a4b7c3a9ce40b63db63d6a65c252b22b434818e4a6b`，與 lock 相符。接續重建並驗收 A1/A2。

## 階段六：真實資料重建

- `.venv/bin/python rebuild_index.py` 已沿用既有 lock 完成注入（29 秒），產出 `07_team27.warc.gz`。已開始建立索引，後續驗收結果待補。
- 重建輸出暫存於 `hw3/.venv/rebuild-real.log`。

- 2026-11-26：HW1 索引建立階段完成（370 秒），開始 prep。
- Prep 完成（338 秒），確認 `N = 31,480`。開始建立額外檢索資料表。
- 額外檢索資料表完成（75 秒），開始建置 doc store。

## 2026-11-26｜階段七：真實索引重建完成

- `.venv/bin/python rebuild_index.py` 成功，耗時 946 秒。Prep 讀到 31,480 個 response records，索引 `N = 31,480`。Doc store 耗時 134 秒，確認 31,480 列及 lock 指定 6 份文件均包含完整事實句。
- 重建驗證 6 項全通過。ltc.ltc 和 BM25 的前三組前兩筆 doc ID 分別為 `13639,23925`、`13964,10810`、`3350,3800`，均與 lock 所指定文件一致。
- `python search.py --trec-id -k 2 115598063` 和 `python search.py --trec-id -k 2 "zorvian lantern"` 的前兩筆 TREC-ID 都與 lock 一致。
- `sha256sum output/postings.txt` → `ac2ba927ea941b5df56e8283fe69bdac666534eb6245a0dc73138c5cf67c2fd1`。Windows canonical 檔尚未提供，跨機器比對待補。建置期間 NLTK 的可選 WordNet 下載遭環境代理設定拒絕，這次的索引設定未啟用 lemmatization，重建正常完成。
- 已開始 `.venv/bin/python -m pytest -q -rs`，等待真實索引測試結果。

## 2026-11-26｜階段八：真實索引驗收完成

- `.venv/bin/python -m pytest -q -rs` → **44 passed、1 skipped、0 failed**（96.79 秒）。三項 `index` 測試全通過，唯一跳過的是正式 `prompts/eval_prompts.jsonl` 的格式測試。
- 另從 lock 產生 6 個特殊 prompt，加上 P01–P06 編號後，用 `evaluate(..., k=5)` 在真實索引執行。Lexical 的 hit@1、hit@5 都是 **6/6**，此輪平均 42.8 ms／題。這是特殊題驗收，正式評估仍需 12 題 collection prompt。
- 下一步：完成 A3 的 18 題評估與報告，並與 B 線完成 Windows canonical 雜湊比對及整合驗收。

## 階段九：正式 A3 評估與 Windows 基準搜尋

- 目標：以真實索引產出 18 題正式評估，以 B 線 Windows canonical SHA-256 核對 Linux postings。
- 已 fetch 並查閱 GitHub 所有分支及可達歷史，尚未找到 `canonical_index.json`。先完成評估，再執行 Linux 端的比對入口並記錄結果。
- 12 題 collection 會從實際 doc store 選定答案與來源，固定為 8 題英文原用字、2 題英文換說法、2 題中文，於評估前定稿並保留未命中結果。

- 已依 doc store 的實際網頁定稿 12 題，涵蓋地方新聞、活動、軟體、觀光、課程、音樂等主題。`.venv/bin/python make_eval_prompts.py` 已產生 P01–P18，接續執行正式評估。

## 2026-11-26｜階段十：正式評估與雜湊比對狀態

- `TZ=Asia/Taipei .venv/bin/python evaluate_retrieval.py` 產生 `results/retrieval.md`。Lexical hit@1 為 14/18，hit@5 為 16/18，平均 40.6 ms／題。
- 特殊題 6/6、英文原用字 6/8、英文換說法 2/2、中文 0/2 命中第一筆。四組 hit@5 分別為 6/6、8/8、2/2、0/2。已回填 `A_LINE_README.md`。
- `.venv/bin/python rebuild_index.py --from check` 再驗證六項排名通過，回報缺少 `canonical_index.json`。目前執行環境是 Linux。未在專案的 GitHub 分支或本機找到 Windows 基準。
- Windows 基準待補：在 Windows 使用同份 A-line 原始 WARC 與 lock，從 `hw3/` 執行 `python rebuild_index.py --canonical`。取得該次產出的 `hw3/canonical_index.json` 後，在 Linux 執行 `python rebuild_index.py --from check` 並核對完整 `postings_sha256`。
- 已核對 18 題、18 份來源文件的答案證據。`.venv/bin/python -m pytest -q -rs` → 45 passed（80.17 秒），含正式題集格式與真實索引測試。
