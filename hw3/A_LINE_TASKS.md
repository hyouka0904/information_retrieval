# HW3 A 線（資料 + 檢索）可交付工作清單

> 負責人：陳長壻（115598095）。依據：`PLAN.md` v2（Track A、F1、F2）與 `CONSENSUS.md`。
> 分支：`claude/lucid-knuth-0c90yl`。程式碼目錄：`hw3/team27_hw3/`，WARC 放在 `hw3/07.warc.gz`（不進 git）。
> 打勾規則：程式 + 測試通過 + commit 才算完成；需要真實 WARC 的步驟另外標 **[WARC]**。

## 前置：Task 0 stub（原屬 B 線，暫代）

B 線的 Task 0 尚未推上任何分支。為了讓 A 線能測試，這個分支先放一份**完全照凍結契約寫的最小版本**，獨立 commit，交給鄧存正審查或整份替換。

- [ ] `team27_hw3/` = `hw2/` 的複本（不含 `results/`、`output/`、`README*`、notebook）
- [ ] `rag/types.py`：`Passage`、`Retriever`、`ChatModel`、`Embedder`、`LLMUnavailable`、`StaticRetriever`（照 PLAN Task 0）
- [ ] `rag/config.py`：`load_config()`，補預設值；未知 `llm.backend` → `ValueError`
- [ ] `rag/registry.py`：`make_retriever`、`make_llm`、`available_retrievers`
- [ ] `rag_config.json`、`pytest.ini`（markers `index`、`llm`）、`tests/conftest.py`、`tests/test_contract.py`
- [ ] `requirements.txt` 加 `numpy`、`pytest`

## A1：嵌入特殊 token、重建 index

交付物：`special_tokens.json`、`inject_tokens.py`、`rebuild_index.py`、`rag/doc_store.py`（build）、`tests/test_inject.py`；產出 `../07_team27.warc.gz`、`special_tokens.lock.json`（commit）、重建的 `output/`（含 `doc_text.tsv`）。

- [ ] `special_tokens.json`：三個 token（`115598095`、`115598063`、`zorvian lantern`）＋虛構事實
- [ ] 測試（5 筆合成 WARC）：`test_inject_preserves_record_count`、`test_injected_fact_in_body_prefix`、`test_other_docs_unchanged`、`test_content_length_updated`、`test_refuses_token_already_in_index`（`index`）
- [ ] `inject()`：逐 byte 保留非目標 record；目標插在 `<body…>` 後（否則 `</head>` 後，否則 payload 開頭）；改寫 `Content-Length`；有 lock 檔就沿用 doc id，否則 `Random(27)` 挑每個 token 2 份（utf-8/iso-8859-1/ascii、≥ 300 字、不重複）
- [ ] `build_doc_text()` + `python -m rag.doc_store build --warc PATH`：`title\tbody[:20000]`，第 n 行 = doc n，TREC-ID 須和 index 一致
- [ ] `rebuild_index.py`：inject → `hw1_indexer.py` → `prep.py` → `prep_extra.py` → `doc_store build`；`--canonical` 寫 `canonical_index.json`，否則比對 SHA（不符只警告）；檢查每個 lock entry 的 doc 在該 token 排第 1（否則 exit 1）
- [ ] **[WARC]** 跑 `rebuild_index.py`，產生並 commit `special_tokens.lock.json`
- [ ] **[WARC]** 驗證：`response records: 31,480`、`N = 31,480`；`search.py 115598063 --trec-id`、`search.py "zorvian lantern" --trec-id` 前 2 筆就是 lock 檔的 doc
- [ ] commit `feat: inject team27 special tokens, rebuild script` → 通知鄧存正本機重建

## A2：Lexical retriever（方法 2）

交付物：`rag/lexical.py`、`rag/doc_store.py`（讀取端：`DocStore`）、`tests/test_lexical.py`。

- [ ] 測試：`test_sanitize_prompt_question_mark`、`test_sanitize_prompt_operators`、`test_window_picks_term_cluster`、`test_window_no_terms_returns_prefix`、`test_stopword_only_prompt_returns_empty`（`index`）、`test_special_token_rank1`（`index`）
- [ ] `sanitize_prompt`：`" ( ) / * ?` → 空白；獨立的 `AND`/`OR`/`NOT` 轉小寫；結果一定是 free 模式
- [ ] `DocStore`：`doc_text.tsv` byte offset 隨機讀取；`title`、`body`、`terms_of`（`lru_cache`）、`window`（沿用 HW2 `ui.doc_text` 規則：從 query term 開始，相異 term 數優先，再比命中數）
- [ ] `LexicalRetriever(cfg)`：BM25（`k1`、`b` 取自設定）+ `impact=True`；每筆命中回傳一個 `Passage`（含 `highlights`）
- [ ] **[WARC]** `index` 測試在真實 index 上通過
- [ ] commit `feat: lexical retriever over HW1 index (Method 2)` → 通知鄧存正把設定改成 `lexical`

## A3：評估 prompt 集與檢索評估

交付物：`prompts/eval_prompts.jsonl`、`evaluate_retrieval.py`、`tests/test_evaluate_retrieval.py`、`results/retrieval.md`。

- [ ] 測試：`test_hit_at_k`（StaticRetriever，hit@1 = 0.5、hit@5 = 1.0）、`test_iterates_available_retrievers`（未建的列為 "not built"）
- [ ] `evaluate(prompts, retrievers, k=5)`：每個 retriever 的 hit@1、hit@5、平均 ms，依 `type`、`lang` 分組；輸出 markdown
- [ ] 6 個特殊 prompt：每個 token 單獨輸入 + 問句（`expected_docs` 取自 lock 檔）
- [ ] **[WARC]** 12 個 collection prompt：讀實際網頁、記 doc id，主題分散；其中 2 個換說法、2 個中文
- [ ] **[WARC]** 跑 `python evaluate_retrieval.py` → `results/retrieval.md`；預期特殊 prompt hit@5 = 6/6，換說法／中文照實記錄
- [ ] commit `feat: eval prompt set and retrieval evaluation`
- [ ] 交給鄧存正：`eval_prompts.jsonl`、README 文字（index 重建、token 嵌入、lexical 檢索、數字）、A 線的「Major difficulties」

## 交接給 B 線（資料，不換程式）

| 時間點 | 交付 |
|---|---|
| A1 完成 | `inject_tokens.py`、`rebuild_index.py`、`special_tokens.lock.json` |
| A2 完成 | `rag/lexical.py`（B 設 `"retriever": "lexical"`） |
| A3 完成 | `prompts/eval_prompts.jsonl` |
| B5 | `A_LINE_README.md`：檢索 / index 章節文字 + Major difficulties |

## 選做（B4 產出評分用輸出後才做）

- [ ] F1 vector retriever（`rag/vector.py`，`bge-m3`，需要 Ollama）
- [ ] F2 hybrid retriever（`rag/hybrid.py`，RRF）
