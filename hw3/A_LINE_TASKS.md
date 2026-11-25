# HW3 A 線（資料 + 檢索）可交付工作清單

> 負責人：陳長壻（115598095）。依據：`PLAN.md` v2（Track A、F1、F2）與 `CONSENSUS.md`。
> 分支：`claude/lucid-knuth-0c90yl`。程式碼目錄：`hw3/`（和 `hw2/` 一樣直接放在作業資料夾，不另開子資料夾）。
> 分工方式：WARC 太大，無法上傳到雲端 session。所以程式和測試都在雲端用**合成資料**完成；需要真實 `07.warc.gz` 的步驟標為 **[本機]**，由陳長壻在自己的電腦照下面的「本機執行手冊」跑。
> 打勾規則：程式 + 測試通過 + commit 才算完成。

## 前置：Task 0 stub（原屬 B 線，暫代）

B 線的 Task 0 尚未推上任何分支。為了讓 A 線能測試，這個分支先放一份**完全照凍結契約寫的最小版本**，獨立 commit（`6c69647`），交給鄧存正審查或整份替換。

- [x] `hw3/` 放 `hw2/` 程式的複本（不含 `results/`、`output/`、`README*`、notebook）
- [x] `rag/types.py`：`Passage`、`Retriever`、`ChatModel`、`Embedder`、`LLMUnavailable`、`StaticRetriever`
- [x] `rag/config.py`、`rag/registry.py`、`rag_config.json`、`pytest.ini`（markers `index`、`llm`）、`tests/conftest.py`、`tests/test_contract.py`
- [x] `requirements.txt` 加 `numpy`、`pytest`

## A1：嵌入特殊 token、重建 index

- [x] `special_tokens.json`：三個 token（`115598095`、`115598063`、`zorvian lantern`）＋虛構事實
- [x] `inject_tokens.py`：非目標 record 逐 byte 不變；事實插在 `<body…>` 後（否則 `</head>` 後，否則 payload 開頭）；改寫 `Content-Length`；有 lock 檔就沿用 doc id，否則 `Random(27)` 挑選；輸出的 gzip 每次都相同（mtime 0）
- [x] `rag/doc_store.py`：`build_doc_text()` 建 `doc_text.tsv`（TREC-ID 須和 index 一致）
- [x] `rebuild_index.py`：inject → `hw1_indexer.py` → `prep.py` → `prep_extra.py` → `doc_text` → check（postings SHA-256、每個 token 的 2 份文件在 ltc.ltc 和 BM25 都排前 2）
- [x] 測試：`tests/test_inject.py`（合成 WARC）、`tests/test_rebuild.py`（在 40 頁合成 WARC 上跑完整的 `rebuild_index.py`）
- [x] commit `e7d4ca4`
- [ ] **[本機]** 跑 `rebuild_index.py`，commit `special_tokens.lock.json`（手冊步驟 2–3）
- [ ] **[本機]** 驗證 `N = 31,480`，兩個 `search.py` 查詢的前 2 筆是 lock 檔的 doc（手冊步驟 4）
- [ ] 通知鄧存正：pull 之後本機重建（**必須在 lock 檔 commit 之後**，否則兩台電腦會各自挑出不同的頁面）

## A2：Lexical retriever（方法 2）

- [x] `sanitize_prompt`：`" “ ” ( ) / * ?` → 空白；獨立的 `AND`/`OR`/`NOT` 轉小寫；結果一定是 free 模式
- [x] `DocStore`：`doc_text.tsv` 依 byte offset 隨機讀取；`window()` 沿用 HW2 `ui.doc_text` 的規則，再往前退到句首；回傳 `highlights`
- [x] `LexicalRetriever(cfg)`：BM25（`k1`、`b`、`passage_words` 取自設定）+ impact ordering，每筆命中一個 `Passage`
- [x] 測試：`tests/test_lexical.py`（合成文字 + 合成 index；另有 2 個 `index` 測試等真實 index）
- [x] commit `a51c78c`
- [ ] **[本機]** `python -m pytest -v` 在真實 index 上全部通過，`index` 測試不再 skip（手冊步驟 5）
- [ ] 通知鄧存正把設定改成 `"retriever": "lexical"`

## A3：評估 prompt 集與檢索評估

- [x] `evaluate_retrieval.py`：每個建得出來的 retriever 的 hit@1、hit@k、平均 ms，依 `type`、`lang` 分組；沒建的列為 "not built"（commit `7980c61`）
- [x] `make_eval_prompts.py`：從 lock 檔自動產生 6 個特殊 prompt（token 單獨輸入 + 問句），合併手寫的 12 個 collection prompt，編號 P01–P18，並檢查數量規則
- [x] `python -m rag.doc_store show <doc_id>`：寫 prompt 時讀網頁用（輸出開頭和投影片第 9 頁相同：`Document ID# … (trec_id) url`）
- [ ] **[本機]** 寫 `prompts/collection_prompts.jsonl`：12 個，主題分散，挑 7B 模型不太可能知道的事實；其中 2 個 `"lang": "zh"`、2 個 `"reworded": true`（手冊步驟 6）
- [ ] **[本機]** `python make_eval_prompts.py` → `prompts/eval_prompts.jsonl`
- [ ] **[本機]** `python evaluate_retrieval.py` → `results/retrieval.md`；預期特殊 prompt hit@5 = 6/6，換說法／中文的結果照實記錄
- [ ] **[本機]** commit `feat(hw3): eval prompt set and retrieval results`
- [ ] 交給鄧存正：`eval_prompts.jsonl`，以及 `A_LINE_README.md`（把數字填好）

## 本機執行手冊

需要：Python 3.10 以上、`07.warc.gz`、約 2 GB 磁碟空間、約 15–20 分鐘。指令在 Windows（PowerShell）和 Linux 都一樣。

**1. 取得程式、放好 WARC。**

```
git fetch origin claude/lucid-knuth-0c90yl
git checkout claude/lucid-knuth-0c90yl
cd hw3
pip install -r requirements.txt
python -m pytest -q                   # 預期 40 passed, 4 skipped
```

把 `07.warc.gz` 放在 `hw3/07.warc.gz`（和 `rebuild_index.py` 同一層；`.gitignore` 已排除）。

**2. 重建。**

```
python rebuild_index.py
```

- 第一次執行時沒有 lock 檔：`inject` 會先掃一次 WARC 挑選目標頁（約 4 分鐘），再寫出 `07_team27.warc.gz` 和 `special_tokens.lock.json`。
- 之後依序是 index（約 4–5 分鐘）、prep（約 4–5 分鐘）、extra、doc_text（約 4 分鐘）、check。
- 中斷時可以從某一步繼續，例如 `python rebuild_index.py --from prep`。
- 最後應該印出 6 行 `[check] OK` 和 `[all] finished`。如果出現 `FAIL`，**不要調參數繞過**：把輸出記下來，這是 Major difficulties 的材料。
- `no canonical_index.json to compare with yet`，或是 SHA 不同的 WARNING，都是預期內的訊息。canonical build 由鄧存正在 Windows 上用 `--canonical` 產生。

**3. 立刻 commit lock 檔。**

```
git add special_tokens.lock.json
git commit -m "chore(hw3): pin special token target docs"
git push
```

push 之後才請鄧存正 pull 再重建。lock 檔推上去之前他就重建的話，他的電腦會自己重新挑選，兩台電腦就會各自挑出不同的目標頁。

**4. 驗證。**

- prep 的輸出有 `response records: 31,480` 和 `N = 31,480`。
- `python search.py 115598063 --trec-id` 和 `python search.py "zorvian lantern" --trec-id`：前 2 筆就是 lock 檔裡該 token 的 `trec_ids`。
- `python -m rag.doc_store show <lock 檔的 doc id>`：內文開頭就是事實句。

**5. 測試。**

```
python -m pytest -v
```

`index` 測試（`test_special_token_rank1`、`test_stopword_only_prompt_returns_empty`、`test_refuses_token_already_in_index`）應該從 skip 變成 PASS。`llm` 測試會繼續 skip。

**6. 寫 12 個 collection prompt。**

0. 先產生範本：`python make_eval_prompts.py --init`。它會建立 `prompts/collection_prompts.jsonl`，內含 12 行 `TODO`：前 8 行英文、第 9–10 行換說法（`"reworded": true`）、第 11–12 行中文。已存在時不會覆蓋。之後把每一行換成真的 prompt。
1. 先用 `python -m rag.lexical "<主題關鍵字>"` 找頁面（和 RAG 一樣用 BM25，會印出標題和 passage）。
2. 用 `python -m rag.doc_store show <doc_id> --words 0` 讀全文，挑一個具體事實，例如日期、數字、地名或產品細節。
3. 把範本的一行換成一筆：

```
{"lang": "en", "prompt": "In what year did ... ?", "expected_docs": [4821], "expected_answer": ["1987"], "reworded": false}
```

- `expected_answer`：答案裡只要出現其中任何一個字串就算答對（不分大小寫）。可以列幾種寫法，例如 `["1987", "nineteen eighty-seven"]`。
- 換說法的 2 個：改用網頁裡沒有的詞，例如網頁寫 "premiere"，問句就寫 "first performed"。
- 中文的 2 個：用中文問英文網頁上的事實，`expected_answer` 填英文或數字。
- 每寫一個，就用 `python -m rag.lexical "<prompt>" --expect <doc_id>` 看它的排名。排名不好也照樣保留，評估本來就要呈現 lexical 的弱點。

**7. 產生 prompt 集、評估、commit。**

```
python make_eval_prompts.py
python evaluate_retrieval.py
python -m pytest -q
git add prompts/collection_prompts.jsonl prompts/eval_prompts.jsonl results/retrieval.md
git commit -m "feat(hw3): eval prompt set and retrieval results"
git push
```

接著把 `results/retrieval.md` 的數字填進 `A_LINE_README.md`，再通知鄧存正。

## 交接給 B 線（資料，不換程式）

| 時間點 | 交付 |
|---|---|
| A1 完成（lock 檔 commit 後） | `inject_tokens.py`、`rebuild_index.py`、`special_tokens.lock.json` |
| A2 完成 | `rag/lexical.py`（B 設 `"retriever": "lexical"`） |
| A3 完成 | `prompts/eval_prompts.jsonl` |
| B5 | `A_LINE_README.md`：檢索 / index 章節文字 + Major difficulties |

## 選做（B4 產出評分用輸出後才做）

- [ ] F1 vector retriever（`rag/vector.py`，`bge-m3`，需要 Ollama）
- [ ] F2 hybrid retriever（`rag/hybrid.py`，RRF）
