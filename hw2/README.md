# IR HW2：Query Processing and Searching

> 一頁的說明文件是 `README.pdf`。這份是完整的技術說明。

接續 HW1 的 positional inverted index，對使用者的 query 回傳依 TF-IDF + cosine similarity 排序的結果（投影片第 4、5 頁）。index 由 HW1 原封不動的 `hw1_indexer.py` 從 WARC 建出，前處理（`prep.py` 或 `hw2_index_prep.ipynb`）再補上 HW2 需要的統計量；之後的查詢都是讀這些小檔案，postings 則依 byte offset 從硬碟隨讀隨用。整條流程（建 index → 前處理 → 加分項的檔案 → 壓縮 postings → 產生結果 → 打包）由 `build.py` 完成，見「建置流程（build.py）」。

## 組員

| 姓名 | 學號 | 負責部分 |
|---|---|---|
| 陳長壻 | 115598095 | HW2 核心：前處理（`prep.py`、notebook）、`index_reader`、`query_parser`、`matcher`、`ranker`（ltc.ltc cosine）、`search` CLI；Bonus：複合查詢（phrase、proximity、wildcard、巢狀 Boolean、NOT）、拼字修正與 Soundex、index elimination、champion lists |
| 鄧存正 | 115598063 | Bonus：weighting schemes（SMART 變體、BM25，`weighting.py`）、inexact top-k（impact ordering、tiered index、cluster pruning，`tf_index.py`、`clusters.py`）、k-gram wildcard index（`kgram.py`）；網頁 demo 與查詢詞標示（`ui.py`）；正確性驗證與效能量測（`evaluate.py`）；64 MB 封裝（`pack_index.py`）；文件 |

下文中「鄧存正新增」的章節以外，都是陳長壻的實作與說明。

## 給助教：解壓後立即執行

壓縮檔已經包含完整的 index 和所有前處理結果（`output/`），**不需要 WARC 檔，也不需要跑任何前處理**。

上傳限制是 64 MB，而 `postings.txt`（200 MB）用 zip 壓完仍有 63 MB，所以壓縮檔裡放的是 `output/postings.delta.xz`（31 MB）。**第一次執行任何指令時，程式會自動把它還原成 `output/postings.txt`**（約 15 秒，會印出進度，並以檔案大小和 SHA-256 確認和打包前的 `postings.txt` 逐 byte 相同），之後就不會再做。細節見「封裝」一節。

```
cd <ID>_HW2                          # 解壓後的資料夾，名稱和壓縮檔相同
pip install -r requirements.txt
python search.py                     # 出現 cmd> 後輸入 query，例如 Hong Kong business；exit 結束
python ui.py                         # 或：網頁 demo，瀏覽器打開 http://localhost:8000（見「網頁 demo 與查詢詞標示」）
```

- 需要 Python 3.8 以上。
- 不需要網路。NLTK 的 stopwords 已經放在 `nltk_data/`，程式會自動使用。HW1 的程式啟動時仍會嘗試下載 wordnet 和 omw-1.4，離線時會印出幾行 `[nltk_data] Error loading ...`，這是正常的：這兩份資料只有 lemmatization 會用到，而這份 index 沒有開啟 lemmatization，查詢結果不受影響。
- 第一次查詢約需 0.1 秒到數百毫秒（postings 還沒進入 OS 的 disk cache），之後每個 query 約 5 – 15 ms（依電腦而定）。
- 第一次使用 `--impact`、`--tiered`、`--tf-index` 或 `--cluster` 時，會先從 `postings.txt` 建出 `tf_index.bin`（約 20 秒，只做一次）；第一次使用中間或多個 `*` 的 wildcard 時會建 `kgram.bin`（約 3 秒，只做一次）。這兩個檔可以從 postings 重建，為了壓在 64 MB 內沒有放進壓縮檔。想事先建好可以執行 `python prep_extra.py`。
- 預設會檢查拼字：發現疑似拼錯的字時，只在 stderr 印出 `# did you mean: ...`，**排序結果仍然是原本輸入的 query 的結果**，所以不會影響評分。要讓程式自動修正後再搜尋，加上 `--spell correct`；完全關閉則用 `--spell off`。

投影片要求的 ranked list 輸出在 `results/results.txt`，是用 `queries.txt`（投影片上的範例 query 加上幾個加分項的 query）跑出來的。`results/` 的每個檔案都可以用下表的指令重新產生，重跑後和壓縮檔裡的檔案逐 byte 相同（`python build.py results` 會一次全部重跑）：

| 檔案 | 指令 |
|---|---|
| `results.txt`（**必做**：TF-IDF + cosine） | `python search.py -f queries.txt -o results/results.txt` |
| `results_champions.txt` | `python search.py -f queries.txt --champions -o results/results_champions.txt` |
| `results_spell_correct.txt` | `python search.py -f queries.txt --spell correct -o results/results_spell_correct.txt` |
| `results_tiered.txt` | `python search.py -f queries.txt --tiered -o results/results_tiered.txt` |
| `results_impact.txt` | `python search.py -f queries.txt --impact -o results/results_impact.txt` |
| `results_cluster_3,8.txt` | `python search.py -f queries.txt --cluster 3,8 -o "results/results_cluster_3,8.txt"` |
| `results_scheme_<scheme>.txt` | `python search.py -f queries.txt --scheme <scheme> -o ...`（`lnc.ltc`、`anc.atc`、`bm25`） |
| `results_bm25_impact.txt` | `python search.py -f queries.txt --scheme bm25 --impact -o results/results_bm25_impact.txt` |
| `webtrack_ltc.txt` / `webtrack_bm25.txt` | `python search.py -f queries_webtrack.txt --trec-id [--scheme bm25] -o ...` |
| `evaluation.md` | `python evaluate.py all`（約 2 分鐘；時間欄位依電腦而異） |

**用自己的 query 評分**：一行一個 query 寫進文字檔（可以寫成 `qid:query`，`#` 開頭是註解），然後 `python search.py -f 你的檔案.txt -o 結果.txt`。輸出是 `doc# <TAB> 分數`，預設前 10 筆（`-k 0` 為全部）；要對 ClueWeb09 的 qrels，加上 `--trec-id` 或 `--trec-run 名稱`。

**什麼時候才需要 WARC？** 只有要從頭重建 index 時（`python build.py all`，見「建置流程」），以及 `ui.py` 要顯示標題和摘要時。

## 檔案結構

```
<ID>_HW2/
├── build.py                 建置腳本：WARC → index → 前處理 → 加分項檔案 → 壓縮 postings → results/ → dist/<ID>_HW2.zip
├── hw1_indexer.py           HW1 原始程式，原封不動；HW2 import 它的 tokenizer / linguistic / Config / 格式解析
├── prep.py                  前處理的程式本體（命令列：python prep.py）
├── hw2_index_prep.ipynb     前處理的說明版：逐步呼叫 prep.py 的函式並寫出推導，產生的檔案和 prep.py 完全相同
├── index_reader.py          index 的唯讀存取層
├── query_parser.py          query 字串 → AST
├── matcher.py               在 index 上執行 AST 的 filter → 符合條件的 docID
├── ranker.py                TF-IDF + cosine 排序、top-k、inexact top-k 選項
├── search.py                命令列介面（互動 / 單一 query / batch）
├── ui.py                    【鄧】網頁 demo（只用標準函式庫）：語法說明與範例、結果的詞頻表、標題與摘要
├── spell.py                 原始字形詞表（wildcard 用）、拼字修正、Soundex
├── weighting.py             【鄧】weighting schemes：SMART ddd.qqq 與 BM25
├── tf_index.py              【鄧】依 tf 分組的 postings（impact order），--impact / --tiered 共用
├── clusters.py              【鄧】cluster pruning：leader / follower 的建立與查詢
├── kgram.py                 【鄧】3-gram index，一般形式的 wildcard 用
├── prep_extra.py            【鄧】第二段前處理：tf_index、doc_stats、clusters、kgram（不需要 WARC）
├── pack_index.py            【鄧】postings.txt ⇄ postings.delta.xz（64 MB 上傳限制）
├── evaluate.py              【鄧】和暴力版比對正確性、量測各 inexact 方法的時間與 overlap
├── queries.txt              產生 results/ 用的 query
├── queries_webtrack.txt     TREC Web Track 2009–2012 的 200 個 topic（ClueWeb09 的公開 query）
├── requirements.txt
├── nltk_data/               NLTK stopwords，讓程式離線也能執行
├── results/
│   ├── results.txt          queries.txt 的排序結果（投影片第 5 頁的格式）
│   ├── results_champions.txt  同上，開啟 --champions
│   ├── results_spell_correct.txt  同上，開啟 --spell correct
│   ├── results_<選項>.txt   【鄧】同上，各 weighting scheme 與 inexact 方法（檔名即選項）
│   ├── webtrack_ltc.txt / webtrack_bm25.txt  【鄧】200 個 Web Track topic 的 top-10（含 TREC-ID）
│   └── evaluation.md        【鄧】evaluate.py 的輸出：正確性比對與效能量測
└── output/
    ├── dictionary.txt       HW1 產出：term, df
    ├── postings.txt         HW1 產出：term, df: <doc, tf: <pos, ...>; ...>
    ├── offsets.tsv          前處理產出：term \t df \t byte_offset
    ├── doc_norms.tsv        前處理產出：doc_id \t ‖d‖
    ├── doc_lengths.tsv      前處理產出：doc_id \t |d|（BM25 用）
    ├── doc_ids.tsv          前處理產出：doc_id \t trec_id \t url
    ├── champions.tsv        前處理產出：term \t doc doc ...（依 cosine 排序的 champion list）
    ├── champions_bm25.tsv   前處理產出：term \t doc doc ...（依 BM25 排序的 champion list）
    ├── surface_vocab.tsv    前處理產出：surface \t term \t cf \t df（原始字形 → index term）
    ├── meta.json            前處理產出：N、log base、avgdl、champion 設定、build 時的 Config、postings 大小
    ├── postings.delta.xz    【鄧】postings.txt 的壓縮形式（壓縮檔裡只有它，第一次執行時還原）
    ├── postings.pack.json   【鄧】還原後應有的大小與 SHA-256
    ├── doc_stats.tsv        【鄧】每份文件的 |d|、相異 term 數、max tf，以及 15 種 tf×idf 組合的向量長度
    ├── cluster_leaders.tsv  【鄧】177 個 leader 的單位向量
    ├── cluster_followers.tsv【鄧】每份文件最近的 3 個 leader
    ├── clusters.json        【鄧】cluster pruning 的設定
    ├── tf_index.bin/.dir/.json  【鄧】依 tf 分組的 postings；不在壓縮檔內，第一次用到時建立
    ├── kgram.bin/.json      【鄧】3-gram index；不在壓縮檔內，第一次用到時建立
    └── doc_text.tsv         【鄧】ui.py 的標題 / 摘要快取（title \t 前 2 萬字內文，第 n 行 = doc n）；需要 WARC，不在壓縮檔內
```

`07.warc.gz` 不在壓縮檔裡。要從頭重建時，把它放在和 `build.py` 同一層，或用 `--warc` 指定位置。`dist/`（打包的輸出）只會出現在建置用的工作目錄，不在壓縮檔裡。

## 環境建置

需要 Python 3.8 以上。HW1 在 Windows 11 + Python 3.14.7 上測過，HW2 在 Linux + Python 3.12.3 / 3.13 和 Windows 11 + Python 3.14.7 上都測過。

建議用全新的虛擬環境：

```
cd <ID>_HW2
python -m venv .venv
.venv\Scripts\activate              # Windows；Linux / macOS 改成 source .venv/bin/activate
pip install -r requirements.txt     # lxml、nltk、tqdm，與 HW1 相同，HW2 沒有新增套件
pip install notebook                # 只有要打開前處理 notebook 時才需要（或用 VS Code 的 Jupyter）
```

確認環境正常：`python search.py Hong Kong business` 應該印出設定（stderr）和 10 行結果，第一行是 `16103	0.408`。

- **單獨執行 `hw1_indexer.py` 時要指定 NLTK 資料夾**：HW2 的模組會自動把 `nltk_data/` 加進 NLTK 的搜尋路徑，但 HW1 的程式不會，沒有下載過 stopwords 的電腦會在 import 時失敗。`build.py` 呼叫它時會自動設定；手動執行時請用 `NLTK_DATA=nltk_data python hw1_indexer.py ...`（Windows：先 `set NLTK_DATA=nltk_data`）。

## 建置流程（build.py）

GitHub 上的 repo 只存 source code、`nltk_data/`、`results/`；`output/`（約 290 MB，`postings.txt` 一個就 200 MB，超過 GitHub 的單檔上限）、`dist/`、WARC、zip 都列在 `.gitignore`。交件的壓縮檔由 `build.py` 從 WARC 重新產生，助教拿到的壓縮檔裡已經有完整的前處理結果，不需要 WARC。

```
python build.py all --id <學號>       # index → prep → extra → pack → results → package，產生 dist/<學號>_HW2.zip
python build.py clean all --id <學號> # 從零開始（換過 index 來源，或 output/ 裡是別的 index 時用這個）
python build.py results evaluate      # 只重跑 results/ 和 evaluation.md
```

| 步驟 | 做的事 | 什麼時候會跳過 |
|---|---|---|
| `index` | `hw1_indexer.py -i 07.warc.gz -o output`（自動設定 `NLTK_DATA`），約 3.5 分鐘 | `postings.txt`、`dictionary.txt` 比 WARC 和 `hw1_indexer.py` 都新；找不到 WARC 但已有 index 時沿用 |
| `prep` | `prep.py -i output --warc 07.warc.gz`，約 3.5 分鐘，最後跑五項驗證 | 八個輸出檔都比 `postings.txt` 和 `prep.py` 新 |
| `extra` | `prep_extra.py --force`（tf index、`doc_stats.tsv`、cluster、k-gram），約 40 秒 | 輸出都比 `meta.json` 和 `prep_extra.py` 新 |
| `pack` | `pack_index.py pack`，約 2.5 分鐘（含一次還原驗證） | `postings.pack.json` 記的大小和 SHA-256 和目前的 `postings.txt` 相同 |
| `results` | 重新產生 `results/` 的 12 個結果檔（上表），約 20 秒 | 不跳過 |
| `evaluate` | `evaluate.py all` → `results/evaluation.md`，約 2 分鐘；**不在 `all` 裡** | 不跳過 |
| `package` | 打包成 `dist/<ID>_HW2.zip`，最上層資料夾也叫 `<ID>_HW2`；不放 `postings.txt`（放壓縮版）和第一次使用時會自動重建的 `tf_index.*`、`kgram.*`、`doc_text.tsv` | 不跳過；缺檔、或 `postings.delta.xz` 和 `postings.txt` 對不上時中止；超過 64 MB 時警告 |

- 「是否跳過」和 Makefile 一樣比較檔案的修改時間；任何一步失敗就停止。
- `extra` 一律加 `--force`：`prep_extra.py` 單獨執行時只要 `doc_stats.tsv` 存在就不重建，index 換過後會留下舊的檔案。
- `package` 會再確認一次壓縮的 postings 屬於目前的 index。最容易出錯的情況是重建 index 後忘了重新 pack：本機有 `postings.txt` 所以一切正常，但壓縮檔裡會是舊的 `postings.delta.xz` 配上新的 `offsets.tsv`，助教那邊還原出舊的 postings、SHA-256 也會通過，查詢結果卻全錯。
- 開發時在只有 git 內容的全新資料夾執行 `python build.py all`（約 11 分鐘），再把產生的壓縮檔解到另一個資料夾、用全新的 venv 只裝 `requirements.txt` 執行：自動還原 `postings.txt`、自動建立 tf index 後，`results.txt`、`results_impact.txt`、`results_scheme_bm25.txt` 和打包前逐 byte 相同。

**和 HW1 交出去的 index 的關係**：這份 index 是用 HW1 的 `hw1_indexer.py`（沒有修改）在 Linux 上從 `07.warc.gz` 重新建出的，和 HW1 當時在 Windows 上交出去的檔案不完全相同：term 數 346,789（HW1 交件版 345,701）、postings 數 8,976,349（8,956,202）。抽查前 300 份文件，有 11 份的 term 不同，差異集中在混合文字的亂碼 token（如 `ÎI4ÙÌ`）和少數 HTML 殘留；在 Windows 上重新解析則和 HW1 交件版一致，所以原因是兩個平台上 lxml 的行為不同。一般英文 query 的排序幾乎不受影響：`queries.txt` 的 10 個 query 在兩份 index 上的 top-10，`results.txt` 只有 1 個位置換了文件。投影片只要求「HW1 從 ClueWeb09 產生的 inverted index」，所以用同一支程式重建是允許的；好處是 index、`surface_vocab.tsv` 和結果檔全部出自同一次建置。README 裡的統計數字、範例輸出和效果量測，除非另外註明，都是在這份重建的 index 上得到的。

## 使用步驟

**1. 準備 index。** 壓縮檔裡已經有了（`postings.txt` 是壓縮版，第一次執行時自動還原）。要從頭重建，用 `python build.py index`（見上一節），約 3.5 分鐘。

**2. 前處理（壓縮檔裡已經做好；index 重建過才需要重跑）。** 兩種方式擇一，產生的檔案逐 byte 相同：

```
python prep.py                                # 約 3.5 分鐘，含驗證；加 --no-surface 約 1 分鐘
```

或打開 `hw2_index_prep.ipynb` 後 Run All。正確的結果應該是：

- `response records: 31,480`，最後一筆的 TREC-ID 是 `clueweb09-en0000-07-31479`
- `346,789 terms, 8,976,349 postings`、`N = 31,480 (warc)`、`avgdl = 568.37`
- 每個 term 的 champion list 合計 1,827,355 個 docID（兩種 scheme 相同）
- surface vocabulary 402,109 行，涵蓋 100.00% 的 index term（見「原始字形詞表」一節）
- 四項驗證都印出 `OK` / `verify: ...`

`prep.py` 的選項：

| 選項 | 說明 |
|---|---|
| `-i DIR` | index 資料夾，預設是 `prep.py` 旁邊的 `output/` |
| `--warc PATH` / `--no-warc` | WARC 的位置；或不讀 WARC |
| `--champion-schemes ...` | 要產生哪些 champion list：`cosine`、`bm25`，預設兩個都產生；沒選到的舊檔會被刪除 |
| `--champion-r R` | 每個 term 保留幾份文件，預設 50 |
| `--bm25-k1 K1` / `--bm25-b B` | 排序 `champions_bm25.tsv` 用的 BM25 參數，預設 1.2 / 0.75 |
| `--no-surface` | 不重建 `surface_vocab.tsv`（沿用既有的檔案），省下約 4 分鐘 |
| `--no-verify` | 跳過驗證 |

**只要 index 重建過，就必須重跑這一步**（`build.py` 會自動判斷）：`index_reader` 啟動時會比對 postings 的檔案大小和 `meta.json` 的紀錄，對不上就拒絕執行，避免拿舊的 offset 去 seek。

**3. 查詢。**

```
python search.py                               # 互動模式，提示字元 cmd>；exit / quit / Ctrl-D / Ctrl-Z 結束
python search.py Hong Kong business            # 單一 query
python search.py -f queries.txt -o run.txt     # batch：一行一個 query，可寫 "qid:query"，# 開頭為註解
```

含有引號、`*`、括號的 query，在 shell 裡要注意跳脫；最簡單的方式是用互動模式直接輸入。

輸出格式和投影片第 5 頁一樣：

```
16103	0.408
16101	0.404
16097	0.276
```

| 選項 | 說明 |
|---|---|
| `-k N` | 每個 query 回傳幾筆，預設 10，`0` 代表全部 |
| `--trec-id` | 第三欄加上 ClueWeb09 TREC-ID |
| `--trec-run NAME` | 輸出 `qid Q0 docno rank score NAME`，可直接交給 `trec_eval` |
| `-v` | 額外顯示 query 的解析結果、candidate 數、使用和被丟掉的 term、merge 統計、耗時 |
| `--digits N` | 分數的小數位數，預設 3 |
| `--idf-threshold X` | index elimination：idf 低於 X 的 query term 不參與計算 |
| `--min-match M` | index elimination（只作用在 free text）：文件至少要含 M 個 query term |
| `--champions` | 只計算 champion list 聯集內的文件（讀 `champions.tsv`；`--scheme bm25` 時讀 `champions_bm25.tsv`） |
| `--scheme DDD.QQQ` | 【鄧】weighting scheme，SMART 記法（文件.查詢）或 `bm25`；預設 `ltc.ltc` 即投影片公式。見「Weighting schemes」 |
| `--k1 X` / `--b X` | 【鄧】BM25 參數，預設 1.2 / 0.75 |
| `--impact` | 【鄧】impact ordering：tf 由大到小的 postings + max-score 提早結束，top-k 和精確結果相同 |
| `--tiered` | 【鄧】tiered index：先只看高 tf 的 tier，不足 k 份才往下一層 |
| `--cluster B1,B2` | 【鄧】cluster pruning：每份文件跟隨最近的 B1 個 leader（≤ 3），查詢時只看離 query 最近的 B2 個 leader 的 follower |
| `--tf-index` | 【鄧】精確排序，但 (doc, tf) 改從二進位的 `tf_index.bin` 讀，不解析 `postings.txt` |
| `--wildcard-limit N` | wildcard 用來計分的展開詞上限，預設 30 |
| `--spell MODE` | 拼字修正：`suggest`（預設，只提示，不改結果）、`correct`（修正後再搜尋）、`off` |
| `--phonetic` | 拼字修正時也考慮發音相近（Soundex）的字 |
| `-i DIR` | index 資料夾，預設是 `search.py` 旁邊的 `output/` |

設定、警告和錯誤一律寫到 stderr，所以 `-o` 產生的檔案裡只有結果。

## 查詢語法

| 語法 | 範例 | 意義 |
|---|---|---|
| 單字 / free text | `United States government` | 經過和文件相同的 tokenizer + linguistic，得到 `unit state govern` |
| phrase | `"United States of America"` | 依 positions 比對，stopword 留下的位置間隔也要對齊 |
| proximity | `hong /3 kong` | 兩個詞相距 ≤ 3 個位置，不限前後順序；可以串接：`a /3 b /2 c` |
| wildcard | `comput*`、`wom?n`、`*ing` | 同時比對 index 裡的 term（已 stem）和文件中的原始字形，取聯集；只做 case folding |
| Boolean | `AND` `OR` `NOT` `( )` | 必須**大寫**；小寫的 `and` 視為一般 stopword |

查詢分成兩種模式：

- **Free text**：沒有任何 operator 或括號。一般單字和 wildcard 只影響排序；加引號的 phrase 和 proximity 是必要條件。
- **Boolean**：出現任何 operator。相鄰的詞視為隱含的 AND，例如 `software engineer OR project manager` 等於 `(software AND engineer) OR (project AND manager)`。優先順序是 proximity > NOT > AND > OR。

## 模組如何運作

### 資料流

```
                        hw1_indexer.py（tokenize / linguistic / Config / parse_posting_line）
                          ▲               ▲                    ▲
                          │               │                    │
 query 字串 ──► query_parser ──Query──► matcher ──docID list──► ranker ──hits──► search（輸出）
                          │               │                    │
                          └─ cfg ◄──── index_reader ───────────┘
                                          │
                              offsets / norms / meta / doc_ids（記憶體）
                              postings.txt（硬碟，seek + readline）
```

`search.py` 先建立 `IndexReader`，再用它建立 `Ranker`（`Ranker` 內部會再建立 `Matcher`）和 `SpellCorrector`。每個 query 依序經過 `SpellCorrector.correct_query`（拼字檢查）→ `parse_query` → `Ranker.rank`；`rank` 內部會呼叫 `Matcher.match` 取得 candidate，然後計分。`Matcher` 和 `SpellCorrector` 共用同一份原始字形詞表，第一次需要時才載入（約 0.3 秒）。

`index_reader.py`、`query_parser.py`、`prep.py`、`spell.py` 在 import `hw1_indexer` 之前，會先把專案的 `nltk_data/` 加進 NLTK 的搜尋路徑，所以不需要事先下載 stopwords。

### prep.py 與 hw2_index_prep.ipynb：補齊 HW1 的缺口

**兩種形式的關係**：所有邏輯只寫在 `prep.py` 的函式裡。`python prep.py` 依序呼叫這些函式；notebook 一格一格呼叫同一批函式，並把推導寫在 markdown 裡。程式只有一份，所以兩邊不會不同步。開發時兩種方式都從頭跑過一次，七個輸出檔逐 byte 相同，也和改版前 notebook 產生的 `offsets.tsv`、`doc_norms.tsv`、`doc_ids.tsv` 相同。

HW1 少了這些 HW2 需要的東西：trie 的 byte offset 只存在 build 時的記憶體裡、沒有把 N 存下來、沒有文件向量的長度（cosine 的分母）、沒有文件長度（BM25 用）。步驟如下：

1. **掃 WARC**：N 取 response record 的數量，這和 HW1 的編號規則一致：每筆 response 都有一個 docID，即使抽不出文字也一樣。同時記下 TREC-ID 和 URL。
2. **第一次掃 postings**：以 binary mode 讀，用累計的 byte 數記下每行的 offset。每行只用 regex 抓 `(docID, tf)`，不解析 positions，並檢查抓到的數量等於 df。同一次掃描累加 norm 需要的三個和，以及每份文件的 $|d|=\sum_t tf$。
3. **算 norm 和 avgdl**：norm 用展開式，因此掃描時不需要事先知道 N。令 $a=1+\log tf$、$L=\log N$、$\ell=\log df$：

   $$\|d\|^2=\sum_t a_t^2(L-\ell_t)^2 = L^2\textstyle\sum a^2-2L\sum a^2\ell+\sum a^2\ell^2$$

   每份文件只累加三個和，等 N 確定後一次算出長度。avgdl $=\sum_d |d| / N$。
4. **第二次掃 postings：champion lists**（見下一節）。
5. **第二次掃 WARC：原始字形詞表**（見「原始字形詞表」一節）。這一步要重新解析每一頁的 HTML，約需 4 分鐘，是前處理最慢的部分。
6. **寫 `meta.json`**：最後才寫，所以它描述的一定是旁邊這一批檔案。

驗證以 HW1 的檔案和直接套公式的暴力計算當作正確答案：offsets 和 `dictionary.txt` 逐行一致、隨機 seek 讀到正確的 record、展開式算出的 norm 和直接套公式的結果相同（相對誤差 < 1e-9）、文件長度和直接加總 tf 的結果完全相同、champion list 和完整排序的結果相同、原始字形詞表的格式和對應關係正確（見下文）。

### Champion lists

每個 term 預先挑出「這個 term 貢獻最大」的前 r 份文件（預設 r = 50）。查詢時只對所有 query term 的 champion list 聯集計分，free text 的候選文件從上萬份降到一兩百份。

同一個 term 的所有 posting 共用同一個 idf，所以挑選時可以忽略 idf，只比較下表的分數：

| 檔案 | 每份文件的排序分數 | 誰使用 |
|---|---|---|
| `champions.tsv` | $(1+\log tf)\,/\,\lVert d\rVert$ | `ranker.py` 的 `--champions`（cosine） |
| `champions_bm25.tsv` | $tf\,(k_1+1)\,/\,\big(tf + k_1(1-b+b\,\lvert d\rvert/\text{avgdl})\big)$ | `--scheme bm25` 的 `--champions` |

- 分數相同時 docID 小的優先，所以每次產生的檔案都一樣。
- df ≤ r 的 term 直接保留全部文件。每個 term 都有一行，行的順序和 `dictionary.txt` 相同；每行的 docID 由小到大排列。
- `ranker.py` 載入時只把每行存成字串，查詢用到某個 term 時才解析它的 docID，避免每次啟動都要解析 183 萬個 ID。
- **Fallback**：Boolean、phrase、proximity 的 filter 和 champion list 取交集後，常常剩不到 k 份文件（例如 `open source AND Linux` 只剩 2 份）。這時 ranker 會依課本的建議，改用完整的 candidate 重新排序；`-v` 會印出 `champion lists gave fewer than k documents`。

效果（開發時量測，free text query，top-10）：10 個 query 中有 9 個的 top-10 和精確結果完全相同，另一個是 9/10；每個 query 的計算時間約減少 30%。時間沒有減少更多，是因為 postings 仍然要從硬碟讀出來，才知道 champion 文件的 tf；champion list 省下的是計分和 top-k 選取的部分。

### 原始字形詞表（surface_vocab.tsv）

index 存的是 stem 之後的 term，原本的字形已經不見了：`running`、`runs` 都變成 `run`。所以只靠 index，`*ing` 找不到任何東西，拼錯的字也沒有正確的拼法可以比對。前處理會重新解析 WARC 的每一頁，用 HW1 的 `tokenize` 和 `linguistic`，記下每個被 index 收進去的 token 原本的字形和它變成的 term：

| 欄位 | 意義 |
|---|---|
| surface | case folding 後的原始 token，例如 `running` |
| term | 這個 token 在 index 裡對應的 term，例如 `run` |
| cf | 這個字形在整個 collection 出現幾次 |
| df | 幾份文件含有這個字形 |

- 依 surface 的 code point 排序，一個 surface 一行。同樣開頭的字形是連續的一段，可以用 bisect 找。
- stopword 不會進 index，所以也不在表裡。
- **只保留 index 裡存在的 term。** 解析 WARC 的程式和建 index 時相同，正常情況下每個 term 都在 index 裡。但 index 若是在另一個平台上建的，兩次解析可能不一致：在 Linux 上解析、對照 HW1 在 Windows 上交出的 index 時，會多出 1,968 組 index 裡沒有的 term，這些會被丟掉。
- 目前這份詞表和 index 都是 `build.py` 在同一台電腦上產生的：402,109 行，丟掉 0 組，涵蓋 100.00% 的 index term 和 token 出現次數。（對照用：在 Linux 上對 HW1 交件版的 index 產生時是 400,141 行、99.75%；在建 HW1 index 的 Windows 電腦上則是 400,989 行、100.00%，這也證實差異來自兩個平台上 lxml 的行為。）
- 驗證（`verify_surface_vocab`）逐行檢查：surface 嚴格遞增、term 都在 index 裡、HW1 的 `linguistic` 處理這個 token 後剛好得到這個 term、$0 < df \le cf$。

### spell.py：wildcard 比對原始字形、拼字修正

**Wildcard。** `Matcher.expand_wildcard` 會把 pattern 同時套在 index term 和原始字形上，取兩邊得到的 term 的聯集。`comput*` 的結果和以前相同，`*ing` 則會找到 `running` → `run` 等 8,290 個 term。查詢方式：

- pattern 開頭的固定字串比結尾長（`comput*`、`c*t*r`）：在排序好的字形表上用 bisect 取出連續的一段，再用 regex 篩。
- 結尾的固定字串比較長（`*ing`、`*ology`）：在「每個字形反過來拼」的排序表上做同樣的事，也就是課本的 reverse B-tree。這張表第一次用到時才建立。

**拼字修正**（課本 3.3–3.4）。`SpellCorrector.correct_query` 逐一檢查 query 裡的英文單字；`AND`/`OR`/`NOT`、wildcard 的片段、和數字黏在一起的字（`mp3`）、`/3` 都不動，引號和括號原樣保留。

1. **哪些字要修**：stopword 不修。所有 term 都在 index 裡、而且這個拼法在 collection 出現超過 50 次的字視為正確。其餘的字是候選：index 裡沒有的字一定修；index 裡有、但這個拼法很少見的字（例如網頁上常見的 `goverment` 只出現 24 次），只有在找到頻率高 50 倍以上的鄰近字時才修。
2. **字典**：原始字形詞表中，只由英文字母（可含 `'`）組成、出現至少 2 次的字，共約 17 萬個。另外加入 NLTK 的 stopword，並給它們最高的頻率，所以 `teh` 這類拼錯的 stopword 也能被修正（`teh` 本身在 collection 出現 72 次，所以預設會被當成正確的字）。
3. **候選詞（bigram index）**：每個字以 `$` 標記頭尾後切成 bigram，依「bigram → 字長 → 字」建立索引。查詢字的長度在 4 以下時最多允許 1 個 edit，5 以上最多 2 個，所以只看長度差在範圍內的字。一個 edit 最多破壞 3 個 bigram（相鄰字元交換的情況），所以 $k$ 個 edit 之後，兩個字至少還共用 $\max(\text{不同 bigram 數}) - 3k$ 個 bigram。這個下界不會漏掉任何候選詞，唯一的例外是兩個字完全沒有共同 bigram 的極端情況。
4. **距離**：Damerau-Levenshtein 的 optimal string alignment 版本，也就是插入、刪除、替換、相鄰字元交換各算 1。超過上限時提早結束計算。
5. **選擇**：分數 $= d\cdot\ln 50 - \ln(\text{頻率}) - \ln(1 + \text{共同文件數})$，取最小值。每多一個 edit，候選詞的頻率要高 50 倍才能勝出。「共同文件數」是課本 3.3.5 的 context-sensitive correction：候選詞和 query 中相鄰單字同時出現的文件數，只計算分數最好的前 20 個候選詞。所以 `hong kng` 會修成 `hong kong`，而不是頻率更高的 `hong king`。
6. **Phonetic（`--phonetic`）**：Soundex 依課本 3.4 的步驟計算。比對時忽略第一個字母，只看後面的三位數字，所以 `kristmas`（K623）和 `christmas`（C623）也算發音相近。發音相近的字可以比一般候選詞多 1 個 edit，而且在分數上有 $\ln 2$ 的加分。

範例（`--spell correct`）：`busness in hong kng` → `business in hong kong`、`infromation retreival` → `information retrieval`、`unted stats govermnet` → `united stats government`、`"hong kng" AND microsfot NOT linnux` → `"hong kong" AND microsoft NOT linux`。

### index_reader.py：唯讀存取層

啟動時讀入 `meta.json`，從中還原 HW1 build 時的 `Config`，放在 `idx.cfg`。所有 query 都用這份設定處理，確保和文件的處理方式一致。接著讀入 `offsets.tsv`、`doc_norms.tsv`、`doc_ids.tsv`，約 0.4 秒；`postings.txt` 的 file handle 保持開啟。TSV 都用 `newline="\n"` 開檔，因為 WARC 中有幾個 URL 含有原始的 `\r`。

| 介面 | 用途 |
|---|---|
| `get_doc_tfs(term)` | `[(doc_id, tf), ...]`，不解析 positions，給排序用 |
| `get_postings(term, skips=False)` | HW1 的 `PostingEntry` list（含 positions），給 phrase / proximity 用 |
| `df` / `idf` / `tf_weight` | 投影片的公式，log 底數讀自 `meta.json` |
| `terms_with_prefix(prefix)` | 字典依 code point 排序，用 bisect 取出連續的一段 |
| `doc_label(doc_id)` | 回傳 TREC-ID |

### query_parser.py：字串 → AST

先由 lexer 切成 `WORD / PHRASE / PROX / AND / OR / NOT / ( )`，再由 recursive-descent parser 建出 AST。AST 的節點有 `Term`、`Phrase`、`Proximity`、`Wildcard`、`And`、`Or`、`Not`。這個模組不讀 index，wildcard 會保留成 pattern。

每個單字都送進 HW1 的 `tokenize` + `linguistic`：

- 變成 0 個 term（stopword）→ 這個 operand 消失，並附上 warning。
- 變成 1 個 term → `Term`。
- 變成多個 term（如 `3.5mm`、`state-of-the-art`）→ `Phrase`，保留位置間隔。

`NOT NOT x` 會化簡成 `x`。巢狀的同種 operator 會被攤平。

輸出是 `Query(raw, mode, filter, scoring, warnings)`：`filter` 決定哪些文件符合條件（`None` 代表不篩選），`scoring` 提供排序用的 term。`positive_leaves()` 會略過 NOT 底下的節點。語法錯誤時丟出 `QuerySyntaxError`。

### matcher.py：執行 filter

每種節點都回傳排序好的 docID list：

- **Term**：直接讀 postings。
- **Phrase**：從 df 最小的 term 開始，記下每份文件可能的起點，再用其他 term 逐一縮小範圍。
- **Proximity**：先用 skip pointers 找出兩個詞都出現的文件，再在文件內以 two-pointer 檢查是否有一組位置相距 ≤ k。
- **Wildcard**：在 index term 上，prefix pattern 用 bisect，其他 pattern 先用開頭的固定字串縮小範圍，再用 regex 比對；再和原始字形詞表的比對結果取聯集（見上一節）。展開結果會跨 query 快取。
- **AND**：由短到長兩兩合併，每 ⌊√n⌋ 個位置放一個 skip pointer。NOT 子節點直接從結果中扣掉，不先取補集。
- **OR**：`heapq.merge` 做 k-way merge 並去重。
- **NOT**：出現在 AND 以外的地方時，才對 1..N 取補集。

同一個 query 內，同一個 term 只從硬碟讀一次。`matcher.stats` 記錄 comparisons、skips、postings_read。

### ranker.py：排序

query 端和 document 端用同樣的權重 $(1+\log tf)\cdot\log(N/df)$，分數為 $\sum_t w_{tq}w_{td}/(|q|\,|d|)$。

- **query term**：從 `positive_leaves` 收集，重複的詞會累計 query tf。wildcard 只取 df 最高的前 `wildcard_limit` 個展開詞。
- **candidate**：有 filter 時就是 matcher 的結果；純 free text 時，含任一 query term 的文件都是 candidate。符合 filter 但分數為 0 的文件（如 `NOT microsoft`）仍會出現在結果裡，排在最後。
- **計算**：term-at-a-time 累加，用 `heapq.nlargest` 取 top-k；同分時 docID 小的排前面。
- **inexact top-k**：`idf_threshold`、`min_match`、`champions` 只縮小計算範圍。$|q|$ 一律用全部的 term 計算，所以開關這些選項不會改變分數的尺度。

### search.py：命令列介面

解析參數，載入 index（失敗時回傳 exit code 1），然後依參數進入互動、單一 query 或 batch 模式。每個 query 都經過 `run_one()`，它負責拼字檢查、parse、rank，並把結果格式化成 plain 或 TREC run 格式。拼字的提示和修正內容寫到 stderr（batch 模式前面加上 `[qid]`）。

## API 參考

陳長壻的模組沒有 docstring，每個函式的用途寫在它上一行的註解；較完整的說明集中在這裡。型別標註直接看程式碼。

### index_reader.py

| 名稱 | 說明 |
|---|---|
| `IndexReader(index_dir)` | 讀入 `meta.json`、`offsets.tsv`、`doc_norms.tsv`、`doc_ids.tsv`，並開啟 `postings.txt`。dictionary 和統計量放在記憶體，postings 留在硬碟。可以用 `with` 使用，或在結束時呼叫 `close()`。`postings.txt` 的大小和 `meta.json` 記錄的不同時丟出 `RuntimeError` |
| `.N`、`.log_base`、`.cfg`、`.meta` | 文件數、log 底數、建 index 時的 HW1 `Config`、整份 `meta.json` |
| `.doc_norms[doc_id]` | 文件向量長度，沒有 term 的文件為 0 |
| `.has_term(t)` / `.df(t)` / `.idf(t)` / `.tf_weight(tf)` | `t` 是處理過的 term；不在 index 的 term df 和 idf 都是 0 |
| `.terms_with_prefix(p)` / `.all_terms()` / `.vocab_size` | 依 code point 排序的字典 |
| `.get_doc_tfs(t)` | `[(doc_id, tf), ...]`，doc_id 遞增，不解析 positions，排序時用 |
| `.get_postings(t, skips=False)` | HW1 的 `PostingEntry` list（含 positions），phrase / proximity 用；`skips=True` 會附上 skip pointer |
| `.doc_label(doc_id)` / `.urls[doc_id]` | TREC-ID（沒有 `doc_ids.tsv` 時回傳 doc_id 字串）、URL |

### query_parser.py

| 名稱 | 說明 |
|---|---|
| `parse_query(text, cfg=None)` | 用 index 的 linguistic 設定（傳 `IndexReader.cfg`；`None` 代表 HW1 預設值）把 query 字串轉成 `Query`。語法錯誤時丟出 `QuerySyntaxError` |
| `Query` | `raw`（原始字串）、`mode`（`"free"` 或 `"boolean"`）、`filter`（結果必須滿足的 AST，`None` 代表不篩選）、`scoring`（提供排序 term 的節點）、`warnings` |
| `Term(term)` | 一個處理過的 term，例如 `Term('busi')` |
| `Phrase(terms)` | `((term, offset), ...)`，第一個 offset 是 0；被移除的 stopword 留下間隔 |
| `Proximity(left, right, k)` | 兩個 term 相距 ≤ k 個位置，不限前後 |
| `Wildcard(pattern)` | 只做過 case folding 的 pattern，由 matcher 展開 |
| `And(children)` / `Or(children)` / `Not(child)` | Boolean 節點；同種節點巢狀時會被攤平 |
| `positive_leaves(node)` | 逐一回傳不在 NOT 底下的葉節點，它們的 term 就是排序用的 term |
| `QuerySyntaxError` | 括號不對稱、operator 缺少運算元、proximity 的運算元不是單一 term 等情況 |

內部：`_lex(text)` 把字串切成 `LPAREN RPAREN AND OR NOT PROX PHRASE WORD` token；`_Parser` 是 recursive-descent parser，文法依優先順序由低到高為 `parse_or` → `parse_and`（相鄰即 AND）→ `parse_not` → `parse_prox`（`a /k b /m c`）→ `parse_atom`（單字、引號 phrase、括號）。

### matcher.py

| 名稱 | 說明 |
|---|---|
| `Matcher(idx)` | 在 `IndexReader` 上執行 query 的 filter |
| `.match(query)` | 滿足 `query.filter` 的 doc_id（遞增）；free text 沒有 filter 時回傳 `None`，由 ranker 自己決定 candidate |
| `.evaluate(node)` | 單一 AST 節點的 doc_id list（遞增）。Term：postings；Phrase：從最稀有的 term 開始，其他 term 必須出現在對應的 offset；Proximity：兩個 term 在 k 個位置以內；Wildcard：所有展開 term 的聯集；And：由短到長以 skip pointer 交集，NOT 子節點直接扣除；Or：聯集；Not：對 1..N 取補集 |
| `.expand_wildcard(pattern)` | 符合 pattern 的 index term，加上符合 pattern 的原始字形所對應的 term，排序後回傳；跨 query 快取 |
| `.surface_vocab()` | 第一次呼叫時載入 `surface_vocab.tsv`；檔案不存在或 `meta.json` 沒有描述它時回傳 `None`，這時 wildcard 只比對 index term |
| `.stats` | 上一個 query 的 `comparisons`、`skips`、`postings_read` |

內部：`_intersect` 是課本 Fig. 2.10 的線性合併，兩邊每隔 ⌊√長度⌋ 個位置放一個 skip pointer；`_union` 是不含重複的 k-way merge；`_close_pair` 用 two-pointer 判斷兩組位置是否有一對相距 ≤ k。

### ranker.py

| 名稱 | 說明 |
|---|---|
| `Ranker(idx, matcher=None, wildcard_limit=30)` | TF-IDF + cosine 排序；`wildcard_limit` 是一個 wildcard 最多用幾個展開詞來計分（取 df 最高的） |
| `.rank(query, top_k=10, idf_threshold=None, min_match=1, champions=False)` | 回傳 `RankResult`。`top_k <= 0` 代表回傳全部 candidate。`idf_threshold`：idf 低於門檻的 query term 不計分（全部低於門檻時保留 idf 最高的一個）。`min_match`：只作用在 free text，文件至少要含幾個 query term。`champions`：只對 champion list 聯集內的文件計分，結果不足 `top_k` 筆時改用完整的 candidate 重新排序 |
| `RankResult` | `hits`（`(doc_id, score)`，分數高的在前）、`num_candidates`、`terms_used`（term → query tf）、`terms_dropped`、`champions_fallback` |
| `.query_terms(query)` | 處理過的 term → 在 query 中的次數，只取不在 NOT 底下的葉節點 |
| `.champion_docs(term)` | 一個 term 的 champion doc_id；`champions.tsv` 每個 process 只讀一次，每個 term 第一次用到時才解析 |
| `.doc_weight(tf, idf)` / `.query_weight(qtf, idf)` | 投影片的權重公式 |

公式：$w_{t,d}=(1+\log tf_{t,d})\log(N/df_t)$，query 端相同；$\text{score}(q,d)=\sum_t w_{t,q}w_{t,d}/(|q|\,|d|)$。

### spell.py

| 名稱 | 說明 |
|---|---|
| `SurfaceVocab(index_dir)` / `SurfaceVocab.load(index_dir)` | 讀入 `surface_vocab.tsv`；`load` 在檔案不存在時回傳 `None`。欄位 `surfaces`、`terms`、`cf`（同一個位置對應同一行）、`position`（字形 → 位置） |
| `.match_pattern(pattern)` | 符合 wildcard pattern 的字形位置（遞增）；開頭或結尾固定字串較長的那邊用來縮小範圍 |
| `.terms_for_pattern(pattern)` | 符合 pattern 的字形所對應的 term，去除重複並排序 |
| `SpellCorrector(idx, vocab_loader, phonetic=False, min_cf=2, rare_cf=50, rare_ratio=50)` | `vocab_loader` 是回傳 `SurfaceVocab`（或 `None`）的函式，search.py 傳入 `Matcher.surface_vocab`。字典和 bigram index 在第一個需要修正的字出現時才建立（約 0.7 秒）。`min_cf`：字典收錄的最低出現次數；`rare_cf`、`rare_ratio`：見「拼字修正」第 1 點 |
| `.correct_query(text)` | 回傳 `(修正後的字串, [(原字, 新字), ...])`，其他字元原樣保留 |
| `.correct_word(word, context=())` | 小寫單字的最佳修正，不需要修正時回傳 `None`；`context` 是相鄰單字的 index term |
| `.edit_candidates(word)` / `.phonetic_candidates(word)` | `[(距離, 字典位置), ...]` |
| `.unavailable` | 找不到 `surface_vocab.tsv` 時為 `True` |
| `osa_distance(a, b, limit)` | Damerau-Levenshtein（optimal string alignment）距離；超過 `limit` 時提早回傳 `limit + 1` |
| `soundex(word)` | 課本 3.4 的 Soundex 碼，例如 `soundex("Hermann") == "H655"` |
| `bigrams(word)` | 前後加上 `$` 的 bigram list |
| `wildcard_regex(pattern)` | `*` → 任意長度、`?` → 一個字元，其餘字元照字面比對 |

### search.py

| 名稱 | 說明 |
|---|---|
| `main(argv=None)` | 程式入口，回傳 exit code（index 打不開時是 1） |
| `run_one(ranker, text, args, qid, speller=None)` | 處理一個 query：拼字檢查、parse、rank、格式化；回傳 `(結果行, 診斷行)` |
| `iter_batch(path)` | 逐一回傳 batch 檔中的 `(qid, query)`；空行和 `#` 開頭的行會被略過，沒有 `qid:` 前綴時以行號當 qid |
| `build_arg_parser()` / `print_settings(...)` | 命令列選項；啟動時把所有設定印到 stderr |

### prep.py

notebook 依序呼叫的就是這些函式；`run()` 是 `python prep.py` 的完整流程。

| 名稱 | 說明 |
|---|---|
| `run(index_dir, warc_path, log_base, schemes, r, k1, b, verify, seed, surface)` | 依序執行全部步驟和驗證，回傳 `meta.json` 的內容 |
| `scan_warc_doc_ids(warc_path)` | `[(doc_id, trec_id, url), ...]`，編號方式和 `hw1.parse_warc` 完全相同（每筆 response record 一個 id） |
| `iter_postings_records(postings_path)` | 逐一回傳 `postings.txt` 每行的 `(byte offset, term, df, [(docID, tf), ...])`（bytes），並檢查格式和 df |
| `scan_postings(postings_path, log_base)` | 第一次掃描，回傳 `PostingsScan`：`term_entries`（term, df, offset）、norm 的三個和 `s0`/`s1`/`s2`、`doc_lengths`、每個 term 的 `term_cf`、`num_postings`、`postings_bytes` |
| `decide_n(scan, warc_docs)` | `(N, N_source, max_doc_id)`；有 WARC 時 N 取 response record 數，否則取最大 doc_id |
| `compute_doc_norms(scan, N, log_base)` | 用展開式 $\sqrt{L^2S_0-2LS_1+S_2}$ 算出每份文件的 $\lVert d\rVert$ |
| `average_doc_length(doc_lengths, N)` | avgdl |
| `write_champions(...)` / `top_r_docs(...)` / `champion_scorers(...)` | 第二次掃描，逐 scheme 寫出 champion list；每個 term 依該 scheme 的分數取前 r 份，同分時 doc_id 小的優先，輸出依 doc_id 遞增 |
| `scan_surface_vocab(warc_path, cfg)` | `(原始字形, term) → [cf, df]`，用 HW1 的 tokenize + linguistic |
| `write_surface_vocab(index_dir, vocab, index_terms)` | 只寫出 term 在 index 裡的字形；回傳（寫出行數, 丟棄組數） |
| `surface_step(index_dir, warc_path, scan, enabled)` | 有 WARC 且 `enabled` 時重建詞表；否則若既有的檔案是針對同一份 `postings.txt` 建的就沿用。回傳 `meta.json` 的 `surface_vocab` 欄位（沒有可用的詞表時為 `None`） |
| `write_tables(...)` / `build_meta(...)` / `write_meta(...)` | 寫出表格檔與 `meta.json`（一律 LF 換行） |
| `verify_dictionary` / `verify_seeks` / `verify_doc_stats` / `verify_champions` / `verify_surface_vocab` | 各項驗證，失敗時丟出 `AssertionError` |

## 鄧存正新增的部分

以下各節是在陳長壻的程式之上加的功能。**所有新選項都關閉時，執行的路徑和計算式與原本相同，`queries.txt` 的三份結果（`results.txt`、`results_champions.txt`、`results_spell_correct.txt`）逐 byte 不變。**

### Weighting schemes（Ch.6，`weighting.py`）

`--scheme ddd.qqq` 用課本 Fig. 6.15 的 SMART 記法，前三個字母是文件端，後三個是查詢端：

| 位置 | 字母 | 公式 |
|---|---|---|
| tf | `n` / `l` / `a` / `b` / `L` | $tf$ ／ $1+\log tf$ ／ $0.5+0.5\,tf/\max tf$ ／ 1 ／ $(1+\log tf)/(1+\log \text{avg}\,tf)$ |
| idf | `n` / `t` / `p` | 1 ／ $\log(N/df)$ ／ $\max(0,\log\frac{N-df}{df})$ |
| normalization | `n` / `c` | 無 ／ cosine（除以向量長度） |

- 共 $(5\times3\times2)^2=900$ 種組合。預設 `ltc.ltc` 就是投影片第 4 頁的公式；`lnc.ltc` 是課本的範例。只寫三個字母（`--scheme lnc`）代表兩端相同。
- `a` 需要每份文件的 max tf，`L` 需要平均 tf（$|d|$ ÷ 相異 term 數），`c` 需要該組 tf×idf 的向量長度。這些都由 `prep_extra.py` 一次算好放在 `doc_stats.tsv`（15 欄向量長度），查詢時只讀用到的欄位。`lt` 那一欄和陳長壻的 `doc_norms.tsv` 相對誤差 < 1e-9，預設 scheme 仍讀 `doc_norms.tsv`。
- 查詢端的 `a`、`L` 用 query 自己的 max tf 和平均 tf。
- `--scheme bm25`：$\sum_t qtf_t\cdot\ln\!\big(1+\frac{N-df+0.5}{df+0.5}\big)\cdot\frac{tf\,(k_1+1)}{tf+k_1(1-b+b\,|d|/\text{avgdl})}$，`--k1`、`--b` 可調。idf 用加 1 的版本，df 超過 N/2 時也不會變成負數。分數不在 [0, 1] 內。
- BM25 模式下 `--champions` 會改讀 `champions_bm25.tsv`；`--k1`/`--b` 和建 champion list 時不同的話，`-v` 會印出提醒（list 仍可用，只是挑選標準和計分標準不完全一致）。
- log 底數、「df = 0 或 df = N 的 term 不計分」、同分時 docID 小者在前，這些規則所有 scheme 都相同。

### tf index：impact ordering 與 tiered index 共用的結構（Ch.7，`tf_index.py`）

`postings.txt` 每行含 positions，排序時要用 regex 掃過整行才拿得到 (doc, tf)。`tf_index.bin` 是不含 positions 的第二份 postings：**每個 term 的文件依 tf 分組，tf 大的組在前，組內 docID 遞增**（uint32 陣列，可以直接二分搜尋）。

- 每組前面記兩個上界：這一組**和之後所有組**裡，一份文件最多能貢獻多少，分別針對 ltc cosine（$\max (1+\log tf)/\lVert d\rVert$）和 BM25（tf 部分的最大值）。寫成 float32 時乘上 $1+10^{-6}$，確保捨入後仍是上界。
- 898 萬筆 posting 分成 57 萬組，檔案 46 MB，建立約 10 – 20 秒。`tf_index.json` 記著它對應的 `postings.txt` 大小，對不上就重建。
- **`--tf-index`**：只換資料來源，排序仍是精確的，結果和預設逐 byte 相同。

**Impact ordering（`--impact`，課本 7.1.5）。**

1. query term 依「最多能貢獻多少」由大到小處理（query tf 相同時就是 idf 由大到小）。
2. 每個 term 的 postings 依 tf 由大到小一組一組讀。
3. 讀每一大組之前檢查：目前第 k 名的分數，是否已經高於「一份還沒出現過的文件最多還能拿到的分數」（這一組的上界 + 後面所有 term 的上界）。成立之後就不再建立新的 accumulator。
4. 之後剩下的 postings 只替已經有 accumulator 的文件補分數（文件少時對每組做二分搜尋），並持續丟掉追不上第 k 名的文件。

因為停止條件是安全的，**top-k 和精確排序相同**（Web Track 200 題全部一致），而且回傳的分數是完整的分數。它不是近似，而是把課本的 early termination 做成不會出錯的版本。限制：上界只存了 ltc 文件端和 BM25（預設 k1、b）兩種；其他 scheme、`-k 0`、`--min-match` 時不提早結束（`-v` 會說明），結果仍然正確。

**Tiered index（`--tiered`，課本 7.2.1）。** tier 直接用 tf 的範圍定義，門檻由 tf 分佈決定（讓 tier 1 不超過 10%、tier 1+2 不超過 35% 的 postings）：

| tier | tf | postings 佔比 |
|---|---|---|
| 1 | ≥ 4 | 9.6% |
| 2 | 2–3 | 21.8% |
| 3 | 1 | 68.6% |

查詢時先收集所有 query term 在 tier 1 的文件；不足 k 份才加入 tier 2，再不足才加入 tier 3（三層都讀完就等於精確）。候選文件決定後，它們的分數用**全部 tier** 算，所以印出的分數和精確排序時相同，只是候選集合比較小。

### Cluster pruning（`--cluster B1,B2`，課本 7.1.6，`clusters.py`）

- 前處理：隨機選 $\sqrt{N}\approx177$ 份文件當 leader（seed 固定，重建結果相同），算出每份文件和每個 leader 的 ltc cosine，記下最近的 3 個 leader。
- 查詢：算 query 和 177 個 leader 的 cosine，取最近的 B2 個；候選文件是「最近的 B1 個 leader 之一在這 B2 個裡面」的文件。候選文件的分數是完整的分數。
- 候選不足 k 份時，B2 加倍重試，直到用完所有 leader，最後退回精確排序。
- 可以和 `--impact`、`--scheme` 一起用。leader 之間的相似度固定用 ltc cosine。

### 一般形式的 wildcard：k-gram index（Ch.3，`kgram.py`）

原本中間有 `*` 或有多個 `*` 的 pattern，要先用開頭的固定字串縮小範圍，再用 regex 逐一比對；開頭就是 `*` 時（`*comput*`）等於掃過 34 萬個 term 和 40 萬個原始字形。現在照課本 3.2.2：

- 每個 index term 和原始字形前後加上 `$`，切成 3-gram，建立「gram → 字串編號」的索引（63,557 個 gram、551 萬筆，22 MB，建立約 1 – 3 秒）。
- pattern 以 `*`、`?` 切開後取出所有 3-gram（`c*t*r` → 沒有；`*comput*` → `com omp mpu put`；`m*soft` → `$m` 不足 3 字不取，`sof oft ft$`），從最短的 list 開始取交集。
- **交集只是必要條件**（`red*` 會找到 `retired`），所以最後仍用原本的 regex 過濾。
- 取不出任何 3-gram 的 pattern（如 `c*t*r`、`*a*`）沿用原本的掃描；單純的 `prefix*` 沿用 bisect。

和原本的實作比對 408 個 pattern（24 個手寫 + 384 個隨機產生），展開結果完全相同；其中 317 個走 k-gram。在開發用的 Linux 沙盒上，`*comput*` 從 85 ms 降到 1.0 ms，`*ation*al` 從 436 ms 降到 1.9 ms。

### 網頁 demo 與查詢詞標示（`ui.py`）

```
python ui.py              # http://localhost:8000
python ui.py 8765         # 換 port
```

單一檔案，只用標準函式庫（`http.server`），沒有新增套件。排序直接呼叫 `Ranker`、`parse_query`、`SpellCorrector`，所以結果和 `search.py` 相同。伺服器是單執行緒的，所有 request 共用同一個 index file handle；伺服器端的例外會把 traceback 同時印在終端機和頁面上，方便除錯。

- **輸入**：搜尋框；top-k、weighting scheme、搜尋方法（exact / champion lists / impact / tiered / cluster）、拼字模式、Soundex。頁面上的語法表列出每種功能並附範例，點範例就會直接執行。
- **輸出**：每筆結果一列，有分數長條、標題（連結到原網址）、摘要、URL 與 TREC-ID，以及詞頻表：每個查詢詞一欄（wildcard 合併成一欄，值是所有展開詞的 tf 總和），格子是該詞在文件中的 tf。列的底色表示文件含有幾個查詢詞：綠色全部、黃色部分、紅色都沒有，表格上方另有「X/k results contain all terms」，讓人一眼判斷結果是否合理。
- **標題與摘要**：WARC 是單一 gzip 串流，無法依 offset 隨機讀取，所以第一次啟動時會在背景執行緒用 HW1 的 `parse_warc` 掃一次 WARC（約 2–4 分鐘），把每份文件的標題和前 2 萬字內文寫進 `output/doc_text.tsv`（約 145 MB），並逐筆比對 TREC-ID 確認和 index 的 docID 一致。建置期間仍可搜尋，頁面會顯示進度；之後啟動時直接讀快取。摘要是含有最多不同查詢詞的 30 個字，查詢詞以和 index 相同的 tokenizer + linguistic 比對後標成黃色（所以 `business` 也會標出 `businesses`）。程式會在解壓後的資料夾（`<ID>_HW2/`）和上一層資料夾找 `07.warc.gz`；找不到時照常搜尋，只是沒有標題和摘要。

### 封裝：壓到 64 MB 以內（`pack_index.py`）

| 內容 | 大小 |
|---|---|
| `postings.txt` 原檔 | 201.1 MB |
| zip（deflate） | 63.2 MB |
| xz | 45.0 MB |
| **改存間隔後再 xz（`postings.delta.xz`）** | **31.1 MB** |

- 每行改寫成 `term<TAB>docID間隔:position間隔,...;...`。df 和 tf 不存，還原時由 posting 數和 position 數得到（封裝時會檢查兩者確實相等）。
- 還原是純 Python 標準函式庫（`lzma`），約 15 秒。先寫到暫存檔，**大小和 SHA-256 都和 `postings.pack.json` 記的相同才改名成 `postings.txt`**，所以中途被打斷不會留下壞檔，`offsets.tsv` 裡的 byte offset 也保證仍然有效。
- 可以由 postings 重建、而且重建很快的 `tf_index.*`（49 MB）和 `kgram.*`（24 MB）不放進壓縮檔，第一次用到時自動建立。其餘前處理結果都在壓縮檔裡。
- 整個壓縮檔約 55.4 MB（52.8 MiB），由 `build.py package` 產生。

### 正確性與效能（`evaluate.py`，完整輸出在 `results/evaluation.md`）

**沒有官方的測試 query**，所以用 TREC Web Track 2009–2012 的 200 個公開 topic（它們就是針對 ClueWeb09 寫的）加上 24 個手寫的邊界情況（全是 stopword、index 裡沒有的字、含 hyphen 的字、數字、小寫 `and`/`or`、重複的字、極常見的字、簡單 Boolean）。

**正確性**：Correctness: search.py against an independent brute-force scorer

| scheme | queries | documents compared | max abs. score difference | rankings identical |
|---|---|---|---|---|
| ltc.ltc | 224 (6 with no result) | 1,049,547 | 3.7e-15 | 224 / 224 |
| lnc.ltc | 60 (4 with no result) | 298,393 | 4.4e-16 | 60 / 60 |
| nnc.ntc | 60 (4 with no result) | 298,393 | 2.2e-16 | 60 / 60 |
| ntn.ntn | 60 (4 with no result) | 298,393 | 0.0e+00 | 60 / 60 |
| anc.atc | 60 (4 with no result) | 298,393 | 8.0e-16 | 60 / 60 |
| atc.atc | 60 (4 with no result) | 298,393 | 1.7e-16 | 60 / 60 |
| bnn.bpn | 60 (4 with no result) | 298,393 | 0.0e+00 | 60 / 60 |
| Lpc.ltc | 60 (4 with no result) | 298,393 | 2.0e-16 | 60 / 60 |
| Ltc.Ltc | 60 (4 with no result) | 298,393 | 1.5e-16 | 60 / 60 |
| lpc.apc | 60 (4 with no result) | 298,393 | 1.4e-16 | 60 / 60 |
| bm25 | 60 (4 with no result) | 298,393 | 3.6e-15 | 60 / 60 |

Every score of every matching document is compared (top-k = all). The brute-force scorer shares only HW1's tokenizer and record parser with search.py: it builds full document vectors from postings.txt and applies the SMART / BM25 definitions directly, including each document's vector length.

**效能**：Inexact top-10: time and agreement with the exact ranking

| method | ms / query | speed-up | postings scored / query | mean overlap@10 | identical top-10 (same order) | queries widened |
|---|---|---|---|---|---|---|
| exact, postings.txt (baseline) | 6.59 | 1.00x | 5,335 | - | - | 0 |
| exact, tf index (`--tf-index`) | 3.00 | 2.20x | 5,335 | 1.000 | 200 / 200 | 0 |
| index elimination `--idf-threshold 1.0` | 2.09 | 3.16x | 1,642 | 0.908 | 121 / 200 | 0 |
| index elimination `--min-match 2` | 6.22 | 1.06x | 5,335 | 0.630 | 78 / 200 | 0 |
| champion lists `--champions` (r = 50) | 4.32 | 1.53x | 117 | 0.982 | 184 / 200 | 4 |
| tiered index `--tiered` | 0.83 | 7.96x | 1,119 | 0.519 | 28 / 200 | 0 |
| impact ordering `--impact` | 0.79 | 8.34x | 1,924 | 1.000 | 200 / 200 | 0 |
| cluster pruning `--cluster 1,4` | 0.67 | 9.88x | 498 | 0.220 | 14 / 200 | 29 |
| cluster pruning `--cluster 3,8` | 1.29 | 5.10x | 1,404 | 0.399 | 21 / 200 | 21 |
| impact + cluster `--impact --cluster 3,8` | 0.89 | 7.42x | 860 | 0.399 | 21 / 200 | 21 |
| BM25 exact, postings.txt | 6.39 | 1.00x | 5,335 | - | - | 0 |
| BM25 `--champions` | 4.28 | 1.49x | 126 | 0.853 | 122 / 200 | 4 |
| BM25 `--tiered` | 0.82 | 7.78x | 1,119 | 0.770 | 78 / 200 | 0 |
| BM25 `--impact` | 0.69 | 9.26x | 1,520 | 1.000 | 200 / 200 | 0 |

（時間在開發用的 Linux 沙盒上量測；隊友在 Windows 筆電上量到的時間約為這裡的兩倍，相對快慢和 overlap 相同。）

200 free-text queries (TREC Web Track 2009-2012 topics), top-10, best of 3 timed passes after one warm-up pass, single process, postings in the OS file cache. Overlap is the share of the exact top-10 that the method also returns, averaged over the queries with at least one result. "Queries widened" counts queries where the pruned candidate set held fewer than 10 documents and the search fell back to a wider or exact set (that time is included).

- **沒有算 MAP / P@10**：Web Track 四年的 relevance judgment 共 79,431 筆，其中只有 47 筆屬於我們這個子集（`clueweb09-en0000-07-*`），判為相關的只有 15 筆、集中在 5 個 topic，不足以比較 scheme 的好壞。
- **tiered 和 cluster pruning 的 overlap 偏低是方法本身的性質**：tier 只看 tf，而 cosine 會除以文件長度，tf 高的多半是長文件；BM25 對長度的懲罰較輕，所以同樣的 tier 在 BM25 下 overlap 高得多。Cluster pruning 用隨機 leader，而 query 只有兩三個詞，和 leader 的相似度很不穩定。
- champion list 的 overlap 最高，但它仍要從 `postings.txt` 讀出整行才知道 tf，所以時間省得不多；`--impact` 兩者兼得。

### 新模組的 API

| 名稱 | 說明 |
|---|---|
| `weighting.Scheme(idx, spec="ltc.ltc", k1=1.2, b=0.75)` | 解析 `ddd.qqq` 或 `bm25`；格式不對時丟出 `SchemeError` |
| `.doc_weight_fn(term)` | 回傳函式 `(doc_id, tf) → 未正規化的文件端權重`（含 idf） |
| `.query_weights(qtf)` | `(term → 查詢端權重, 查詢向量長度)` |
| `.doc_norms()` | 此 scheme 文件端的向量長度（不做 cosine 時全為 1） |
| `.bound_kind(tf_index_meta)` | `tf_index.bin` 裡哪一種上界適用：`"ltc"`、`"bm25"` 或 `None` |
| `tf_index.TfIndex.open(idx)` | 開啟 tf index，不存在或過期時先建立 |
| `.blocks(term)` | `Block(tf, docs, bound_ltc, bound_bm25)` 的 list，tf 由大到小 |
| `.doc_tfs(term, allowed=None)` | `(doc_id, tf)`；給 `allowed` 時只回傳集合內的文件 |
| `.tier_docs(term, tier)` / `.tier_of(tf)` / `.tier_min_tf` | tiered index 的存取 |
| `tf_index.build(index_dir)` | 由 `postings.txt`、`doc_norms.tsv`、`doc_lengths.tsv` 建立三個檔 |
| `clusters.Clusters(idx)` / `.candidates(qtf, b1, b2)` / `.nearest_leaders(qtf)` | cluster pruning 的查詢端 |
| `clusters.build(idx, tfi, seed=27)` | 選 leader、算相似度、寫出兩個 tsv |
| `kgram.KGramIndex(index_dir)` / `.candidates(pattern)` | `(term 編號, 字形編號)`，是符合 pattern 的超集；pattern 取不出 3-gram 時回傳 `None` |
| `kgram.pattern_grams(pattern)` / `kgram.grams_of(text)` | pattern 必須包含的 3-gram／一個字串的所有 3-gram |
| `pack_index.pack(index_dir)` / `.unpack(index_dir)` / `.ensure_postings(index_dir)` | 封裝、還原、需要時才還原 |
| `Ranker(idx, ..., scheme=None)` | 新增 `scheme`；`None` 即 `ltc.ltc` |
| `Ranker.rank(..., tiered=False, impact=False, cluster=None, tf_index=False)` | 新增的四個選項；`RankResult` 多了 `notes`（給 `-v` 的說明）和 `postings_scored` |
| `Matcher.use_kgram` | 設成 `False` 可強制走原本的掃描（比對兩種實作用） |

`prep_extra.py` 的選項：`--only tf_index doc_stats clusters kgram`（只建指定的部分）、`--force`（全部重建）、`--no-verify`。驗證會抽 320 個 term 確認 tf index 和 `postings.txt` 的 (doc, tf) 完全相同、上界確實不小於實際貢獻，並抽 22 份文件用 HW1 的 parser 直接套公式重算 `doc_stats.tsv` 的每一欄（相對誤差 < 1e-9）。

## 驗證方式

每個模組都可以單獨執行，看自己的 demo：

```
python index_reader.py output          # 載入統計，以及幾個 term 的 df / idf / postings 開頭
python query_parser.py 'your query'    # 顯示 parse 結果（mode / filter / scoring / warnings）
python matcher.py 'your query'         # filter 結果數、前幾個 docID、merge 統計
python ranker.py 'your query'          # top 5 及分數
python prep.py                         # 前處理，最後會跑驗證
```

開發時在沙盒裡做過以下交叉驗證，對照組都是另外寫、演算法不同的暴力版本：

- matcher：18 個指定 query 加 1,500 組隨機 term 組合，結果和用 Python set 實作的 naive evaluator 完全一致。wildcard 展開和 `fnmatch` 掃整個字典的結果相同。
- ranker：14 個 query 的每份 candidate，分數和直接套公式的結果誤差 < 1e-16，排序相同，分數都落在 [0, 1]。
- prep：見上面「prep.py 與 hw2_index_prep.ipynb」的驗證項目；champion list 抽 320 個 term（含 df 最大的 20 個），兩種 scheme 都和完整排序一致。
- 加入 `prep.py` 和 champion list 之後，`queries.txt` 的精確排序結果和改版前逐 byte 相同。
- spell / wildcard（對照組都是獨立的實作）：
  - edit distance 和 rapidfuzz 的 OSA 實作比對 2 萬組字串，含提早結束的情況，共 6 萬次比較，結果全部相同。
  - Soundex 通過課本的範例（Hermann → H655 等），並和逐字照課本步驟、用字串替換寫成的版本比對字典中的 17 萬個字，結果全部相同。
  - 候選詞：299 個拼錯的字，和「掃過整個字典逐一計算距離」的暴力結果比對，5,387 個正確候選詞只漏掉 1 個（兩個字完全沒有共同 bigram），沒有多出不該有的候選詞。
  - wildcard：296 個隨機 pattern（前綴、後綴、中間、`?`），字形比對結果和完整展開結果都和 `fnmatch` 掃過全部字形與 term 的結果相同。
  - 修正正確率：從出現 200 次以上的字隨機製造一個錯誤（刪除、插入、替換、交換），修回原字的比例是 88.7%（399/450）；長度 7 以上的字製造兩個錯誤，修回原字的比例是 78.6%（209/266）。
  - query 改寫：引號、括號、`AND`/`OR`/`NOT`、wildcard、`/3`、`mp3` 都保持原樣。
  - 開啟 spell / wildcard 的新功能後，`queries.txt` 中原有 query 的排序結果和改版前逐 byte 相同。
- **模擬助教的環境**：把交件用的壓縮檔解到全新的資料夾，用全新的 Python 虛擬環境只安裝 `requirements.txt`，並在沒有網路、沒有任何 NLTK 資料的狀態下執行互動、單一 query 和 batch 三種模式，結果和開發環境相同。

## 已知限制

- **index 和 HW1 交件版不完全相同**：見「建置流程」一節。原始字形詞表必須和 index 在同一個平台上產生才會完全一致；`build.py` 一律在同一台電腦上依序產生兩者，所以涵蓋率是 100%。
- **很寬的 wildcard 在 Boolean 模式下很慢。** 例如 `*ing AND linux`：`*ing` 會展開成 8,290 個 term，Boolean 模式必須讀完它們全部的 postings，約需 6 – 8 秒。free text 模式只用 df 最高的 30 個來計分，不受影響。
- **拼字修正無法處理 collection 本身常見的錯字。** 網頁上 `teh`、`recieve`、`definately` 都出現數十次以上，會被視為正確的字。門檻（50 次、50 倍）是在「常見錯字」和「真實但少見的字」（`tcp` 74 次，旁邊有出現 1 萬多次的 `top`）之間取捨的結果。
- **Soundex 很粗略。** 它只看子音的類別，`fonetik` → `phonetic` 這類修正大多做不到，所以 `--phonetic` 只是在距離相同時偏好發音相近的字，不會取代 edit distance。
- **只支援 HW1 的單行 postings 格式**（`multiline=False`，也就是交件時的設定）。
- **proximity 的距離以原始 token 位置計算**，被移除的 stopword 也佔位置。
- **champion list 的加速有限**：見「Champion lists」一節的說明。

## 未做的加分項

- App 形式的介面（只做了網頁 demo `ui.py`）。
- 交件的壓縮檔裡沒有 WARC（178 MB 放不進 64 MB），所以 `ui.py` 的標題、摘要和摘要中的查詢詞標示，要把 `07.warc.gz` 放在解壓後的資料夾（`<ID>_HW2/`）或上一層才會出現；沒有 WARC 時其餘功能（排序、詞頻表）照常。
