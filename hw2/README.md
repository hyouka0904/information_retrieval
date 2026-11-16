# IR HW2：Query Processing and Searching

接續 HW1 的 positional inverted index，對使用者的 query 回傳依 TF-IDF + cosine similarity 排序的結果（投影片第 4、5 頁）。另外實作 BM25、champion lists、tiered index、phrase / proximity / wildcard / 複雜 Boolean、拼字與發音修正等加分項（完整列表見「加分項完成狀況」）。HW1 的 index 不用重建：前處理（`prep.py` 或 `hw2_index_prep.ipynb`）補上 HW2 需要的統計量，之後的查詢都是讀這些小檔案，postings 則依 byte offset 從硬碟隨讀隨用。

## 組員

| 姓名 | 學號 | 負責部分 |
|---|---|---|
| TODO | TODO | TODO |
| TODO | TODO | TODO |

## 給助教：解壓後立即執行

壓縮檔已經包含完整的 index 和所有前處理結果（`output/`），**不需要 WARC 檔，也不需要跑任何前處理**。

```
cd team27_hw2
pip install -r requirements.txt
python search.py                     # 出現 cmd> 後輸入 query，例如 Hong Kong business；exit 結束
```

- 需要 Python 3.8 以上。
- 不需要網路。NLTK 的 stopwords 已經放在 `nltk_data/`，程式會自動使用。HW1 的程式啟動時仍會嘗試下載 wordnet 和 omw-1.4，離線時會印出幾行 `[nltk_data] Error loading ...`，這是正常的：這兩份資料只有 lemmatization 會用到，而這份 index 沒有開啟 lemmatization，查詢結果不受影響。
- 第一次查詢約需 0.1 秒到數百毫秒（postings 還沒進入 OS 的 disk cache），之後每個 query 約 15 ms。
- 預設會檢查拼字：發現疑似拼錯的字時，只在 stderr 印出 `# did you mean: ...`，**排序結果仍然是原本輸入的 query 的結果**，所以不會影響評分。要讓程式自動修正後再搜尋，加上 `--spell correct`；完全關閉則用 `--spell off`。

投影片要求的 ranked list 輸出在 `results/results.txt`，是用 `queries.txt`（投影片上的範例 query 加上幾個加分項的 query）跑出來的。`results/` 的每個檔案都可以用下表的指令重新產生，重跑後應該和壓縮檔裡的檔案逐 byte 相同：

| 檔案 | 指令 | 內容 |
|---|---|---|
| `results/results.txt` | `python search.py -f queries.txt -o results/results.txt` | **必做**：TF-IDF + cosine 的精確排序 |
| `results/results_bm25.txt` | `python search.py -f queries.txt -w bm25 -o results/results_bm25.txt` | 加分：BM25 |
| `results/results_champions.txt` | `python search.py -f queries.txt --champions -o results/results_champions.txt` | 加分：champion lists |
| `results/results_tiered.txt` | `python search.py -f queries.txt --tiered -o results/results_tiered.txt` | 加分：tiered index |
| `results/results_spell_correct.txt` | `python search.py -f queries.txt --spell correct -o results/results_spell_correct.txt` | 加分：拼字修正後再搜尋 |

**用自己的 query 評分**：一行一個 query 寫進文字檔（可以寫成 `qid:query`，`#` 開頭是註解），然後 `python search.py -f 你的檔案.txt -o 結果.txt`。每個 query 前面會有一行 `cmd> query`，接著是 `doc# <TAB> 分數`，預設前 10 筆（`-k 0` 為全部）。doc# 是 HW1 的 docID；要對 ClueWeb09 的 qrels，加上 `--trec-id`（第三欄印 TREC-ID）或 `--trec-run 名稱`（輸出 `trec_eval` 的 run 格式）。

各加分功能的單一 query 範例：

```
python search.py -w bm25 Hong Kong business                 # BM25
python search.py --tiered Hong Kong business                # tiered index（加 -v 會印出用了幾層）
python search.py --champions Hong Kong business             # champion lists
python search.py "\"Hong Kong\" business"                   # phrase（引號在 shell 裡要跳脫，建議用互動模式）
python search.py "hong /3 kong"                             # proximity
python search.py "(open OR free) AND software NOT microsoft" # 複雜 Boolean
python search.py "*ology research"                          # wildcard（比對原始字形）
python search.py --spell correct infromation retreival      # 拼字修正
```

**什麼時候才需要 WARC？** 只有在 index 重建過、要重新跑前處理時。前處理需要 WARC 才能取得精確的文件數 N 和 TREC-ID；沒有 WARC 也能跑，但 N 會改用最大的 docID（這份資料剛好兩者相同，都是 31,480），而且不會重新產生 `doc_ids.tsv`。

## 檔案結構

```
team27_hw2/
├── hw1_indexer.py           HW1 原始程式，原封不動；HW2 import 它的 tokenizer / linguistic / Config / 格式解析
├── prep.py                  前處理的程式本體（命令列：python prep.py）
├── hw2_index_prep.ipynb     前處理的說明版：逐步呼叫 prep.py 的函式並寫出推導，產生的檔案和 prep.py 完全相同
├── index_reader.py          index 的唯讀存取層
├── query_parser.py          query 字串 → AST
├── matcher.py               在 index 上執行 AST 的 filter → 符合條件的 docID
├── ranker.py                TF-IDF + cosine / BM25 排序、top-k、inexact top-k（index elimination、champion lists、tiered index）
├── search.py                命令列介面（互動 / 單一 query / batch）
├── spell.py                 原始字形詞表（wildcard 用）、拼字修正、Soundex
├── queries.txt              產生 results/ 用的 query
├── requirements.txt
├── nltk_data/               NLTK stopwords，讓程式離線也能執行
├── results/
│   ├── results.txt          queries.txt 的排序結果（投影片第 5 頁的格式，TF-IDF + cosine）
│   ├── results_bm25.txt     同上，-w bm25
│   ├── results_champions.txt  同上，開啟 --champions
│   ├── results_tiered.txt   同上，開啟 --tiered
│   └── results_spell_correct.txt  同上，開啟 --spell correct
└── output/
    ├── dictionary.txt       HW1 產出：term, df
    ├── postings.txt         HW1 產出：term, df: <doc, tf: <pos, ...>; ...>
    ├── offsets.tsv          前處理產出：term \t df \t byte_offset
    ├── doc_norms.tsv        前處理產出：doc_id \t ‖d‖
    ├── doc_lengths.tsv      前處理產出：doc_id \t |d|（BM25 用）
    ├── doc_ids.tsv          前處理產出：doc_id \t trec_id \t url
    ├── champions.tsv        前處理產出：term \t doc doc ...（依 cosine 排序的 champion list）
    ├── champions_bm25.tsv   前處理產出：term \t doc doc ...（依 BM25 排序的 champion list）
    ├── tiers.tsv            前處理產出：term \t tier1 docs \t tier2 docs（依 cosine 排序的 tiered index）
    ├── tiers_bm25.tsv       前處理產出：同上，依 BM25 排序
    ├── surface_vocab.tsv    前處理產出：surface \t term \t cf \t df（原始字形 → index term）
    └── meta.json            前處理產出：N、log base、avgdl、champion / tier 設定、build 時的 Config、postings 大小
```

`07.warc.gz` 不在壓縮檔裡。要重跑前處理時，把它放在和 `prep.py` 同一層。

## 環境建置

需要 Python 3.8 以上。HW1 在 Windows 11 + Python 3.14.7 上測過，HW2 在 Linux + Python 3.12.3 與 3.13 上測過。

建議用全新的虛擬環境，避免和其他專案的套件版本衝突：

```
cd team27_hw2
python -m venv .venv
.venv\Scripts\activate              # Windows；Linux / macOS 改成 source .venv/bin/activate
pip install -r requirements.txt     # lxml、nltk、tqdm，與 HW1 相同；BM25 和 tiered index 也沒有新增套件
pip install notebook                # 只有要打開前處理 notebook 時才需要（或用 VS Code 的 Jupyter）
```

確認環境正常：`python search.py Hong Kong business` 應該印出設定（stderr）和 10 行結果，第一行是 `16103	0.407`。

- **不需要網路**：NLTK stopwords 已經放在 `nltk_data/`。HW1 的程式啟動時會嘗試下載 wordnet，離線時印出的 `[nltk_data] Error loading ...` 可以忽略（見「給助教」）。
- **硬碟**：解壓後約 280 MB，大部分是 HW1 的 `postings.txt`（200 MB）。
- **記憶體**：查詢時約 300 – 500 MB（字典、統計量、第一次用到時載入的 champion / tier / 原始字形表）。

## 使用步驟

**1. 準備 HW1 的 index。** 壓縮檔裡已經有了。如果要重建，指令是 `python hw1_indexer.py -i 07.warc.gz -o output`，約 5 分鐘。

**2. 前處理（壓縮檔裡已經做好；index 重建過才需要重跑）。** 兩種方式擇一，產生的檔案逐 byte 相同：

```
python prep.py                                # 約 5 分鐘，含驗證；加 --no-surface 約 1 分鐘
```

或打開 `hw2_index_prep.ipynb` 後 Run All。正確的結果應該是：

- `response records: 31,480`，最後一筆的 TREC-ID 是 `clueweb09-en0000-07-31479`
- `N = 31,480 (source: warc)`、`avgdl = 565.54`
- 每個 term 的 champion list 合計 1,823,089 個 docID（兩種 scheme 相同）
- tiered index 每種 scheme 都是 tier 1 有 1,263,480 個 docID、tier 2 有 1,917,466 個
- surface vocabulary 約 40 萬行，涵蓋 99.7% 以上的 index term（見「原始字形詞表」一節）
- 六項驗證都印出 `OK` / `verify: ...`

`prep.py` 會掃 WARC 兩次、`postings.txt` 三次，每一步都印出 `[n/7]` 和耗時。選項：

| 選項 | 說明 |
|---|---|
| `-i DIR` | index 資料夾，預設是 `prep.py` 旁邊的 `output/` |
| `--warc PATH` / `--no-warc` | WARC 的位置；或不讀 WARC |
| `--champion-schemes ...` | 要產生哪些 champion list：`cosine`、`bm25`，預設兩個都產生；沒選到的舊檔會被刪除 |
| `--champion-r R` | 每個 term 保留幾份文件，預設 50 |
| `--bm25-k1 K1` / `--bm25-b B` | 排序 `champions_bm25.tsv`、`tiers_bm25.tsv` 用的 BM25 參數，預設 1.2 / 0.75；也是 `search.py -w bm25` 的預設參數 |
| `--tier-sizes S1 S2 ...` | tiered index 每層的名次邊界，預設 `20 200`（tier 1 = 前 20 名、tier 2 = 第 21–200 名、其餘是最後一層）；不給數字代表不建。和 champion list 用同一組 `--champion-schemes` |
| `--no-surface` | 不重建 `surface_vocab.tsv`（沿用既有的檔案），省下約 4 分鐘 |
| `--no-verify` | 跳過驗證 |

**只要 HW1 的 index 重建過，就必須重跑這一步**：`index_reader` 啟動時會比對 postings 的檔案大小和 `meta.json` 的紀錄，對不上就拒絕執行，避免拿舊的 offset 去 seek。

**3. 查詢。**

```
python search.py                               # 互動模式，提示字元 cmd>；exit / quit / Ctrl-D / Ctrl-Z 結束
python search.py Hong Kong business            # 單一 query
python search.py -f queries.txt -o run.txt     # batch：一行一個 query，可寫 "qid:query"，# 開頭為註解
```

含有引號、`*`、括號的 query，在 shell 裡要注意跳脫；最簡單的方式是用互動模式直接輸入。

輸出格式和投影片第 5 頁一樣：

```
16103	0.407
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
| `-w`、`--weighting {tfidf,bm25}` | 計分方式：`tfidf`（預設，投影片的 TF-IDF + cosine）或 `bm25` |
| `--bm25-k1 K1` / `--bm25-b B` | BM25 參數；預設取 `meta.json` 裡建 champion list 時的值（1.2 / 0.75） |
| `--champions` | 只計算 champion list 聯集內的文件（TF-IDF 讀 `champions.tsv`，BM25 讀 `champions_bm25.tsv`） |
| `--tiered` | tiered index：先只算最高一層，不足 k 筆再往下一層（TF-IDF 讀 `tiers.tsv`，BM25 讀 `tiers_bm25.tsv`）；不能和 `--champions` 同時使用 |
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

**兩種形式的關係**：所有邏輯只寫在 `prep.py` 的函式裡。`python prep.py` 依序呼叫這些函式；notebook 一格一格呼叫同一批函式，並把推導寫在 markdown 裡。程式只有一份，所以兩邊不會不同步。開發時兩種方式都從頭跑過一次，十個輸出檔（含 `meta.json`）逐 byte 相同；加入 tiered index 後，原本的九個檔案也和加入前逐 byte 相同。

HW1 少了這些 HW2 需要的東西：trie 的 byte offset 只存在 build 時的記憶體裡、沒有把 N 存下來、沒有文件向量的長度（cosine 的分母）、沒有文件長度（BM25 用）。步驟如下：

1. **掃 WARC**：N 取 response record 的數量，這和 HW1 的編號規則一致：每筆 response 都有一個 docID，即使抽不出文字也一樣。同時記下 TREC-ID 和 URL。
2. **第一次掃 postings**：以 binary mode 讀，用累計的 byte 數記下每行的 offset。每行只用 regex 抓 `(docID, tf)`，不解析 positions，並檢查抓到的數量等於 df。同一次掃描累加 norm 需要的三個和，以及每份文件的 $|d|=\sum_t tf$。
3. **算 norm 和 avgdl**：norm 用展開式，因此掃描時不需要事先知道 N。令 $a=1+\log tf$、$L=\log N$、$\ell=\log df$：

   $$\|d\|^2=\sum_t a_t^2(L-\ell_t)^2 = L^2\textstyle\sum a^2-2L\sum a^2\ell+\sum a^2\ell^2$$

   每份文件只累加三個和，等 N 確定後一次算出長度。avgdl $=\sum_d |d| / N$。
4. **第二次掃 postings：champion lists**（見下一節）。
5. **第三次掃 postings：tiered index**（見「Tiered index」一節）。
6. **第二次掃 WARC：原始字形詞表**（見「原始字形詞表」一節）。這一步要重新解析每一頁的 HTML，約需 4 分鐘，是前處理最慢的部分。
7. **寫 `meta.json`**：最後才寫，所以它描述的一定是旁邊這一批檔案。

驗證以 HW1 的檔案和直接套公式的暴力計算當作正確答案：offsets 和 `dictionary.txt` 逐行一致、隨機 seek 讀到正確的 record、展開式算出的 norm 和直接套公式的結果相同（相對誤差 < 1e-9）、文件長度和直接加總 tf 的結果完全相同、champion list 和 tiered index 都和完整排序的結果相同、原始字形詞表的格式和對應關係正確（見下文）。

### Champion lists

每個 term 預先挑出「這個 term 貢獻最大」的前 r 份文件（預設 r = 50）。查詢時只對所有 query term 的 champion list 聯集計分，free text 的候選文件從上萬份降到一兩百份。

同一個 term 的所有 posting 共用同一個 idf，所以挑選時可以忽略 idf，只比較下表的分數：

| 檔案 | 每份文件的排序分數 | 誰使用 |
|---|---|---|
| `champions.tsv` | $(1+\log tf)\,/\,\lVert d\rVert$ | `--champions`（TF-IDF） |
| `champions_bm25.tsv` | $tf\,(k_1+1)\,/\,\big(tf + k_1(1-b+b\,\lvert d\rvert/\text{avgdl})\big)$ | `-w bm25 --champions`；建檔時的 $k_1$、$b$ 和查詢時不同會印出警告 |

- 分數相同時 docID 小的優先，所以每次產生的檔案都一樣。
- df ≤ r 的 term 直接保留全部文件。每個 term 都有一行，行的順序和 `dictionary.txt` 相同；每行的 docID 由小到大排列。
- `ranker.py` 載入時只把每行存成字串，查詢用到某個 term 時才解析它的 docID，避免每次啟動都要解析 182 萬個 ID。
- **Fallback**：Boolean、phrase、proximity 的 filter 和 champion list 取交集後，常常剩不到 k 份文件（例如 `open source AND Linux` 只剩 2 份）。這時 ranker 會依課本的建議，改用完整的 candidate 重新排序；`-v` 會印出 `champion lists gave fewer than k documents`。

效果（開發時量測，free text query，top-10）：10 個 query 中有 9 個的 top-10 和精確結果完全相同，另一個是 9/10；每個 query 的計算時間約減少 30%。時間沒有減少更多，是因為 postings 仍然要從硬碟讀出來，才知道 champion 文件的 tf；champion list 省下的是計分和 top-k 選取的部分。

### Tiered index

課本 7.2.1 的 tiered index 是 champion list 的推廣：每個 term 的 postings 依「這個 term 對該文件的貢獻」分成好幾層，查詢時由上往下，湊滿 k 份就停。

**分層**：排序分數和 champion list 相同（上一節的表），依名次切層，預設 `--tier-sizes 20 200`：

| 層 | 內容 | 存放位置 |
|---|---|---|
| tier 1 | 每個 term 的前 20 名 | `tiers.tsv` 第 2 欄 |
| tier 2 | 第 21 – 200 名 | `tiers.tsv` 第 3 欄 |
| tier 3 | 其餘 | 不另外存，就是 postings 裡的其他文件 |

每行格式是 `term \t tier1 docs \t tier2 docs`，每層的 docID 由小到大排列，行的順序和 `dictionary.txt` 相同。BM25 有自己的 `tiers_bm25.tsv`，依 BM25 的 tf 部分排序。

**查詢**（`Ranker.rank(..., tiered=True)`）：

1. 取所有 query term 的 tier 1 聯集，和 Boolean / phrase filter 取交集，對這些文件算**完整**的分數（用全部 query term，不只是 tier 裡的部分）。
2. 符合條件的文件 ≥ k 份就輸出前 k 名；否則加入每個 term 的 tier 2 重算。
3. 到了最後一層就是全部 postings，結果等於精確排序。所以 tiered index 一定會湊滿 k 份（只要真的有 k 份符合條件），不像 champion list 只能「全有或全無」地退回完整排序。

`-v` 會印出 `tiers read: 2 of 3` 這樣的資訊。

**為什麼不用課本例子的 tf 門檻**：我們先實作了「tf ≥ 10 一層、3 ≤ tf < 10 一層、其餘一層」的版本。結果和定義一致，但品質很差：tf 門檻偏好很長的文件，而 cosine 會除以 ‖d‖，真正排前面的常常是短文件。投影片的 free text query 中，tf 門檻版的前 10 名和精確結果平均只重疊 2.4 份；改成依正規化後的貢獻分層（也就是現在的作法），平均重疊 9.7 份。

**效果**（開發時量測，top-10）：

| 計分方式 | query 類型 | 和精確 top-10 的重疊 | 讀到第幾層 |
|---|---|---|---|
| TF-IDF | 7 個 free text | 6 個 10/10，1 個 8/10 | 都只用 tier 1 |
| TF-IDF | 4 個 Boolean / phrase | 10、9、10、10 | tier 1 或 2 |
| BM25 | 11 個 query | 9 個 8/10 以上；`United States government` 2/10、`software engineer OR project manager` 1/10 | tier 1 或 2 |

BM25 在常見詞組成的 query 上表現較差，是因為 BM25 的 tf 部分在 tf 稍大之後就飽和，很多文件的排序分數幾乎一樣，前 20 名不一定是加總後最好的。把 k 調大（例如 `-k 50`）時，會更常用到 tier 2。

速度：和 champion list 一樣，postings 仍然要整條讀出來才能知道候選文件的 tf，所以在這份資料上每個 query 的時間和精確排序差不多（約 12 – 17 ms）。省下的是計分的文件數：投影片的 query 平均從約 5,000 份降到約 40 份。

### BM25

`-w bm25`（`Ranker(scheme="bm25")`）改用 Okapi BM25：

$$\text{score}(q,d)=\sum_{t\in q} qtf_t\cdot\ln\!\Big(1+\frac{N-df_t+0.5}{df_t+0.5}\Big)\cdot\frac{tf_{t,d}\,(k_1+1)}{tf_{t,d}+k_1\big(1-b+b\cdot|d|/\text{avgdl}\big)}$$

- **idf**：用 $\ln(1+\frac{N-df+0.5}{df+0.5})$（Lucene 的寫法）。原始的 Robertson–Spärck Jones 形式在 df > N/2 時會變成負數，這個寫法不會。
- **query tf**：直接乘上去（相當於 $k_3\to\infty$），所以 query 裡重複的詞權重會加倍，和 TF-IDF 模式一樣會累計 query tf。
- **$|d|$ 與 avgdl**：前處理的 `doc_lengths.tsv` 與 `meta.json`。$|d|$ 是 stopword 移除、stemming 之後留在 index 裡的 token 數。ranker 第一次用到 BM25 時，把每份文件的 $k_1(1-b+b|d|/\text{avgdl})$ 算好存成陣列。
- **參數**：預設 $k_1=1.2$、$b=0.75$，可以用 `--bm25-k1`、`--bm25-b` 改。若和建 `champions_bm25.tsv` / `tiers_bm25.tsv` 時的參數不同，加上 `--champions` 或 `--tiered` 時會印出警告：檔案仍然可以用，只是挑出的文件和計分標準不完全一致。要完全一致就用同樣的參數重跑 `python prep.py --bm25-k1 X --bm25-b Y`。
- 分數不在 [0, 1]，也不做任何長度正規化（BM25 的長度正規化已經在分母裡），輸出格式仍是 `doc# \t score`。
- index elimination（`--idf-threshold`）仍然用 $\log_{10}(N/df)$ 判斷，所以同一個門檻在兩種計分方式下刪掉的是同一批 term。
- Boolean / phrase filter、wildcard、拼字修正、champion lists、tiered index 都和 TF-IDF 模式共用同一套流程，只有計分公式不同。

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
- **只保留 index 裡存在的 term。** 用目前的 `hw1_indexer.py` 在 Linux 上重新解析 WARC，結果和交出去的 HW1 index 並不完全相同：抽查前 300 份文件，有 11 份不同，差異集中在混合文字的亂碼 token（如 `ÎI4ÙÌ`）和少數 HTML 殘留。換成 Python 3.14 結果也一樣，所以原因比較可能是建 index 時的 HW1 程式版本，或 Windows 和 Linux 上 lxml 的差異。這會多出 1,968 組 index 裡沒有的 term，留著沒有用，所以會被丟掉。
- 目前這份詞表有 400,141 行，涵蓋 99.75% 的 index term、99.97% 的 token 出現次數。在建 index 的那台 Windows 電腦上重跑 `python prep.py`，涵蓋率應該會更接近 100%，驗證時會印出實際數字。
- 驗證（`verify_surface_vocab`）逐行檢查：surface 嚴格遞增、term 都在 index 裡、HW1 的 `linguistic` 處理這個 token 後剛好得到這個 term、$0 < df \le cf$。

### spell.py：wildcard 比對原始字形、拼字修正

**Wildcard。** `Matcher.expand_wildcard` 會把 pattern 同時套在 index term 和原始字形上，取兩邊得到的 term 的聯集。`comput*` 的結果和以前相同，`*ing` 則會找到 `running` → `run` 等 8,281 個 term。查詢方式：

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

預設（TF-IDF）query 端和 document 端用同樣的權重 $(1+\log tf)\cdot\log(N/df)$，分數為 $\sum_t w_{tq}w_{td}/(|q|\,|d|)$。BM25 的公式見「BM25」一節；兩者共用以下流程，只有累加的那一項和最後要不要除以長度不同。

- **query term**：從 `positive_leaves` 收集，重複的詞會累計 query tf。wildcard 只取 df 最高的前 `wildcard_limit` 個展開詞。
- **candidate**：有 filter 時就是 matcher 的結果；純 free text 時，含任一 query term 的文件都是 candidate。符合 filter 但分數為 0 的文件（如 `NOT microsoft`）仍會出現在結果裡，排在最後。
- **計算**：term-at-a-time 累加，用 `heapq.nlargest` 取 top-k；同分時 docID 小的排前面。
- **inexact top-k**：`idf_threshold`、`min_match`、`champions`、`tiered` 只縮小計算範圍。$|q|$ 一律用全部的 term 計算，所以開關這些選項不會改變分數的尺度。champion list 和 tiered index 都是把候選文件限制在某個集合內；champion list 不足 k 份時一次退回完整排序，tiered index 則一層一層往下。兩者不能同時使用。

### search.py：命令列介面

解析參數，載入 index（失敗時回傳 exit code 1），然後依參數進入互動、單一 query 或 batch 模式。每個 query 都經過 `run_one()`，它負責拼字檢查、parse、rank，並把結果格式化成 plain 或 TREC run 格式。拼字的提示和修正內容寫到 stderr（batch 模式前面加上 `[qid]`）。

## API 參考

程式碼裡沒有 docstring，每個函式的用途寫在它上一行的註解；較完整的說明集中在這裡。型別標註直接看程式碼。

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
| `SCHEMES` | 可用的計分方式 `("tfidf", "bm25")` |
| `Ranker(idx, matcher=None, wildcard_limit=30, scheme="tfidf", k1=None, b=None)` | `scheme` 選 TF-IDF + cosine 或 BM25；`k1`、`b` 為 `None` 時依序取 `meta.json` 裡建 BM25 champion list 時的值、1.2 / 0.75。`wildcard_limit` 是一個 wildcard 最多用幾個展開詞來計分（取 df 最高的）。`scheme` 不在 `SCHEMES` 時丟出 `ValueError` |
| `.rank(query, top_k=10, idf_threshold=None, min_match=1, champions=False, tiered=False)` | 回傳 `RankResult`。`top_k <= 0` 代表回傳全部 candidate。`idf_threshold`：idf 低於門檻的 query term 不計分（全部低於門檻時保留 idf 最高的一個）。`min_match`：只作用在 free text，文件至少要含幾個 query term。`champions`：只對 champion list 聯集內的文件計分，結果不足 `top_k` 筆時改用完整的 candidate 重新排序。`tiered`：由 tier 1 開始，不足 `top_k` 筆就加入下一層，最後一層等於完整排序。`champions` 和 `tiered` 同時為真時丟出 `ValueError`。內部參數 `_tier_level` 不要自己傳 |
| `RankResult` | `hits`（`(doc_id, score)`，分數高的在前）、`num_candidates`、`terms_used`（term → query tf）、`terms_dropped`、`champions_fallback`、`tiers_used` / `tiers_total`（tiered 模式讀了幾層 / 總共幾層，含不另外存的最後一層；其他模式為 `None`） |
| `.query_terms(query)` | 處理過的 term → 在 query 中的次數，只取不在 NOT 底下的葉節點 |
| `.champion_docs(term)` | 一個 term 的 champion doc_id；依 `scheme` 讀 `champions.tsv` 或 `champions_bm25.tsv`（檔名取自 `meta.json`），每個 process 只讀一次，每個 term 第一次用到時才解析 |
| `.tier_docs(term, level)` | 一個 term 在 tier 0 到 `level`（從 0 起算）的 doc_id 聯集；依 `scheme` 讀 `tiers.tsv` 或 `tiers_bm25.tsv`，同樣延遲解析 |
| `.stored_tiers()` | 檔案裡存了幾層（= `meta.json` 的 `tiers.sizes` 長度）；沒有建 tiered index 時丟出 `FileNotFoundError` |
| `.bm25_idf(term)` | $\ln(1+\frac{N-df+0.5}{df+0.5})$，不在 index 的 term 為 0 |
| `.bm25_denominators()` | 每份文件的 $k_1(1-b+b|d|/\text{avgdl})$，讀 `doc_lengths.tsv` 後快取；缺檔或 `meta.json` 沒有 avgdl 時丟出 `FileNotFoundError` |
| `.champion_params_differ(section="champions")` | BM25 模式下，`meta.json` 的 `champions`（或 `"tiers"`）記錄的 $k_1$、$b$ 和目前的不同時回傳 `True` |
| `.doc_weight(tf, idf)` / `.query_weight(qtf, idf)` | 投影片的權重公式 |

公式：TF-IDF 為 $w_{t,d}=(1+\log tf_{t,d})\log(N/df_t)$，query 端相同，$\text{score}(q,d)=\sum_t w_{t,q}w_{t,d}/(|q|\,|d|)$；BM25 見「BM25」一節。

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
| `build_arg_parser()` / `print_settings(args, ranker, out)` | 命令列選項；啟動時把所有設定（含計分方式和 BM25 參數）印到 stderr |

### prep.py

notebook 依序呼叫的就是這些函式；`run()` 是 `python prep.py` 的完整流程。

| 名稱 | 說明 |
|---|---|
| `run(index_dir, warc_path, log_base, schemes, r, k1, b, verify, seed, surface, tier_sizes)` | 依序執行全部步驟和驗證，回傳 `meta.json` 的內容；`tier_sizes` 不是正的遞增數列時丟出 `ValueError` |
| `scan_warc_doc_ids(warc_path)` | `[(doc_id, trec_id, url), ...]`，編號方式和 `hw1.parse_warc` 完全相同（每筆 response record 一個 id） |
| `iter_postings_records(postings_path)` | 逐一回傳 `postings.txt` 每行的 `(byte offset, term, df, [(docID, tf), ...])`（bytes），並檢查格式和 df |
| `scan_postings(postings_path, log_base)` | 第一次掃描，回傳 `PostingsScan`：`term_entries`（term, df, offset）、norm 的三個和 `s0`/`s1`/`s2`、`doc_lengths`、每個 term 的 `term_cf`、`num_postings`、`postings_bytes` |
| `decide_n(scan, warc_docs)` | `(N, N_source, max_doc_id)`；有 WARC 時 N 取 response record 數，否則取最大 doc_id |
| `compute_doc_norms(scan, N, log_base)` | 用展開式 $\sqrt{L^2S_0-2LS_1+S_2}$ 算出每份文件的 $\lVert d\rVert$ |
| `average_doc_length(doc_lengths, N)` | avgdl |
| `write_champions(...)` / `top_r_docs(...)` / `champion_scorers(...)` | 第二次掃描，逐 scheme 寫出 champion list；每個 term 依該 scheme 的分數取前 r 份，同分時 doc_id 小的優先，輸出依 doc_id 遞增 |
| `write_tiers(...)` / `split_tiers(pairs, scorer, sizes)` | 第三次掃描，逐 scheme 寫出 tiered index；排序方式和 champion list 相同，依 `sizes` 切成各層，每層依 doc_id 遞增。`tier_sizes` 為空時不寫檔，並刪除舊檔 |
| `scan_surface_vocab(warc_path, cfg)` | `(原始字形, term) → [cf, df]`，用 HW1 的 tokenize + linguistic |
| `write_surface_vocab(index_dir, vocab, index_terms)` | 只寫出 term 在 index 裡的字形；回傳（寫出行數, 丟棄組數） |
| `surface_step(index_dir, warc_path, scan, enabled)` | 有 WARC 且 `enabled` 時重建詞表；否則若既有的檔案是針對同一份 `postings.txt` 建的就沿用。回傳 `meta.json` 的 `surface_vocab` 欄位（沒有可用的詞表時為 `None`） |
| `write_tables(...)` / `build_meta(...)` / `write_meta(...)` | 寫出表格檔與 `meta.json`（一律 LF 換行） |
| `verify_dictionary` / `verify_seeks` / `verify_doc_stats` / `verify_champions` / `verify_tiers` / `verify_surface_vocab` | 各項驗證，失敗時丟出 `AssertionError` |

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
- BM25（對照組直接 parse `postings.txt`，不經過專案的程式）：
  - 自己加總每份文件的 tf，31,480 份的 $|d|$ 都和 `doc_lengths.tsv` 相同，avgdl 和 `meta.json` 相同。
  - 6 個 free text query（含重複詞的 `business business hong`）的前 20 名，doc 和分數都和暴力 BM25 相同（誤差 < 1e-6）。
  - 3 個 Boolean / phrase query：每個結果的分數都正確；把 filter 出來的整組文件用暴力 BM25 重新排序，前 20 名也相同。
  - `champions_bm25.tsv` 抽查的 term 都是正確的 top-50；`-w bm25 --champions` 的結果等於「暴力 BM25 只在 champion 聯集裡排序」。
- tiered index：
  - `prep.py` 的 `verify_tiers` 抽 320 個 term（含 df 最大的 20 個），兩種 scheme 的每一層都和完整排序後切出來的結果相同。tier 1 + tier 2 的 docID 總數也和「每個 term 取 min(df, 200)」的加總相同。
  - 查詢端：另外寫的暴力版直接從 postings 算出每個 term 的名次、決定要讀到第幾層、在候選文件上算精確分數。11 個 query × 兩種計分方式 × k = 10 / 50，共 44 組，前 k 名、分數、讀到的層數全部相同。
- 加入 BM25 和 tiered index 後，`results.txt`、`results_champions.txt`、`results_spell_correct.txt` 重跑的結果和加入前逐 byte 相同。
- **模擬助教的環境**：把交件用的壓縮檔解到全新的資料夾，用全新的 Python 虛擬環境只安裝 `requirements.txt`，並在沒有網路、沒有任何 NLTK 資料的狀態下執行互動、單一 query 和 batch 三種模式，結果和開發環境相同。

## 已知限制

- **原始字形詞表無法和 HW1 index 完全一致**（見「原始字形詞表」一節），約 0.25% 的 index term 沒有對應的字形；這些 term 仍然可以用 stem 的 wildcard 找到。
- **很寬的 wildcard 在 Boolean 模式下很慢。** 例如 `*ing AND linux`：`*ing` 會展開成 8,281 個 term，Boolean 模式必須讀完它們全部的 postings，約需 8 秒。free text 模式只用 df 最高的 30 個來計分，不受影響。
- **拼字修正無法處理 collection 本身常見的錯字。** 網頁上 `teh`、`recieve`、`definately` 都出現數十次以上，會被視為正確的字。門檻（50 次、50 倍）是在「常見錯字」和「真實但少見的字」（`tcp` 74 次，旁邊有出現 1 萬多次的 `top`）之間取捨的結果。
- **Soundex 很粗略。** 它只看子音的類別，`fonetik` → `phonetic` 這類修正大多做不到，所以 `--phonetic` 只是在距離相同時偏好發音相近的字，不會取代 edit distance。
- **只支援 HW1 的單行 postings 格式**（`multiline=False`，也就是交件時的設定）。
- **proximity 的距離以原始 token 位置計算**，被移除的 stopword 也佔位置。
- **champion list 和 tiered index 的加速有限**：postings 仍然要整條讀出來，見「Champion lists」、「Tiered index」兩節的說明。
- **tiered index 在 BM25、常見詞組成的 query 上和精確結果差異較大**：見「Tiered index」一節的表格。這是 inexact top-k 本來就有的取捨，需要精確結果時不要加 `--tiered`。

## 加分項完成狀況

對照投影片的 optional features：

| 投影片項目 | 狀態 | 用法 |
|---|---|---|
| phrase、proximity、wildcard、substring、複雜 Boolean | 完成 | 見「查詢語法」；substring 用 `*olog*` 這種兩端都有 `*` 的 wildcard |
| spell correction、phonetic correction | 完成 | `--spell suggest/correct`、`--phonetic` |
| 其他 term weighting | 完成：BM25 | `-w bm25` |
| index elimination | 完成 | `--idf-threshold`、`--min-match` |
| champion lists | 完成（cosine 與 BM25 兩種排序） | `--champions` |
| tiered index | 完成（cosine 與 BM25 兩種排序） | `--tiered` |
| Web / App 介面、query term highlighting | 未實作 | — |
