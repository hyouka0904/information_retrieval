# HW3 LLM + RAG Implementation Plan (team27), v2

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Status:** agreed on 2026-11-25. 陳長壻 accepted every proposal in `CONSENSUS.md`, so execution can start. Still open: each member's Python version and OS. If an agreed decision changes, update this plan first.

**Goal:** A local, offline RAG system. A user prompt goes to a retriever over our HW1 index of `07.warc.gz`. The top passages go into the final prompt, and a local open LLM (Ollama) answers with citations. Every answer can be shown side by side with the same LLM answering **without** RAG.

**Architecture:** One retriever is built now: **Method 2, lexical**, the HW2 `Ranker` with BM25 over the HW1 index. Methods 1 (vector DB) and 3 (hybrid) are planned as later tasks behind the same `Retriever` interface. Adding one means a new module plus a config value, with no change to the pipeline, CLI or other retrievers. The LLM sits behind a `ChatModel` interface, so the model or the runtime can change through `rag_config.json` alone. Special tokens are injected into our part of the collection (`07.warc.gz`), and the HW1 index is rebuilt from the modified WARC.

**Tech Stack:** Python 3.10+, Ollama (default chat model `qwen2.5:7b`), the HW1/HW2 code as-is, numpy, pytest. HTTP to Ollama uses the standard library (`urllib.request`), with no LangChain/LlamaIndex.

**Spec:** `HW3.pdf` (11 slides). Prior work reused: `../hw2/team27_hw2/` (HW1 indexer + HW2 search; its README documents every module).

## Retrieval methods: evaluation and order

| | Method 1: vector DB | **Method 2: lexical (HW2 search)** | Method 3: hybrid |
|---|---|---|---|
| What it is | Embed passages of every doc; cosine search | BM25 over the HW1 inverted index; best window of each hit as the passage | Run 1 and 2, fuse the ranks (RRF) |
| Reuses HW1/HW2 | Only the doc text | Index, `Ranker`, BM25, `--impact`: all of it | Both |
| New code | Chunker, embedding build (resumable), vector store, search | Prompt sanitizer, passage window, wrapper | RRF (~50 lines) on top of 1 and 2 |
| Extra build | Embed ~100k chunks on an RTX 3060: estimated 10–30 min, ~200–400 MB of vectors | None beyond the index rebuild | Same as Method 1 |
| Special tokens (`115598063`) | Weak: embeddings blur rare numeric strings | **Exact match**: rare term, top BM25 score | Good, because the lexical list carries it |
| Prompts reworded from the page | Strong | Weaker (needs shared stems) | Strong |
| Chinese prompts about English pages | Good with a multilingual embedder (`bge-m3`) | Fails (no shared terms) | Good |
| Shows off the HW1 index (the spec's stated input) | Indirectly | **Directly** | Directly |
| Risk to the graded minimum | Medium | **Low** | Medium (depends on 1) |

**Decision (agreed): build Method 2 first.** The minimum grade checks two things, prompts quoting collection content and prompts with our special tokens, and lexical retrieval is the most reliable on both. It reuses the most tested code and needs no extra build. Methods 1 and 3 are Tasks F1 and F2, done only after the core works. They are bonus material and also fix the reworded-prompt and Chinese-prompt weaknesses.

**Room for change, built in from Task 0:**
- `Retriever` protocol + `make_retriever(name, cfg)`. The registry already lists `lexical`, `vector` and `hybrid` as lazy module paths, so adding F1/F2 never edits files owned by the other member.
- `ChatModel` and `Embedder` protocols + `make_llm(cfg)`. The backend and model are config values.
- The `--retriever` CLI flag and `evaluate_retrieval.py` iterate over whatever names the registry can build, so the comparison tables grow by themselves.

## Team split: two independent tracks

| Member | Student ID | Track | Owns (files) |
|---|---|---|---|
| 陳長壻 | 115598095 | **A: data + retrieval** | `special_tokens*.json`, `inject_tokens.py`, `rebuild_index.py`, `rag/doc_store.py`, `rag/lexical.py`, `prompts/eval_prompts.jsonl`, `evaluate_retrieval.py`; later `rag/vector.py`, `rag/hybrid.py` |
| 鄧存正 | 115598063 | **B: generation + interface** | `rag/types.py`, `rag/config.py`, `rag/registry.py`, `rag_config.json`, `canonical_index.json`, `rag/llm.py`, `rag/prompting.py`, `rag/pipeline.py`, `rag_cli.py`, `run_snapshots.py`, `evaluate_answers.py`, `rag_ui.py`, `make_zip.py`, READMEs |

**Each file has exactly one owner, and nobody edits the other track's files.** If you need a change in the other track, ask its owner.

**How the tracks stay independent:**
- **One shared contract**, `rag/types.py` (Task 0), is agreed in `CONSENSUS.md` and then frozen. Changing it needs both members.
- **Track B never needs Track A's code to make progress.** It develops and tests against `StaticRetriever` (in `types.py`) and the existing HW2 index. It switches to the real retriever by setting `"retriever": "lexical"` in config once A2 lands.
- **Track A never needs the LLM.** All of A's tests are retrieval-only. Highlighting is A's job too: each `Passage` carries the words to mark (`highlights`), so B's formatter needs no tokenizer.
- **The rebuilt index is not handed over as files.** Each member runs `rebuild_index.py` locally. The committed `special_tokens.lock.json` fixes which docs get which token, so both machines inject the same pages. `canonical_index.json` records the postings SHA-256 of the **canonical Windows build**, which is used for snapshots and the submission. It is owned by 鄧存正, who builds on Windows; it is a separate file so nobody writes to A's lock file. On Linux the SHA may differ because lxml extracts text slightly differently (found in HW2), so a mismatch there is a warning, not an error.

**Hand-off points (data only, no code swaps):**

| When | From → to | What |
|---|---|---|
| End of Task 0 | B → A | `rag/types.py`, `rag/registry.py`, `rag_config.json` |
| End of A1 | A → B | `inject_tokens.py`, `rebuild_index.py`, `special_tokens.lock.json` (B rebuilds locally) |
| End of A2 | A → B | `rag/lexical.py` (B sets `"retriever": "lexical"`) |
| End of A3 | A → B | `prompts/eval_prompts.jsonl` (B runs snapshots on it) |
| Task B5 | A → B | README text for the retrieval/index sections + A's lines for "Major difficulties" |

**Order:** Task 0 (B, short) → in parallel: A1 → A2 → A3 ‖ B1 → B2 → B3 → B4 → B5. A can start A1 during Task 0 because injection does not use `types.py`. F1/F2 (A) and B6 (B) are optional, after B4 has produced the graded snapshots.

## Global Constraints

- Run fully offline except for Ollama at `http://localhost:11434`. Chat calls use `temperature 0`, `seed 27` and an explicit `num_ctx` (default 8192), so snapshots are reproducible.
- Default models are set for an RTX 3060-class GPU (6–12 GB VRAM): chat `qwen2.5:7b` (Q4, ~4.7 GB). Alternatives are listed in `rag_config.json` comments and the README: `llama3.1:8b`, `qwen3:8b` (thinking turned off), and `gemma3:12b` (12 GB cards only). Embedder for F1: `bge-m3` (multilingual, no prefixes). Alternative `nomic-embed-text` needs the `search_query: ` / `search_document: ` prefixes, which are config keys.
- The HW1/HW2 code is copied into `team27_hw3/` unchanged. New code lives in `rag/` and new top-level scripts.
- The index is rebuilt with the same HW1 `Config` as HW2's `output/meta.json`: title included, hyphen `split`, case folding, stopwords removed, Porter stemming, no lemmatization, single-line postings, `pos_base` 0.
- Every displayed answer shows both: **Without RAG** (same model, same options, no documents) and **With RAG** (sources + answer). The spec requires the difference.
- Sources are printed as `Document ID# <doc_id> (<trec_id>) <url>` followed by the passage excerpt with `highlights` marked as `**word**`, as on slide 9.
- Submission (slide 10): source + library configuration (`requirements.txt`, `rag_config.json`, model names), snapshots of prompts and responses, and a one-page doc with Major features / Major difficulties / Special execution requirements / Team members list with each member's parts.

## Review Focus

1. **Natural-language prompts hit HW2 query syntax.** A trailing `?` is parsed as a wildcard; quotes, `*`, `/3`, parentheses and uppercase `AND`/`OR`/`NOT` switch HW2 into phrase/Boolean mode. Expected: every prompt is treated as free text. Test: A2 `test_sanitize_prompt_*`.
2. **The injected text is not in the passage the LLM sees.** `doc_text.tsv` keeps only the first 20,000 body characters, and the passage window may miss the sentence. Expected: an injected doc's passage contains its token. Tests: A1 `test_injected_fact_in_body_prefix`, A2 `test_special_token_rank1`.
3. **Context silently truncated by Ollama.** Ollama drops the *start* of a prompt longer than `num_ctx`, which is where the instructions are. Expected: passages are trimmed to `context_char_budget` before sending. Test: B2 `test_messages_fit_budget`.
4. **Nothing to retrieve, or Ollama not ready.** Examples: a stopword-only prompt, a word not in the index, Ollama not running, the model not pulled. Expected: one line (`no relevant documents found; answering without context`, `Ollama is not running: start it with "ollama serve"`, `run "ollama pull qwen2.5:7b"`), never a traceback. Tests: A2 `test_stopword_only_prompt_returns_empty`, B1 `test_check_*`, B2 `test_pipeline_no_hits_falls_back`, B3 `test_ollama_down_exit_code`.
5. **Retrieved documents that do not answer the question.** Lexical search always returns something when any term matches. Expected: the LLM says so instead of inventing an answer and citing it. Test: B2 `test_unanswerable_says_so` (marker `llm`).

---

## File structure

```
team27_hw3/
├── (from HW2, unchanged) hw1_indexer.py index_reader.py query_parser.py matcher.py ranker.py weighting.py
│   spell.py tf_index.py kgram.py clusters.py prep.py prep_extra.py pack_index.py search.py ui.py nltk_data/
├── rag_config.json            [B] all tunables: retriever name, LLM backend/model, budgets, per-retriever settings
├── rag/
│   ├── types.py               [B, shared contract] Passage, Retriever, ChatModel, Embedder, StaticRetriever
│   ├── config.py              [B] load + validate rag_config.json
│   ├── registry.py            [B] make_retriever(name, cfg), make_llm(cfg), available_retrievers(cfg)
│   ├── llm.py                 [B] OllamaClient (ChatModel + Embedder)
│   ├── prompting.py           [B] final prompt, source formatting
│   ├── pipeline.py            [B] RagPipeline
│   ├── doc_store.py           [A] doc_text.tsv build/access, passage windows
│   ├── lexical.py             [A] Method 2
│   ├── vector.py              [A, later F1] Method 1
│   └── hybrid.py              [A, later F2] Method 3
├── special_tokens.json        [A] tokens + fact sentences (input)
├── special_tokens.lock.json   [A] target doc ids per token (output, committed)
├── canonical_index.json       [B] postings SHA-256 + platform of the canonical Windows build (committed)
├── inject_tokens.py           [A] 07.warc.gz → 07_team27.warc.gz
├── rebuild_index.py          [A] indexer → prep → prep_extra → doc_text (→ vectors, after F1)
├── prompts/eval_prompts.jsonl [A] graded prompt set with expected docs/answers
├── evaluate_retrieval.py      [A] → results/retrieval.md
├── rag_cli.py                 [B] prompt> loop, --compare, batch
├── run_snapshots.py           [B] → results/snapshots.md
├── evaluate_answers.py        [B] → results/answers.md
├── rag_ui.py                  [B, bonus B6] web page, both answers side by side
├── make_zip.py                [B]
├── tests/                     test_<module>.py; markers: index (needs rebuilt index), llm (needs Ollama)
└── README.md, README_onepage.md / README.pdf   [B assembles; A supplies retrieval sections]
```

---

### Task 0: Repo, shared contract, config (鄧存正)

**Files:**
- Create: `team27_hw3/` (copy of `../hw2/team27_hw2/` without `results/`, `output/tf_index.*`, `output/kgram.*`), `.gitignore`, `pytest.ini`, `tests/conftest.py`, `rag/__init__.py`, `rag/types.py`, `rag/config.py`, `rag/registry.py`, `rag_config.json`, `tests/test_contract.py`
- Modify: `requirements.txt` (add `numpy`, `pytest`)

**Interfaces (the shared contract, `rag/types.py`):**

```python
@dataclass(frozen=True)
class Passage:
    doc_id: int                 # HW1 docID, 1..N
    trec_id: str
    url: str
    title: str
    text: str                   # exactly what goes into the final prompt
    score: float                # retriever-specific, higher is better
    source: str                 # retriever name that produced it
    highlights: tuple[str, ...] = ()   # surface words in `text` to mark in the output

class Retriever(Protocol):
    name: str
    def search(self, prompt: str, k: int) -> list[Passage]: ...   # best first; [] if nothing matches

class ChatModel(Protocol):
    model: str
    def chat(self, messages: list[dict]) -> str: ...
    def check(self) -> None: ...      # raises LLMUnavailable with a one-line fix

class Embedder(Protocol):
    def embed(self, texts: list[str], kind: Literal["query", "document"]) -> np.ndarray: ...  # float32, L2-normalized rows

class LLMUnavailable(RuntimeError): ...

class StaticRetriever:          # fixed passages; for tests and for track B before A2 lands
    def __init__(self, passages: list[Passage], name: str = "static"): ...
```

- `rag/registry.py`: `RETRIEVERS = {"lexical": "rag.lexical:LexicalRetriever", "vector": "rag.vector:VectorRetriever", "hybrid": "rag.hybrid:HybridRetriever"}` and `LLM_BACKENDS = {"ollama": "rag.llm:OllamaClient"}`. Every class is built as `Cls(cfg: dict)`. Provides `make_retriever(name: str, cfg: dict) -> Retriever`, `make_llm(cfg: dict) -> ChatModel`, and `available_retrievers(cfg) -> list[str]` (the names whose module imports).
- `rag_config.json` (defaults):

```json
{
  "index_dir": "output",
  "retriever": "lexical",
  "top_k": 5,
  "context_char_budget": 16000,
  "llm": {"backend": "ollama", "host": "http://localhost:11434", "model": "qwen2.5:7b",
          "num_ctx": 8192, "temperature": 0, "seed": 27, "timeout_s": 300},
  "lexical": {"scheme": "bm25", "k1": 1.2, "b": 0.75, "passage_words": 150},
  "vector": {"embed_model": "bge-m3", "query_prefix": "", "doc_prefix": "",
             "chunk_words": 150, "chunk_stride": 120, "max_chunks_per_doc": 4},
  "hybrid": {"members": ["lexical", "vector"], "rrf_k": 60, "retrieve_k": 20}
}
```

- [ ] **Step 1:** Copy the HW2 folder to `E:\ir_hw\hw3\team27_hw3`. Put `07.warc.gz` next to the folder, not inside it. Run `git init`, and add `.gitignore` entries for `output/`, `*.warc.gz`, `.venv/`.
- [ ] **Step 2: Write the failing tests** in `tests/test_contract.py`:
  - `test_static_retriever_returns_first_k`.
  - `test_make_retriever_unknown_name`: `make_retriever("nope", cfg)` raises `ValueError` listing `lexical, vector, hybrid`.
  - `test_make_retriever_not_built_yet`: a registered name whose module is missing raises `NotImplementedError` naming the module.
  - `test_load_config_defaults`: `load_config()` fills missing keys with the defaults above, and an unknown `llm.backend` raises `ValueError`.
- [ ] **Step 3:** Run `pytest tests/test_contract.py -v`. Expected: FAIL.
- [ ] **Step 4:** Implement `types.py`, `config.py`, `registry.py`. `conftest.py` skips `llm` tests when `{host}/api/tags` does not answer, and `index` tests when `output/meta.json` is missing.
- [ ] **Step 5:** Run the tests. Expected: PASS. `python search.py "Hong Kong business"` still prints 10 lines.
- [ ] **Step 6:** Commit `chore: HW3 repo, shared contract, config`, and send `rag/types.py` to 陳長壻.

---

## Track A: data + retrieval (陳長壻)

### Task A1: Inject special tokens and rebuild the index

**Files:**
- Create: `special_tokens.json`, `inject_tokens.py`, `rebuild_index.py`, `rag/doc_store.py` (build part only), `tests/test_inject.py`
- Output: `../07_team27.warc.gz`, `special_tokens.lock.json`, rebuilt `output/` incl. `doc_text.tsv`

**Interfaces:**
- Produces: `inject(src: str, dst: str, entries: list[dict], seed: int = 27) -> list[dict]`.
- Produces: `special_tokens.lock.json` = `{"warc_sha256": str, "entries": [{"token": str, "fact": str, "doc_ids": [int, int], "trec_ids": [str, str]}]}`. Track B's eval prompts and the special-token tests read it.
- Produces: `build_doc_text(warc: str, index_dir: str) -> None`, writing `output/doc_text.tsv` in HW2 `ui.py`'s format (`title\tbody[:20000]`, line n = doc n, `\r` → space), and asserting TREC-IDs match the index. CLI: `python -m rag.doc_store build --warc PATH`.

`special_tokens.json` (final wording is a consensus item; the facts are invented so no LLM can know them):

```json
[
  {"token": "115598095", "fact": "Team27 registry: student 115598095 maintains the retriever, and the project codename for 115598095 is BLUE-HERON-7."},
  {"token": "115598063", "fact": "Team27 registry: student 115598063 maintains the generator, and the project codename for 115598063 is AMBER-LYNX-3."},
  {"token": "zorvian lantern", "fact": "The Zorvian Lantern Festival is held every year on March 14 in Keelung, where 2,718 paper lanterns are released over the harbor."}
]
```

- [ ] **Step 1: Write the failing tests** against a 5-record WARC built inside the test (gzip + hand-written `WARC/0.18` records):
  - `test_inject_preserves_record_count`: `parse_warc(dst)` and `parse_warc(src)` yield the same number of documents.
  - `test_injected_fact_in_body_prefix`: each fact appears in `doc.body[:20000]` of each target doc.
  - `test_other_docs_unchanged`: every non-target doc has identical `(trec_id, url, title, body)`.
  - `test_content_length_updated`: `hw1.iter_warc_records(dst)` gives contents whose length equals each `Content-Length`.
  - `test_refuses_token_already_in_index` (marker `index`): `inject` raises `ValueError` unless every token has at least one term with df = 0 in the current index (`115598095`, `115598063`, `zorvian`).
- [ ] **Step 2:** Run `pytest tests/test_inject.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement `inject()`.** Read the WARC as raw records (header block up to the blank line, `Content-Length` bytes, separator) and write all bytes back unchanged except in target records. In a target, insert `<p>{fact}</p>` right after the `<body…>` tag (else after `</head>`, else at the start of the HTTP payload), then rewrite `Content-Length`. If `special_tokens.lock.json` exists, reuse its doc ids. Otherwise choose 2 targets per entry with `random.Random(seed)` among response records with a utf-8/iso-8859-1/ascii charset and ≥ 300 body words, using no doc twice. Output is one gzip stream. CLI: `python inject_tokens.py -i ../07.warc.gz -o ../07_team27.warc.gz`.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5: `rebuild_index.py`** (plain Python with `subprocess`, so it runs on Windows and Linux). Runs `python inject_tokens.py …`, `python hw1_indexer.py -i ../07_team27.warc.gz -o output` (~5 min), `python prep.py --warc ../07_team27.warc.gz` (~5 min), `python prep_extra.py`, `python -m rag.doc_store build --warc ../07_team27.warc.gz`. Then it handles the postings SHA-256. With `--canonical` (the Windows build, run by 鄧存正), it writes `canonical_index.json` (`postings_sha256`, `platform`, `python`, `built_at`). Otherwise it compares against that file and prints a warning on a mismatch, then checks that every lock entry's docs rank first for its token (exit 1 if not).
- [ ] **Step 6: Verify.** `prep.py` prints `response records: 31,480` and `N = 31,480`. `python search.py 115598063 --trec-id` and `python search.py "zorvian lantern" --trec-id` each list exactly the lock file's 2 doc ids at the top.
- [ ] **Step 7:** Commit `feat: inject team27 special tokens, rebuild script`. Tell 鄧存正 to run `rebuild_index.py`.

### Task A2: Lexical retriever (Method 2)

**Files:**
- Create: `rag/lexical.py`, `tests/test_lexical.py`
- Modify: `rag/doc_store.py` (add the read side)

**Interfaces:**
- Consumes: `Passage`, `Retriever` (Task 0); HW2 `IndexReader`, `Ranker`, `weighting.Scheme`, `parse_query`.
- Produces: `class DocStore(index_dir: str, idx: IndexReader)` with `.title(doc_id) -> str`, `.body(doc_id) -> str`, `.terms_of(word: str) -> frozenset[str]`, `.window(doc_id, terms: frozenset[str], width: int) -> tuple[str, tuple[str, ...]]` (text, highlighted words).
- Produces: `sanitize_prompt(text: str) -> str` and `class LexicalRetriever(cfg: dict)` with `name = "lexical"` and `.search(prompt, k) -> list[Passage]`. F1/F2 will reuse `DocStore`.

- [ ] **Step 1: Write the failing tests:**
  - `test_sanitize_prompt_question_mark`: `sanitize_prompt("When was the premiere of The Magic Flute?") == "When was the premiere of The Magic Flute"`.
  - `test_sanitize_prompt_operators`: the result of `sanitize_prompt('Is "Linux" AND (GNU) /3 free* OR NOT')` has none of `" ( ) / * ?` and no uppercase `AND`/`OR`/`NOT` word, and `parse_query(result, cfg).mode == "free"`.
  - `test_window_picks_term_cluster`: on a synthetic body, `window(..., {"magic", "flute"}, 10)` contains both words, and `highlights` holds the surface forms (`Magic`, `Flute`). It uses HW2 `ui.doc_text`'s rule: start on a query term, most distinct terms first, then most hits.
  - `test_window_no_terms_returns_prefix`.
  - `test_stopword_only_prompt_returns_empty` (marker `index`): `search("what is the", 5) == []`.
  - `test_special_token_rank1` (marker `index`): for each lock entry, `search(token, 5)[0].doc_id` is in its `doc_ids`, and that passage's `text` contains the token.
- [ ] **Step 2:** Run `pytest tests/test_lexical.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** `sanitize_prompt` replaces `" ( ) / * ?` with spaces and lowercases standalone `AND`/`OR`/`NOT`. Ranking is `Ranker(idx, scheme=Scheme(idx, "bm25", k1, b)).rank(q, top_k=k, impact=True)`, which is exact and fast. `passage_words` comes from config. `DocStore` keeps byte offsets of `doc_text.tsv` lines and seeks per read; `terms_of` = HW1 `tokenize` + `linguistic` with `idx.cfg`, `lru_cache`d.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit `feat: lexical retriever over HW1 index (Method 2)` and tell 鄧存正 to switch config to `lexical`.

### Task A3: Evaluation prompt set and retrieval evaluation

**Files:**
- Create: `prompts/eval_prompts.jsonl`, `evaluate_retrieval.py`, `tests/test_evaluate_retrieval.py`
- Output: `results/retrieval.md`

**Interfaces:**
- Produces: one JSON object per line: `{"id": "P01", "type": "special" | "collection", "lang": "en" | "zh", "prompt": str, "expected_docs": [int], "expected_answer": [str]}`. An answer counts as correct if it contains any `expected_answer` string, case-insensitive. B's snapshot and answer scripts consume this.
- Produces: `evaluate(prompts: list[dict], retrievers: dict[str, Retriever], k: int = 5) -> list[dict]` (per retriever: hit@1, hit@5, mean ms, broken down by `type` and `lang`).

- [ ] **Step 1: Write the prompt set.**
  - **6 special prompts:** each token bare, as on slide 9 (`115598063`), and as a question (`What is the project codename for 115598095?`, `When is the Zorvian Lantern Festival held?`).
  - **12 collection prompts:** each written from a fact read in a specific document (record its doc id) on different topics. Prefer facts a 7B model is unlikely to know (local news, product details, dates on 2009 pages).
  - Of the 12, **2 are reworded** so they share few words with the page, and **2 are in Chinese**. These 4 measure the lexical weaknesses that would justify F1/F2.
- [ ] **Step 2: Write the failing tests:** `test_hit_at_k` (a `StaticRetriever` with known docs gives hit@1 = 0.5, hit@5 = 1.0 on a 2-prompt set); `test_iterates_available_retrievers` (only names that `available_retrievers` returns are evaluated, and the others are listed as "not built").
- [ ] **Step 3:** Run the tests. Expected: FAIL. **Step 4:** Implement. **Step 5:** Run the tests. Expected: PASS.
- [ ] **Step 6:** Run `python evaluate_retrieval.py` and write `results/retrieval.md`. Expected: lexical hit@5 = 6/6 on special prompts. Record the reworded/Chinese results as they are.
- [ ] **Step 7:** Commit `feat: eval prompt set and retrieval evaluation`. Send `eval_prompts.jsonl` to 鄧存正, plus README text: index rebuild, injection, lexical retrieval, numbers, and A's "Major difficulties" lines.

### Task F1 (later, optional): Vector retriever (Method 1)

**Files:** Create `rag/vector.py`, `tests/test_vector.py`. Modify `rebuild_index.py` (add the `python -m rag.vector build` step).

**Interfaces:** Consumes `Embedder` (B1's `OllamaClient` implements it; built via `make_llm(cfg)`) and `DocStore`. Produces `chunk_text(body, width, stride, max_chunks) -> list[tuple[int, str]]`, `build(cfg, embedder, resume=True) -> None` (writes `output/vectors.f16.npy`, `output/chunks.tsv`, `output/vectors.json` with model, dim, count and the `doc_text.tsv` SHA-256), and `VectorRetriever(cfg)` with `name = "vector"`.

- [ ] Tests (fake `Embedder`, hashed bag of words): `test_chunk_text_overlap` (400 words, 150/120/4 → offsets `[0, 120, 240]`), `test_build_and_search_roundtrip`, `test_one_passage_per_doc`, `test_stale_index_detected` (`RuntimeError` mentioning `rebuild`), `test_resume_after_interrupt`.
- [ ] Implement: chunks of `title + body`, skipping docs with < 20 words; float16 vectors; exact numpy dot-product search; a checkpoint every 2,000 chunks. Run the build, then record time/size. Re-run `evaluate_retrieval.py`, which picks up `vector` automatically. Commit `feat: vector retriever (Method 1)`.

### Task F2 (later, optional): Hybrid retriever (Method 3)

**Files:** Create `rag/hybrid.py`, `tests/test_hybrid.py`.

**Interfaces:** Produces `rrf_fuse(rankings: list[list[Passage]], rrf_k: int, top_n: int) -> list[Passage]` and `HybridRetriever(cfg)` with `name = "hybrid"`, which builds its members via `make_retriever`.

- [ ] Tests: `test_rrf_scores` (first in both of 2 lists → `2/61`), `test_rrf_keeps_single_list_doc`, `test_rrf_passage_from_better_rank` (lexical wins ties), `test_hybrid_special_token_rank1` (markers `index`, `llm`).
- [ ] Implement RRF over the top `retrieve_k` of each member; ties go to the smaller doc_id. If the special-token test fails, report it and don't tune around it, because it is a "Major difficulties" finding. Re-run `evaluate_retrieval.py`. Commit `feat: hybrid retriever (Method 3)`.

---

## Track B: generation + interface (鄧存正)

### Task B1: Ollama client

**Files:**
- Create: `rag/llm.py`, `tests/test_llm.py`

**Interfaces:**
- Consumes: `ChatModel`, `Embedder`, `LLMUnavailable` (Task 0).
- Produces: `class OllamaClient(cfg: dict)` implementing `.chat`, `.check`, `.embed` (`embed` is used by F1 later; it is built now so Track A never edits this file).

- [ ] **Step 1: Write the failing tests:**
  - `test_check_reports_not_running`: with host `http://127.0.0.1:9`, `check()` raises `LLMUnavailable` whose message contains `ollama serve`.
  - `test_check_reports_missing_model` (marker `llm`): with model `no-such-model:1b`, the message contains `ollama pull no-such-model:1b`.
  - `test_chat_sends_options` (HTTP monkeypatched): the request body has `stream: false` and `options` with `temperature`, `seed`, `num_ctx` from config.
  - `test_embed_prefixes` (monkeypatched): `vector.query_prefix` / `doc_prefix` are prepended for `kind="query"` / `"document"`.
  - `test_chat_deterministic` (marker `llm`): the same messages twice give the same text.
- [ ] **Step 2:** Run `pytest tests/test_llm.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** `/api/chat` for chat, `/api/embed` in batches of 64, `/api/tags` for `check` (looks up `llm.model`). Timeouts come from `llm.timeout_s`.
- [ ] **Step 4:** Run the tests. Expected: PASS. **Step 5:** Commit `feat: Ollama client`.

### Task B2: Prompt formulation and pipeline

**Files:**
- Create: `rag/prompting.py`, `rag/pipeline.py`, `tests/test_prompting.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `Passage`, `Retriever`, `ChatModel`, `make_retriever`, `make_llm`.
- Produces: `build_messages(prompt: str, passages: list[Passage], char_budget: int) -> list[dict]`, `plain_messages(prompt: str) -> list[dict]`, `format_sources(passages: list[Passage]) -> str`.
- Produces: `@dataclass RagAnswer(prompt: str, mode: str, passages: list[Passage], messages: list[dict], response: str, seconds: float, note: str)` and `class RagPipeline(cfg: dict, retriever: Retriever | None = None, llm: ChatModel | None = None)`. Its methods are `.answer(prompt, use_rag: bool) -> RagAnswer` and `.compare(prompt) -> tuple[RagAnswer, RagAnswer]` (without, with). `mode` is `"none"` or the retriever's `name`.

System prompts (exact text):

- RAG: `You are a question-answering assistant. Answer the user's question using only the numbered documents below, which come from the ClueWeb09 web collection. Cite every fact with its document number in brackets, like [1] or [2][3]. If the documents do not contain the answer, say "The retrieved documents do not contain the answer." Answer in the language of the question.`
- Plain: `You are a question-answering assistant. Answer the user's question. Answer in the language of the question.`
- RAG user message: `Documents:\n[1] <title> (Document ID# <doc_id>)\n<text>\n\n[2] ...\n\nQuestion: <prompt>`

- [ ] **Step 1: Write the failing tests** (`StaticRetriever` + a fake `ChatModel` that records calls):
  - `test_messages_number_passages`: 2 passages become `[1]`, `[2]` with doc ids, and the question comes last.
  - `test_messages_fit_budget`: ten 3,000-char passages with budget 9,000 give a documents part of ≤ 9,000 chars. Passages are dropped from the end, and the last kept one is cut at a word boundary with `…`.
  - `test_format_sources_slide9`: the output starts `Document ID# 42 (clueweb09-en0000-07-00041)` and marks `**115598063**` from `highlights`.
  - `test_pipeline_no_hits_falls_back`: the retriever returns `[]`, so `note == "no relevant documents found; answering without context"` and plain messages are sent.
  - `test_compare_same_model_same_settings`: both calls go to the same `ChatModel`; only the messages differ.
  - `test_special_token_answer_contains_fact` (markers `index`, `llm`): the `use_rag=True` answer to `What is the project codename for 115598063?` contains `AMBER-LYNX-3`; the `use_rag=False` answer does not.
  - `test_unanswerable_says_so` (marker `llm`): a passage about something else, asked `When is the Zorvian Lantern Festival held?`, gives a response containing `do not contain`.
- [ ] **Step 2:** Run `pytest tests/test_prompting.py tests/test_pipeline.py -v`. Expected: FAIL.
- [ ] **Step 3: Implement.** The retriever and LLM come from the registry when not passed in, and are built lazily (no-RAG mode needs no index). `check()` is called once. `LLMUnavailable` propagates.
- [ ] **Step 4:** Run the tests. Expected: PASS. **Step 5:** Commit `feat: prompt formulation, citations, pipeline`.

### Task B3: Command-line interface

**Files:**
- Create: `rag_cli.py`, `tests/test_cli.py`

**Interfaces:** Produces `main(argv: list[str] | None = None) -> int`.

Usage: `python rag_cli.py` (interactive, `prompt> `, `exit` quits), `python rag_cli.py "question"`, `python rag_cli.py -f prompts.txt -o out.md`. Options:
- `--retriever NAME` (default from config; choices = registry names).
- `--no-compare` (by default both sections are printed: `=== Without RAG ===` then `=== With RAG (<retriever>) ===` with sources and answer).
- `-k N`, `--model NAME` (overrides `llm.model`), `--show-prompt`, `--config PATH`.

- [ ] **Step 1: Write the failing tests** (pipeline factory monkeypatched with fakes): `test_compare_output_sections` (both headers and `Document ID#`, in order); `test_ollama_down_exit_code` (exit 2, a one-line stderr message, no `Traceback`); `test_unbuilt_retriever_message` (`--retriever vector` before F1 prints `retriever "vector" is not built yet` and exits 2); `test_batch_markdown` (2 prompts → 2 `## ` sections).
- [ ] **Step 2:** Run the tests. Expected: FAIL. **Step 3:** Implement. Settings and timings go to stderr, results to stdout/`-o`, as in HW2 `search.py`.
- [ ] **Step 4:** Run the tests. Expected: PASS. Manually run `python rag_cli.py` and type `115598063`.
- [ ] **Step 5:** Commit `feat: rag_cli with/without-RAG comparison`.

### Task B4: Snapshots and answer evaluation

**Files:**
- Create: `run_snapshots.py`, `evaluate_answers.py`, `tests/test_evaluate_answers.py`
- Output: `results/snapshots.md`, `results/snapshots.jsonl`, `results/answers.md`, `results/screenshots/*.png`

**Interfaces:** Consumes `prompts/eval_prompts.jsonl` (A3; until then a 3-prompt `prompts/dev_prompts.jsonl` in the same format) and `RagPipeline.compare`.

- [ ] **Step 1: Write the failing test:** `test_answer_correct_counts` (with fake answers: with RAG 2/2, without 0/2, per `type`).
- [ ] **Step 2:** Run it. Expected: FAIL. **Step 3:** Implement both scripts.
  - `snapshots.md` holds, per prompt: the prompt, the answer without RAG, the sources block, the answer with RAG, and timings, plus a header with model, retriever, config hash and date.
  - `answers.md` holds the correct-without vs correct-with counts by `type` and `lang`.
  - Both take `--retriever` and `--model`, so future methods and models are compared by re-running.
- [ ] **Step 4:** Run the test. Expected: PASS.
- [ ] **Step 5:** Run on `eval_prompts.jsonl`. Verify: 18 sections; every special prompt's RAG answer contains its fact; no non-RAG answer does.
- [ ] **Step 6:** Take 3–4 screenshots of `rag_cli.py` comparisons. Commit `docs: snapshots and answer evaluation`.

### Task B5: Packaging and documentation

**Files:**
- Create: `README.md`, `README_onepage.md` → `README.pdf`, `make_zip.py`
- Output: `team27_hw3.zip`

- [ ] **Step 1: One-page doc**, with exactly the slide-10 sections:
  - **Major features:** the RAG methods built, with hit@k; an offline local LLM; slide-9 citations; with/without comparison; swappable retrievers/models; verified injection.
  - **Major difficulties:** B's lines plus A's.
  - **Special execution requirements:** Ollama + pulled models, GPU/VRAM, Python, build times.
  - **Team members list:** what each member actually did.
- [ ] **Step 2: Zip:** code, `rag_config.json`, `requirements.txt`, `special_tokens*.json`, `prompts/`, `results/`, docs, and the index as `postings.delta.xz` plus small tables (HW2 `pack_index.py`). `doc_text.tsv` (and vectors) are rebuilt from `07.warc.gz` by `rebuild_index.py` unless they fit when compressed. `make_zip.py` prints the size and fails above 64 MB (confirmed: a single file of at most 64 MB, same as HW2).
- [ ] **Step 3: Clean-room check.** Unzip into a new folder, make a new venv, run `pip install -r requirements.txt` and `rebuild_index.py` if needed, then `python rag_cli.py 115598063`. Expected: an injected doc comes first and the RAG answer contains `AMBER-LYNX-3`.
- [ ] **Step 4:** Commit `docs: HW3 README, one-page doc, packaging`.

### Task B6 (bonus, optional): Web demo

**Files:** Create `rag_ui.py`.

Follow HW2 `ui.py` (standard-library `http.server`, port 8000). It has a prompt box, retriever and model selectors (from the registry and `/api/tags`), two columns ("Without RAG", "With RAG") from `RagPipeline.compare`, sources with linked URLs/TREC-IDs/highlights, and a collapsible final prompt. Verify by hand with the 6 special prompts, take a screenshot, and commit `feat: web demo`.

---

## Spec coverage

| HW3.pdf requirement | Task |
|---|---|
| Set up an open LLM on the local host (Ollama) | 0, B1 |
| Make the HW1 index accessible to RAG | A1 (rebuild), A2 (Method 2); later F1 (Method 1), F2 (Method 3) |
| Formulate the final prompt with search results; call the LLM; knowledge citation | B2 |
| Input: a prompt + the HW1 inverted index | A2, B3 |
| Prompts about ClueWeb09 content and about special tokens in our part | A1, A3 |
| Show the difference with and without RAG | B2 (`compare`), B3, B4 (B6) |
| Example output `Document ID#: …token…` | A2 (`highlights`), B2 (`format_sources`) |
| Submission: source + library configs, snapshots, one-page doc, team list | B4, B5 |
| Optional features (bonus) | F1, F2, B6, evaluation tables, swappable models |
