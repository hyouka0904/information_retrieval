"""Minimal web demo for HW2 search.  Run:  python ui.py   then open http://localhost:8000
Standard library only; reuses the same modules as search.py.  Single-threaded on purpose (one shared index handle)."""
import json
import os
import sys
import threading
import time
import traceback
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

# read-only index access
from index_reader import IndexReader
# after index_reader, which points NLTK at nltk_data/
import hw1_indexer as hw1
# query parsing and the AST node types used by the term grid
from query_parser import Phrase, QuerySyntaxError, Term, Wildcard, parse_query, positive_leaves
# ranking
from ranker import Ranker
# same default index folder as the CLI
from search import _DEFAULT_INDEX
# spelling correction
from spell import SpellCorrector
# weighting schemes
from weighting import Scheme, SchemeError

# port from the command line, default 8000
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
# columns in the term/tf grid (wildcards can expand to 30 terms)
MAX_GRID_TERMS = 12

# one index handle shared by every request
idx = IndexReader(_DEFAULT_INDEX)
# scheme name -> Ranker, phonetic flag -> SpellCorrector, created on first use
rankers, spellers = {}, {}


# ---------- titles / snippets from the WARC ----------
# The WARC is one gzip stream (no random access), so it is scanned once in a background thread and every
# document's title + text is cached in output/doc_text.tsv ("title<TAB>text", line n = doc n).
TEXT_PATH = os.path.join(_DEFAULT_INDEX, "doc_text.tsv")
# characters of body text kept per document
TEXT_LIMIT = 20000
# snippet length in words
SNIPPET_WORDS = 30
# places to look for the WARC: next to output/ and one folder above
WARC_PATHS = [os.path.join(os.path.dirname(_DEFAULT_INDEX), "07.warc.gz"),
              os.path.join(os.path.dirname(os.path.dirname(_DEFAULT_INDEX)), "07.warc.gz")]
# byte offset of every line of doc_text.tsv, index 0 = doc 1
text_offsets = None
# shown in the UI while the cache is missing / being built
text_status = ""


# byte offset of every line of doc_text.tsv
def load_text_offsets():
    # offsets so far and the current position
    offs, pos = [], 0
    # binary mode: offsets in bytes
    with open(TEXT_PATH, "rb") as f:
        # one document per line
        for line in f:
            # where this line starts
            offs.append(pos)
            # advance past it
            pos += len(line)
    # one offset per document
    return offs


# background thread: parse the WARC once and write doc_text.tsv
def build_text_cache(warc):
    # results are published through the module globals
    global text_offsets, text_status
    # WARC size for the progress figure, and the temporary output name
    size, tmp = os.path.getsize(warc), TEXT_PATH + ".tmp"
    # progress callback of HW1's parser: update the status line shown in the UI
    progress = lambda b: globals().__setitem__("text_status", f"building title/snippet cache from WARC: {100 * b // size}%")
    # a failure must not kill the server, only disable titles and snippets
    try:
        # write to a temporary name first
        with open(tmp, "wb") as f:
            # every document, numbered like the index
            for n, doc in enumerate(hw1.parse_warc(warc, progress), start=1):
                # doc_id and TREC-ID must agree with the index, or the snippets would belong to other documents
                assert doc.doc_id == n and idx.trec_ids[n] in ("", doc.trec_id), f"WARC does not match the index at doc {n}"
                # one "title<TAB>text" line; carriage returns would break the line structure
                f.write(f"{doc.title}\t{doc.body[:TEXT_LIMIT]}\n".replace("\r", " ").encode("utf-8"))
        # complete: give it its final name
        os.replace(tmp, TEXT_PATH)
        # load the offsets and clear the status line
        text_offsets, text_status = load_text_offsets(), ""
        # report on the console
        print("title/snippet cache ready:", TEXT_PATH)
    # any error while parsing or writing
    except Exception as e:
        # full traceback on the console
        traceback.print_exc()
        # short message in the UI
        text_status = f"building title/snippet cache failed: {e}"


# load doc_text.tsv, or start building it in the background when the WARC is available
def init_text():
    # results are published through the module globals
    global text_offsets, text_status
    # cache already built
    if os.path.exists(TEXT_PATH):
        # just load the offsets
        text_offsets = load_text_offsets()
        # done
        return
    # first WARC location that exists
    warc = next((w for w in WARC_PATHS if os.path.exists(w)), None)
    # no WARC anywhere
    if warc is None:
        # search still works, without titles and snippets
        text_status = "no titles/snippets: 07.warc.gz not found (put it next to team27_hw2/ or inside it)"
        # nothing to build
        return
    # initial status
    text_status = "building title/snippet cache from WARC: 0%"
    # daemon thread, so Ctrl-C does not wait for it
    threading.Thread(target=build_text_cache, args=(warc,), daemon=True).start()


# cached: the same words appear in many snippets
@lru_cache(maxsize=200000)
def word_terms(word):
    """index terms of one display word (same tokenizer + linguistic as the index)."""
    # tokenize and normalize the word exactly like the indexer did
    return frozenset(t for t, _ in hw1.linguistic(hw1.tokenize(word, idx.cfg), idx.cfg))


# [word, is_query_term] pairs for a list of display words
def mark(words, terms):
    # a word is highlighted when any of its index terms is a query term
    return [[w, bool(word_terms(w) & terms)] for w in words]


# title and snippet of one document with the query terms marked
def doc_text(d, terms):
    """(title segments, snippet segments) for doc d; segments are [word, is_query_term]."""
    # cache not ready, or a doc_id beyond it
    if text_offsets is None or d > len(text_offsets):
        # no title, no snippet
        return None, None
    # open the cache for one random read
    with open(TEXT_PATH, "rb") as f:
        # jump to the document's line
        f.seek(text_offsets[d - 1])
        # title before the first tab, body after it
        title, _, body = f.readline().decode("utf-8").rstrip("\n").partition("\t")
    # body as words
    words = body.split()
    # which words are query terms
    hit = [bool(word_terms(w) & terms) for w in words]
    # window of SNIPPET_WORDS words with the most distinct query terms, then the most hits
    best, best_i = (-1, -1), 0
    # every possible window start (at least one, for an empty body)
    for i in range(len(words) or 1):
        # windows worth looking at start on a query term
        if i and not hit[i]:
            # next start
            continue
        # distinct query terms inside the window
        found = set()
        # words of the window with their hit flags
        for w, h in zip(words[i:i + SNIPPET_WORDS], hit[i:i + SNIPPET_WORDS]):
            # query word
            if h:
                # add the query terms it maps to
                found |= word_terms(w) & terms
        # distinct terms first, then total hits
        key = (len(found), sum(hit[i:i + SNIPPET_WORDS]))
        # better window
        if key > best:
            # remember it
            best, best_i = key, i
    # marked words of the best window
    snippet = mark(words[best_i:best_i + SNIPPET_WORDS], terms)
    # text was cut at the start
    if best_i > 0:
        # leading ellipsis
        snippet.insert(0, ["…", False])
    # text was cut at the end
    if best_i + SNIPPET_WORDS < len(words):
        # trailing ellipsis
        snippet.append(["…", False])
    # marked title and snippet
    return mark(title.split(), terms), snippet


# one Ranker per weighting scheme, created on first use
def get_ranker(scheme):
    # not created yet
    if scheme not in rankers:
        # SchemeError propagates to the request handler
        rankers[scheme] = Ranker(idx, scheme=Scheme(idx, scheme))
    # cached ranker
    return rankers[scheme]


# one SpellCorrector per Soundex setting, created on first use
def get_speller(phonetic):
    # not created yet
    if phonetic not in spellers:
        # shares the surface vocabulary loaded by the default ranker's matcher
        spellers[phonetic] = SpellCorrector(idx, get_ranker("ltc.ltc").matcher.surface_vocab, phonetic=phonetic)
    # cached corrector
    return spellers[phonetic]


# run one search request
def search(p):
    """p: dict of form values -> JSON-able result dict."""
    # query text from the form
    text = p.get("q", "").strip()
    # response skeleton
    out = {"query": text, "messages": [], "hits": [], "terms": []}
    # nothing typed
    if not text:
        # explain
        out["messages"].append("empty query")
        # empty result
        return out
    # ranker of the chosen scheme
    ranker = get_ranker(p.get("scheme", "ltc.ltc"))
    # spelling mode: suggest, correct or off
    spell = p.get("spell", "suggest")
    # spelling check enabled
    if spell != "off":
        # corrected query and the list of changes
        fixed, changes = get_speller(p.get("phonetic") == "1").correct_query(text)
        # something looked misspelled
        if changes:
            # correct mode: search the corrected query
            if spell == "correct":
                # show what was changed
                out["messages"].append("spelling corrected: " + ", ".join(f"{a} → {b}" for a, b in changes))
                # use the corrected text
                text = fixed
            # suggest mode: only offer it
            else:
                # clickable suggestion in the page
                out["suggest"] = fixed
    # syntax errors are reported, not raised
    try:
        # parse with the index configuration
        parsed = parse_query(text, idx.cfg)
    # malformed query
    except QuerySyntaxError as e:
        # error message for the page
        out["error"] = f"syntax error: {e}"
        # nothing to rank
        return out
    # parse result, shown in the page
    out["parsed"] = str(parsed)
    # parser warnings (stopwords, ignored words, ...)
    out["messages"] += parsed.warnings
    # exact or one of the inexact top-k methods
    method = p.get("method", "exact")
    # cluster pruning parameters (None = off)
    cluster = None
    # cluster pruning chosen
    if method == "cluster":
        # "B1,B2" from the form
        b1, b2 = (int(x) for x in p.get("cluster", "1,4").split(","))
        # as a tuple for the ranker
        cluster = (b1, b2)
    # start timer
    t0 = time.time()
    # rank with the chosen method
    res = ranker.rank(parsed, top_k=int(p.get("k", 10)), champions=method == "champions",
                      tiered=method == "tiered", impact=method == "impact", cluster=cluster)
    # ranking time
    out["ms"] = round((time.time() - t0) * 1000, 1)
    # number of eligible documents
    out["candidates"] = res.num_candidates
    # query terms not used for scoring
    out["dropped"] = res.terms_dropped
    # remarks of the ranker
    out["messages"] += res.notes
    # the pruned search was widened
    if res.champions_fallback:
        # tell the user
        out["messages"].append("pruned candidate set too small; search was widened to exact")
    # term grid: one column per query word (a wildcard is one column = sum of its expansions' tf)
    cols = {}
    # every scoring subtree of the query
    for root in parsed.scoring:
        # its leaves that are not under NOT
        for leaf in positive_leaves(root):
            # wildcard: one column for all its expansions used in scoring
            if isinstance(leaf, Wildcard):
                # expansions that were actually scored
                cols[leaf.pattern] = [t for t in ranker.matcher.expand_wildcard(leaf.pattern) if t in res.terms_used]
            # term, phrase or proximity leaf
            else:
                # its processed terms
                names = [leaf.term] if isinstance(leaf, Term) else \
                    [t for t, _ in leaf.terms] if isinstance(leaf, Phrase) else [leaf.left, leaf.right]
                # one column per scored term
                cols.update((t, [t]) for t in names if t in res.terms_used)
    # cap the number of columns
    cols = list(cols.items())[:MAX_GRID_TERMS]
    # documents in the result list
    hit_ids = {d for d, _ in res.hits}
    # term -> {doc_id: tf} for the result documents only
    tf = {t: {d: f for d, f in idx.get_doc_tfs(t) if d in hit_ids} for _, ts in cols for t in ts}
    # column headers with their (summed) df
    out["terms"] = [{"term": c, "df": sum(idx.df(t) for t in ts)} for c, ts in cols]
    # one row per result: doc_id, score, labels and the tf of every column
    out["hits"] = [{"doc": d, "score": s, "trec": idx.doc_label(d), "url": idx.urls[d],
                    "tf": [sum(tf[t].get(d, 0) for t in ts) for _, ts in cols]} for d, s in res.hits]
    # every scored term, for highlighting
    all_terms = frozenset(res.terms_used)
    # every result row
    for h in out["hits"]:
        # title and snippet (None while the cache is not ready)
        h["title"], h["snippet"] = doc_text(h["doc"], all_terms)
    # cache status, if any
    if text_status:
        # show it under the query
        out["messages"].append(text_status)
    # JSON-able result
    return out


# HTTP handler: "/search" returns JSON, every other path returns the page
class Handler(BaseHTTPRequestHandler):
    # GET requests only
    def do_GET(self):
        # split path and query string
        u = urlparse(self.path)
        # search API
        if u.path == "/search":
            # first value of every form field
            p = {k: v[0] for k, v in parse_qs(u.query).items()}
            # expected errors become a message, unexpected ones a traceback in the page
            try:
                # run the search
                body, ctype = json.dumps(search(p)), "application/json"
            # bad scheme, bad number, missing files
            except (SchemeError, ValueError, FileNotFoundError) as e:
                # short message
                body, ctype = json.dumps({"error": str(e), "hits": [], "terms": [], "messages": []}), "application/json"
            # anything else
            except Exception:
                # traceback on the console
                traceback.print_exc()
                # and in the page, for debugging
                body, ctype = json.dumps({"error": traceback.format_exc(), "hits": [], "terms": [], "messages": []}), "application/json"
        # any other path: the single HTML page
        else:
            # page and its content type
            body, ctype = PAGE, "text/html; charset=utf-8"
        # response body as bytes
        data = body.encode("utf-8")
        # status line
        self.send_response(200)
        # content type header
        self.send_header("Content-Type", ctype)
        # content length header
        self.send_header("Content-Length", str(len(data)))
        # end of headers
        self.end_headers()
        # body
        self.wfile.write(data)


# the single HTML page (CSS and JavaScript inline)
PAGE = r"""<!doctype html><meta charset="utf-8"><title>HW2 Search Demo</title>
<style>
body{font:14px sans-serif;max-width:1100px;margin:16px auto;padding:0 12px}
input[type=text]{width:60%;font-size:16px;padding:4px}
table{border-collapse:collapse;margin-top:8px}td,th{border:1px solid #ccc;padding:2px 6px;text-align:left;vertical-align:top}
.ex{cursor:pointer;color:#03c;text-decoration:underline}
.bar{background:#4a8;height:10px;display:inline-block}
.tf0{background:#eee;color:#aaa;text-align:center}.tf{background:#7c7;text-align:center}
.all{background:#dfd}.part{background:#ffd}.none{background:#fdd}
.msg{color:#a60}.err{color:#c00;white-space:pre-wrap}.url{font-size:12px;color:#080;word-break:break-all}
.snip{font-size:13px;color:#333;margin:2px 0}mark{background:#fe6}
details{margin:8px 0}
</style>
<h3>HW2 Search Demo</h3>
<form id=f>
<input type=text name=q id=q placeholder='e.g. Hong Kong business' autofocus> <button>Search</button><br>
top-k <input name=k value=10 size=3>
scheme <select name=scheme><option>ltc.ltc<option>lnc.ltc<option>anc.atc<option>bm25</select>
method <select name=method><option value=exact>exact<option value=champions>champion lists<option value=impact>impact ordering<option value=tiered>tiered index<option value=cluster>cluster pruning</select>
cluster B1,B2 <input name=cluster value=1,4 size=4>
spell <select name=spell><option>suggest<option>correct<option>off</select>
<label><input type=checkbox name=phonetic value=1>soundex</label>
</form>

<details open><summary>Format guide &amp; examples (click an example to run it)</summary>
<table>
<tr><th>Feature<th>Syntax<th>Examples
<tr><td>Free text (ranked by TF-IDF cosine)<td>words<td><span class=ex>Microsoft</span> · <span class=ex>United States government</span> · <span class=ex>Hong Kong business</span>
<tr><td>Phrase (exact word order)<td>"..."<td><span class=ex>"Hong Kong" business</span> · <span class=ex>"United States of America"</span>
<tr><td>Proximity (within N words, any order)<td>a /N b<td><span class=ex>hong /3 kong</span>
<tr><td>Wildcard<td>* = any chars, ? = one char<td><span class=ex>comput* science</span> · <span class=ex>*ology research</span> · <span class=ex>wom?n rights</span>
<tr><td>Boolean (UPPERCASE; adjacent words = AND)<td>AND OR NOT ( )<td><span class=ex>open source AND Linux</span> · <span class=ex>software engineer OR project manager</span> · <span class=ex>(open OR free) AND software NOT microsoft</span>
<tr><td>Spelling correction (set spell = correct)<td>just type<td><span class=ex>busness in hong kng</span> · <span class=ex>infromation retreival</span>
</table>
Options: <b>scheme</b> = term weighting (SMART ddd.qqq or BM25); <b>method</b> = exact or an inexact top-k shortcut
(first use of impact/tiered/cluster builds tf_index.bin, ~20 s).
</details>

<div id=out></div>
<script>
const f=document.getElementById('f'), out=document.getElementById('out');
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const seg=a=>a.map(([w,hl])=>hl?`<mark>${esc(w)}</mark>`:esc(w)).join(' ');
document.querySelectorAll('.ex').forEach(e=>e.onclick=()=>{q.value=e.textContent;run()});
f.onsubmit=e=>{e.preventDefault();run()};
async function run(){
  out.innerHTML='searching...';
  const r=await (await fetch('/search?'+new URLSearchParams(new FormData(f)))).json();
  let h='';
  if(r.error) h+=`<p class=err>${esc(r.error)}</p>`;
  if(r.suggest) h+=`<p class=msg>Did you mean: <span class=ex onclick="q.value=this.textContent;run()">${esc(r.suggest)}</span></p>`;
  r.messages.forEach(m=>h+=`<p class=msg>${esc(m)}</p>`);
  if(r.parsed!==undefined) h+=`<pre>${esc(r.parsed)}</pre><p>${r.candidates} candidates · ${r.ms} ms`+
    (r.dropped&&r.dropped.length?` · not in index: ${esc(r.dropped.join(', '))}`:'')+`</p>`;
  if(r.hits.length){
    const n=r.terms.length, max=r.hits[0].score;
    const full=r.hits.filter(x=>x.tf.every(v=>v>0)).length;
    h+=`<p><b>${full}/${r.hits.length}</b> results contain all ${n} grid terms. Row color: green = all terms, yellow = some, red = none. Cells = term frequency in the doc.</p>`;
    h+='<table><tr><th>#<th>doc<th>score<th>title / snippet / URL'+r.terms.map(t=>`<th title="df=${t.df}">${esc(t.term)}`).join('');
    r.hits.forEach((x,i)=>{
      const c=x.tf.filter(v=>v>0).length, cls=c==n?'all':c?'part':'none';
      h+=`<tr class=${cls}><td>${i+1}<td>${x.doc}<td>${x.score.toFixed(3)}<br><span class=bar style="width:${Math.round(80*x.score/max)}px"></span>`+
         `<td><a href="${esc(x.url)}" target=_blank>${x.title&&x.title.length?seg(x.title):'(no title)'}</a>`+
         (x.snippet?`<div class=snip>${seg(x.snippet)}</div>`:'')+`<div class=url>${esc(x.url)} · ${esc(x.trec)}</div>`+
         x.tf.map(v=>`<td class=${v?'tf':'tf0'}>${v}`).join('');
    });
    h+='</table>';
  } else if(!r.error) h+='<p>No results.</p>';
  out.innerHTML=h;
}
</script>"""

# run as a script
if __name__ == "__main__":
    # load or start building the title/snippet cache
    init_text()
    # cache is being built or unavailable
    if text_status:
        # explain on the console
        print(text_status, "(one-time, ~4 min; search works meanwhile)")
    # where to open the page
    print(f"HW2 search UI: http://localhost:{PORT}  (Ctrl-C to stop)")
    # serve until Ctrl-C
    try:
        # single-threaded server on localhost only
        HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
    # Ctrl-C
    except KeyboardInterrupt:
        # normal way to stop
        pass
    # always release the index
    finally:
        # close the postings file
        idx.close()
