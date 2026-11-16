"""IR HW1: Positional Inverted Indexer for the ClueWeb09 WARC collection.

Pipeline: WARC parser -> tokenizer -> linguistic module -> streaming SPIMI indexer.

Usage:
    pip install -r requirements.txt
    python hw1_indexer.py -i 07.warc.gz -o output
    python hw1_indexer.py --help        # every optional feature can be switched on / off by a flag

Default settings (running with no flags; these produced the submitted output/ files):
    input             07.warc.gz         --input / -i
    output dir        output/            --output-dir / -o
    include_title     True    index <title> + body, title tokens take the first positions    (--no-title)
    hyphen_mode       split   "state-of-the-art" -> state / of / the / art, one position each (--hyphen-mode)
    pos_base          0       the first token of a document has position 0                     (--pos-base)
    case_folding      True    lower-case every token                                           (--no-case-folding)
    stopword_removal  True    NLTK English list, matched before lemmatization / stemming,
                              removed words leave position gaps                                (--no-stopwords)
    lemmatization     False   WordNet lemma (noun, then verb): "mice" -> "mouse"               (--lemmatize)
    pos_tagging       False   lemmatize with NLTK POS tags instead of noun-then-verb, slower;
                              turns lemmatization on                                           (--pos-tag)
    stemming          True    Porter stem (after lemmatization when both are on)               (--no-stem)
    use_trie          True    vocabulary trie storing each term's byte offset in postings.txt  (--no-trie)
    use_skip_pointers True    skip pointers every floor(sqrt(df)) postings                     (--no-skip-pointers)
    spimi_block_limit 1000000 (term, pos) pairs per SPIMI block before it is written to disk  (--block-limit)
The settings in effect are printed when the program starts; progress bars go to stderr (--no-progress).

Outputs (in --output-dir):
    dictionary.txt    "term, df" per line, sorted by term
    postings.txt      "term, df: <doc, tf: <pos, ...>; ...>", one term per line (Slide 5 format)
"""
import argparse
import codecs
import gzip
import zlib
import heapq
import html
from html.entities import html5 as HTML5_ENTITIES
import math
import os
import re
import shutil
import tempfile
import time
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import lxml.html
from lxml import etree
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import PerceptronTagger

try:
    # installed together with nltk; only used for the progress bars
    from tqdm import tqdm
except ImportError:
    tqdm = None


def nltk_has(resource: str) -> bool:
    """True if NLTK data `resource` (e.g. "corpora/wordnet") is installed, unpacked or as a .zip."""
    for name in (resource, resource + ".zip"):
        try:
            nltk.data.find(name)
            return True
        except LookupError:
            pass
    return False


# NLTK data used by the linguistic module; downloaded only if missing, so later runs work offline
for _resource, _package in (("corpora/stopwords", "stopwords"), ("corpora/wordnet", "wordnet"),
                            ("corpora/omw-1.4", "omw-1.4")):
    if not nltk_has(_resource):
        nltk.download(_package, quiet=True)


@dataclass
class Config:
    """Every switch of the pipeline in one place (Slide 7: optional features can be turned on/off)."""

    # ---------- Parser ----------
    # prepend the <title> text to the body, so title words are indexed and counted in positions
    include_title: bool = True

    # ---------- Tokenizer ----------
    # hyphenated words such as "state-of-the-art":
    #   "split" -> state / of / the / art   (4 tokens, 4 positions)
    #   "join"  -> stateoftheart            (1 token)
    #   "keep"  -> state-of-the-art         (1 token, hyphens kept)
    hyphen_mode: str = "split"
    # position of the first token: 0 (default) or 1 (as in the Slide 5 example)
    pos_base: int = 0

    # ---------- Linguistic module (Slide 7) ----------
    # lower-case every token
    case_folding: bool = True
    # drop NLTK English stopwords (matched on the surface token, before stemming)
    stopword_removal: bool = True
    # WordNet lemma, e.g. "children" -> "child", "mice" -> "mouse"
    lemmatization: bool = False
    # lemmatize with the POS tag of each token (NLTK perceptron tagger, run on the original tokens) instead of
    # trying noun, then verb; only inflected forms (NNS, VBD, VBG, ...) change. Much slower; needs lemmatization
    pos_tagging: bool = False
    # Porter stem, e.g. "running" -> "run"
    stemming: bool = True

    # ---------- Indexer (Slide 7 efficiency features) ----------
    # keep the vocabulary in a trie (exact and prefix lookups)
    use_trie: bool = True
    # attach skip pointers every floor(sqrt(df)) postings
    use_skip_pointers: bool = True
    # SPIMI: (term, pos) pairs collected in memory before a block is sorted and written to disk
    spimi_block_limit: int = 1_000_000

    @classmethod
    def baseline(cls, **overrides) -> "Config":
        """Config with every linguistic step OFF (keeps stopwords such as 'to' and unstemmed words
        such as 'retrieval', as in the Slide 5 example). Keyword arguments override single fields."""
        params = dict(
            case_folding=False,
            stopword_removal=False,
            lemmatization=False,
            stemming=False,
        )
        params.update(overrides)
        return cls(**params)


@dataclass
class WarcRecord:
    """One WARC record: its header fields and its raw content bytes."""
    # WARC header fields, e.g. {"WARC-Type": "response", "WARC-TREC-ID": "...", ...}
    headers: dict
    # raw content; for a "response" record this is the HTTP header + HTML
    content: bytes


def iter_warc_records(path: str, on_progress: Optional[Callable[[int], None]] = None) -> Iterator[WarcRecord]:
    """Stream WARC records one at a time from a .warc or .warc.gz file (gzip is auto-detected).
    on_progress, if given, is called after each record with the byte offset reached in the input file
    (the compressed offset for .gz), so a progress bar can show a percentage of the file size."""
    with open(path, "rb") as raw:
        # gzip files always start with the magic bytes 1f 8b
        is_gzip = raw.read(2) == b"\x1f\x8b"
        raw.seek(0)
        with (gzip.GzipFile(fileobj=raw) if is_gzip else raw) as f:
            try:
                # readline() keeps the newline and returns b"" only at end of file
                line = f.readline()
                while line:
                    # look for the first line of the next record; blank lines between records are skipped
                    if not line.startswith(b"WARC/"):
                        line = f.readline()
                        continue
                    # header lines "Key: Value" until the first blank line (or EOF)
                    headers = {}
                    while True:
                        line = f.readline()
                        if not line or line in (b"\r\n", b"\n"):
                            break
                        k, _, v = line.decode("utf-8", "replace").partition(":")
                        headers[k.strip()] = v.strip()
                    # A missing, non-numeric or negative Content-Length counts as 0 (empty content):
                    # int() would raise ValueError, and f.read(-n) would swallow the rest of the file.
                    # The outer loop then resyncs on the next "WARC/" line.
                    try:
                        length = max(0, int(headers.get("Content-Length", "0") or 0))
                    except ValueError:
                        length = 0
                    content = f.read(length)
                    yield WarcRecord(headers, content)
                    if on_progress is not None:
                        on_progress(raw.tell())
                    line = f.readline()
            except (EOFError, zlib.error):
                # truncated or corrupted gzip stream: every complete record was already yielded, so just stop
                return


# captures the encoding name after "charset=", e.g. "big5" in 'charset="big5"'
_CHARSET_RE = re.compile(rb"""charset\s*=\s*["']?\s*([A-Za-z0-9_\-:.]+)""", re.I)
# declared charset -> codec that decodes a superset of it (fewer garbled characters)
_CHARSET_ALIASES = {
    # Simplified Chinese: gb18030 covers gb2312 and gbk
    "gb2312": "gb18030", "gbk": "gb18030",
    # Traditional Chinese: Microsoft's Big5 superset
    "big5": "cp950",
    # non-standard name for Shift-JIS
    "x-sjis": "shift_jis",
    # browsers decode latin-1 / ascii pages as cp1252, and such pages often contain cp1252 bytes
    "iso-8859-1": "cp1252", "latin1": "cp1252",
    "us-ascii": "cp1252", "ascii": "cp1252",
}


def split_http(content: bytes) -> Tuple[bytes, bytes]:
    """Split an HTTP response into (http_header, payload) at the EARLIEST blank line.

    Both b"\\r\\n\\r\\n" and b"\\n\\n" are searched: ClueWeb09 headers end with "\\n\\n" while the HTML
    often contains "\\r\\n\\r\\n", so searching only one form could cut the page in the middle.
    """
    # (start index, separator) for every blank-line form that occurs
    hits = [(i, sep)
            for sep in (b"\r\n\r\n", b"\n\n")
            if (i := content.find(sep)) != -1]
    # no blank line at all -> no header, everything is payload
    if not hits:
        return b"", content
    i, sep = min(hits)
    return content[:i], content[i + len(sep):]


def _detect_charset(http_header: bytes, html_bytes: bytes) -> str:
    """Return a Python codec name: HTTP header charset, else <meta> charset in the first 4 KB, else utf-8."""
    for blob in (http_header, html_bytes[:4096]):
        m = _CHARSET_RE.search(blob)
        if m:
            cs = m.group(1).decode("ascii", "ignore").lower()
            cs = _CHARSET_ALIASES.get(cs, cs)
            # accept the name only if Python knows the codec (pages contain typos); otherwise keep looking
            try:
                "".encode(cs)
                return cs
            except LookupError:
                pass
    return "utf-8"


# leading <?xml ... ?> declaration (lxml rejects a str input that carries an encoding declaration)
_XML_DECL_RE = re.compile(r"^\s*<\?xml[^>]*\?>", re.I)
# control characters that are never visible text (tab, LF, FF, CR are kept); libxml2 may stop reading at NUL
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0e-\x1f]")
# HTML5 "legacy" entities that browsers decode even without the closing ';' ("Email: &nbsp<input>");
# libxml2 does not, so the name would be indexed as a word ("nbsp"). Not touched when followed by a letter,
# digit, '=' or ';' (e.g. "&copyright", "?a=1&amp=2").
_LEGACY_ENTITY_RE = re.compile(
    r"&(" + "|".join(sorted((n for n in HTML5_ENTITIES if not n.endswith(";")), key=len, reverse=True))
    + r")(?![A-Za-z0-9;=])")
# C1 control characters U+0080..U+009F -> the cp1252 character a browser shows instead (U+0092 -> U+2019 '’').
# They come from raw cp1252 bytes (libxml2 silently drops them: "don\x92t" -> "dont") and from references
# such as "&#146;" (lxml keeps U+0092, which splits "client&#146;s" into "client" + "s"). The 5 code points
# cp1252 leaves undefined stay unchanged (they are not word characters).
_C1_TO_CP1252 = {}
for _b in range(0x80, 0xA0):
    try:
        _C1_TO_CP1252[_b] = bytes([_b]).decode("cp1252")
    except UnicodeDecodeError:
        pass
# Mojibake from broken Word -> HTML converters: only the low byte of U+2018/2019/201C/201D/2013/2014 survives
# as a control character ("don\x19t"). Restored before control characters are dropped; otherwise the
# apostrophe vanishes and "don't" becomes "dont".
_LOW_BYTE_PUNCT = {0x13: "\u2013", 0x14: "\u2014", 0x18: "\u2018", 0x19: "\u2019", 0x1C: "\u201c", 0x1D: "\u201d"}
# patterns used only by the regex fallback
_TAG_RE = re.compile(r"<[^>]+>")
_SCRIPT_RE = re.compile(r"<(script|style|noscript)\b.*?</\1\s*>", re.I | re.S)
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title\s*>", re.I | re.S)
_HEAD_RE = re.compile(r"<head\b.*?</head\s*>", re.I | re.S)
_COMMENT_RE = re.compile(r"<!--.*?-->", re.S)


def _clean_source(html_str: str) -> str:
    """Before parsing: drop the <?xml?> declaration, restore low-byte punctuation mojibake, drop the other control
    characters, map raw C1 characters through cp1252, close legacy entities ("&nbsp" -> "&nbsp;")."""
    html_str = _XML_DECL_RE.sub("", html_str).translate(_LOW_BYTE_PUNCT)
    html_str = _CONTROL_RE.sub("", html_str).translate(_C1_TO_CP1252)
    return _LEGACY_ENTITY_RE.sub(r"&\1;", html_str)


def _regex_text(html_str: str) -> Tuple[str, str]:
    """Fallback for pages lxml cannot handle: strip comments, script/style, <head> and tags with regex."""
    m = _TITLE_RE.search(html_str)
    title = html.unescape(_TAG_RE.sub(" ", m.group(1))) if m else ""
    body = _COMMENT_RE.sub(" ", html_str)
    body = _SCRIPT_RE.sub(" ", body)
    body = _HEAD_RE.sub(" ", body)
    return title, html.unescape(_TAG_RE.sub(" ", body))


def html_to_text(html_str: str) -> Tuple[str, str]:
    """Return (title, body_text) with whitespace collapsed.

    body = visible text of the whole document minus <head> (not only <body>), because some pages close
    </body> early and lxml then places the remaining content outside <body>. Every tag separates words:
    joining text across inline tags would fix "<big>C</big>ORPORATE" but glue far more words that CSS shows
    apart ("Home</a><a>About" -> "HomeAbout").
    """
    html_str = _clean_source(html_str)
    try:
        root = lxml.html.document_fromstring(html_str)
        # drop non-visible nodes; tail text after them is kept by drop_tree()
        for el in root.xpath("//script|//style|//noscript|//comment()"):
            # comments outside <html> have no parent and cannot be dropped
            if el.getparent() is not None:
                el.drop_tree()
        title = root.findtext(".//title") or ""
        # remove <head>, so the title is not repeated in the body
        for head in root.xpath("//head"):
            if head.getparent() is not None:
                head.drop_tree()
        text = " ".join(root.itertext())
        # nothing extracted (e.g. a broken </title> swallowed the page): use the regex fallback,
        # but keep lxml's title if it found one
        if not text.strip():
            title2, text = _regex_text(html_str)
            title = title or title2
    except (etree.ParserError, ValueError):
        # lxml gave up on this document entirely
        title, text = _regex_text(html_str)
    # C1 characters that lxml created from references ("&#146;" -> U+0092); before collapsing whitespace,
    # because U+0085 would otherwise count as whitespace
    title, text = title.translate(_C1_TO_CP1252), text.translate(_C1_TO_CP1252)
    return " ".join(title.split()), " ".join(text.split())


@dataclass
class Document:
    """One web page after parsing."""
    # sequential integer id, starting at 1
    doc_id: int
    # original ClueWeb09 id, e.g. "clueweb09-en0000-07-00000"
    trec_id: str
    # WARC-Target-URI of the page
    url: str
    # text inside <title>
    title: str
    # visible text of the page (everything except <head>)
    body: str


def _cp1252_fallback(err: UnicodeDecodeError) -> Tuple[str, int]:
    """Decode error handler: bytes that are not valid UTF-8 are read as cp1252 (U+FFFD where cp1252 is undefined)."""
    return err.object[err.start:err.end].decode("cp1252", errors="replace"), err.end


# Many pages labelled (or defaulted to) UTF-8 contain cp1252 bytes ("caf\xe9"). Only the invalid bytes are read
# as cp1252, so valid UTF-8 sequences stay correct and "café" is not cut into "caf" + U+FFFD.
codecs.register_error("cp1252_fallback", _cp1252_fallback)


def parse_warc(path: str, on_progress: Optional[Callable[[int], None]] = None) -> Iterator[Document]:
    """Yield one Document per WARC 'response' record, numbered 1, 2, 3, ... in file order.
    on_progress is passed to iter_warc_records (byte offset after each record)."""
    doc_id = 0
    for rec in iter_warc_records(path, on_progress):
        # warcinfo and other record types are not web pages
        if rec.headers.get("WARC-Type") != "response":
            continue

        # every response gets an id, even if no text can be extracted, so ids stay aligned with TREC ids
        doc_id += 1
        http_header, payload = split_http(rec.content)
        charset = _detect_charset(http_header, payload)
        # utf-8: invalid bytes are read as cp1252; other charsets: undecodable bytes become U+FFFD instead of raising
        errors = "cp1252_fallback" if codecs.lookup(charset).name == "utf-8" else "replace"
        html_str = payload.decode(charset, errors=errors)
        # an empty payload gives an empty document
        title, body = html_to_text(html_str) if html_str.strip() else ("", "")
        yield Document(doc_id=doc_id,
                       trec_id=rec.headers.get("WARC-TREC-ID", ""),
                       url=rec.headers.get("WARC-Target-URI", ""),
                       title=title, body=body)


# kana + CJK ideograph ranges (used inside [...])
_CJK = r"\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff"
# one "word character": any Unicode letter or digit, except '_' and CJK (CJK is handled one char at a time)
_W = rf"[^\W_{_CJK}]"
# one letter (a word character that is not a digit)
_L = rf"[^\W\d_{_CJK}]"


def _char_ranges(pred) -> str:
    """Regex character-class body (\\uXXXX-\\uYYYY...) of every BMP character c with pred(c)."""
    out, start = [], None
    for cp in range(0x10001):
        hit = cp < 0x10000 and pred(chr(cp))
        if hit and start is None:
            start = cp
        elif not hit and start is not None:
            out.append(f"\\u{start:04x}" + (f"-\\u{cp - 1:04x}" if cp - 1 > start else ""))
            start = None
    return "".join(out)


# combining marks (Unicode category M*), e.g. Devanagari vowel signs and virama, Arabic / Hebrew points.
# re does not count them as word characters, so without them "हिन्दी" would fall apart into "ह न द".
_MARKS = _char_ranges(lambda c: unicodedata.category(c).startswith("M"))
# a word character or a combining mark (used after the first character of a word)
_WM = rf"(?:{_W}|[{_MARKS}])"
# Alternatives are tried left to right, so ACRONYM and NUMBER must come before WORD.
# re.X (verbose mode) ignores the whitespace and comments inside the pattern.
_TOKEN_RE = re.compile(rf"""
    # 2+ single letters each followed by '.', e.g. U.S.A.
    (?P<ACRONYM>(?:[A-Za-z]\.){{2,}})
  # NUMBER group: either of the two number forms below
  | (?P<NUMBER>
        # thousands-grouped number: 1,000,000 or 1,234.5
        \d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?(?!{_W})
      # dotted number: 3.14, 2.0.1, 192.168.0.1
      | \d+(?:\.\d+)+(?!{_W})
    # end of the NUMBER group
    )
  # a number glued to letters ("3.5mm", "1,000th"): only the number is matched, the letters become the next
  # WORD. The next character must be a letter, so a list such as "8,1995" is never joined into one number.
  | (?P<NUMUNIT>
        \d{{1,3}}(?:,\d{{3}})+(?:\.\d+)?(?={_L})
      | \d+(?:\.\d+)+(?={_L})
    )
  # version / model number: letters, digits, then .digits ("v2.0.1", "PM2.5", "web2.0"). Digits are required
  # before the dot ("No.1" stays No + 1), and nothing may be glued after it ("v2.0.1beta" uses the other rules).
  | (?P<VERSION>{_L}+\d+(?:\.\d+)+(?![.']?{_W}))
  # alphanumeric run (combining marks allowed after the first character), optionally joined by - or '
  | (?P<WORD>{_W}{_WM}*(?:[-']{_WM}+)*)
  # a single CJK / kana character
  | (?P<CJK>[{_CJK}])
""", re.X)
# whole-string check for thousands grouping
_THOUSANDS_RE = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?$")


@dataclass
class Token:
    """A token string and its position (token index, starting at cfg.pos_base)."""
    text: str
    pos: int


# characters a browser does not show: zero-width space, word joiner, BOM, soft hyphen ("Shuff\u200ble" is "Shuffle")
_INVISIBLE_RE = re.compile(r"[\u200b\u2060\ufeff\u00ad]")
# apostrophe look-alikes between two letters ("Let`s", "don\u00b4t", "don\u02bct") -> "'"
_APOS_LIKE_RE = re.compile(r"(?<=[^\W\d_])[\u0060\u00b4\u02bc](?=[^\W\d_])")


def _normalize(text: str) -> str:
    """Unicode clean-up before tokenizing: invisible characters dropped, apostrophe look-alikes -> ', NFKC,
    curly apostrophes -> ', Unicode hyphens -> -."""
    text = _INVISIBLE_RE.sub("", text)
    # before NFKC, which turns U+00B4 into a space + combining accent
    text = _APOS_LIKE_RE.sub("'", text)
    # full-width -> half-width, ligatures split, etc.
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u2010", "-").replace("\u2011", "-")
    return text


def _rule_outputs(kind: str, raw: str, cfg: Config) -> List[str]:
    """Turn one regex match (rule `kind`, matched text `raw`) into 0..n token strings."""
    # "U.S.A." -> "USA"
    if kind == "ACRONYM":
        return [raw.replace(".", "")]
    if kind == "NUMBER":
        # "1,000,000" -> "1000000"
        if _THOUSANDS_RE.match(raw):
            return [raw.replace(",", "")]
        # decimals, versions and IP addresses stay whole
        return [raw]
    # number in front of letters: "1,000" (before "th") -> "1000", "3.5" (before "mm") unchanged
    if kind == "NUMUNIT":
        return [raw.replace(",", "")]
    # version / model numbers stay whole: "v2.0.1", "PM2.5"
    if kind == "VERSION":
        return [raw]
    # one CJK character = one token
    if kind == "CJK":
        return [raw]
    # kind == "WORD" from here on
    # possessive: "John's" -> "John" (other apostrophes such as "don't" are kept)
    if raw.lower().endswith("'s"):
        raw = raw[:-2]
    if "-" in raw:
        # one token per part, each with its own position
        if cfg.hyphen_mode == "split":
            return [p for p in raw.split("-") if p]
        # one token without hyphens
        if cfg.hyphen_mode == "join":
            return [raw.replace("-", "")]
        # "keep" falls through and returns the word unchanged
    return [raw] if raw else []


def tokenize(text: str, cfg: Config = Config()) -> List[Token]:
    """Split `text` into Tokens; every output token takes the next position, starting at cfg.pos_base."""
    tokens = []
    # next position to hand out
    pos = cfg.pos_base
    for m in _TOKEN_RE.finditer(_normalize(text)):
        # m.lastgroup is the name of the matching group: ACRONYM / NUMBER / NUMUNIT / VERSION / WORD / CJK
        for o in _rule_outputs(m.lastgroup, m.group(), cfg):
            tokens.append(Token(o, pos))
            pos += 1
    return tokens


# NLTK English stopword list; frozenset gives O(1) membership tests
STOPWORDS = frozenset(stopwords.words("english"))

# shared NLTK instances
_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()
# words the stemmer / lemmatizer may touch: ASCII letters, apostrophes allowed (numbers, CJK, "mp3" are skipped)
_ASCII_ALPHA_RE = re.compile(r"^[A-Za-z][A-Za-z']*$")


def _restore_case(word: str, out: str) -> str:
    """Copy the casing pattern of `word` (lower / UPPER / Title) onto the lowercase result `out`.
    Only matters when case_folding=False, e.g. "Running" -> "Run", "MICE" -> "MOUSE"."""
    if word.islower():
        return out
    if word.isupper():
        return out.upper()
    if word[0].isupper():
        return out.capitalize()
    return out


@lru_cache(maxsize=500_000)
def _lemmatize(word: str) -> str:
    """WordNet lemma of an English word (noun first, then verb), keeping its casing pattern."""
    if not _ASCII_ALPHA_RE.match(word):
        return word
    low = word.lower()
    lemma = _lemmatizer.lemmatize(low, pos="n")
    if lemma == low:
        lemma = _lemmatizer.lemmatize(low, pos="v")
    return _restore_case(word, lemma)


# Penn Treebank tags of inflected forms -> WordNet POS. Only these are lemmatized: a base-form tag (NN, VB, JJ, ...)
# means the word is already a lemma, and proper nouns, pronouns, numbers, ... have no WordNet lemma
_INFLECTED_TAG_TO_WORDNET = {"NNS": "n", "VBD": "v", "VBG": "v", "VBN": "v", "VBZ": "v",
                             "JJR": "a", "JJS": "a", "RBR": "r", "RBS": "r"}
# NLTK perceptron POS tagger, loaded on first use (only when pos_tagging is on)
_tagger = None


def _pos_tags(words: List[str]) -> List[str]:
    """Penn Treebank tag of every word, tagged as one sequence so each tag depends on its neighbours."""
    global _tagger
    if _tagger is None:
        if not nltk_has("taggers/averaged_perceptron_tagger_eng"):
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        _tagger = PerceptronTagger()
    return [tag for _, tag in _tagger.tag(words)]


@lru_cache(maxsize=500_000)
def _lemmatize_pos(word: str, tag: str) -> str:
    """WordNet lemma of an inflected English word for its Penn Treebank tag, keeping its casing pattern.

    WordNet lists every candidate lemma, irregular forms first (saw -> [saw, see], uses -> [use, us]). The first
    candidate that differs from the word is used, not NLTK's shortest one (feed -> fee, rating -> rat, uses -> us).
    Words with any other tag are returned unchanged."""
    wn_pos = _INFLECTED_TAG_TO_WORDNET.get(tag)
    if wn_pos is None or not _ASCII_ALPHA_RE.match(word):
        return word
    low = word.lower()
    for lemma in _lemmatizer._morphy(low, wn_pos):
        if lemma != low:
            return _restore_case(word, lemma)
    return word


@lru_cache(maxsize=500_000)
def _stem(word: str) -> str:
    """Porter stem of an English word, keeping its casing pattern; numbers, CJK and mixed tokens are unchanged."""
    if not _ASCII_ALPHA_RE.match(word):
        return word
    return _restore_case(word, _stemmer.stem(word.lower()))


def linguistic(tokens: List[Token], cfg: Config = Config()) -> List[Tuple[str, int]]:
    """Apply case folding -> stopword removal -> lemmatization -> stemming (each switchable in cfg).

    Returns [(term, pos), ...]. Positions are never renumbered, so removed tokens leave gaps.
    """
    cur = [(t.text, t.pos) for t in tokens]
    # POS tags come from the original tokens (case and stopwords still present), so the tagger sees the real context
    tag_at = None
    if cfg.lemmatization and cfg.pos_tagging and tokens:
        tag_at = dict(zip((t.pos for t in tokens), _pos_tags([t.text for t in tokens])))
    if cfg.case_folding:
        cur = [(w.lower(), p) for w, p in cur]
    if cfg.stopword_removal:
        # matched case-insensitively on the surface token; positions are kept, so removed words leave gaps
        cur = [(w, p) for w, p in cur if w.lower() not in STOPWORDS]
    # Stopwords are removed only once, on the surface tokens above. A content word whose lemma / stem
    # happens to equal a stopword string ("others" -> "other", "owned" -> "own", "DOS" -> "do") is kept,
    # so it stays searchable (same order as Lucene's EnglishAnalyzer: lowercase -> stop filter -> stemmer).
    if cfg.lemmatization:
        if tag_at is None:
            cur = [(_lemmatize(w), p) for w, p in cur]
        else:
            cur = [(_lemmatize_pos(w, tag_at[p]), p) for w, p in cur]
    if cfg.stemming:
        cur = [(_stem(w), p) for w, p in cur]
    return cur


def process_document(doc: Document, cfg: Config = Config()) -> List[Tuple[str, int]]:
    """Tokenize + linguistic-process one document; returns [(term, pos), ...]."""
    # title words come first, so they take the first positions
    text = f"{doc.title} {doc.body}" if cfg.include_title else doc.body
    return linguistic(tokenize(text, cfg), cfg)


def iter_processed_docs(path: str, cfg: Config = Config(),
                        max_docs: Optional[int] = None,
                        on_progress: Optional[Callable[[int], None]] = None,
                        ) -> Iterator[Tuple[int, str, List[Tuple[str, int]]]]:
    """Yield (doc_id, trec_id, [(term, pos), ...]) for every document in a WARC file.
    max_docs=None processes the whole file; a number stops after that doc_id.
    on_progress is passed to iter_warc_records (byte offset after each record)."""
    for doc in parse_warc(path, on_progress):
        if max_docs is not None and doc.doc_id > max_docs:
            break
        yield doc.doc_id, doc.trec_id, process_document(doc, cfg)


def iter_processed_file(
    path: str,
    cfg: Config,
    max_docs: Optional[int] = None,
    raw_tokens: Optional[bool] = None,
) -> Iterator[Tuple[int, str, List[Tuple[str, int]]]]:
    """Yield (doc_id, trec_id, [(term, pos), ...]) from a `doc_id<TAB>trec_id<TAB>term:pos ...` text file.

    Positions are shifted to cfg.pos_base, and linguistic() is applied if the file holds raw tokens.
    raw_tokens=True / False says so explicitly; None (default) decides by the first document: capitals or a
    stopword share of at least 20% mean raw tokens.
    Processed files are never processed again, so terms are not stemmed twice.
    Lines that are not in the tab-separated format are treated as plain text, one document per line.
    """
    # None = not decided yet (decided by the first document)
    is_raw_tokens: Optional[bool] = raw_tokens
    pos_shift: Optional[int] = None

    with open(path, encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            raw_line = line.rstrip("\r\n")
            parts = raw_line.split("\t", 2)
            # plain-text fallback: the whole line is one document, doc_id = line number
            if len(parts) < 2 or not parts[0].isdigit():
                if max_docs is not None and line_idx > max_docs:
                    break
                doc = Document(doc_id=line_idx, trec_id=f"txt-{line_idx:05d}", url="", title="", body=raw_line)
                yield line_idx, doc.trec_id, process_document(doc, cfg)
                continue

            doc_id = int(parts[0])
            if max_docs is not None and doc_id > max_docs:
                break
            trec_id = parts[1]
            rest = parts[2] if len(parts) > 2 else ""

            # "term:pos" pairs; rpartition keeps any ':' inside the term
            terms: List[Tuple[str, int]] = []
            valid_pairs = True
            for item in rest.split():
                w, sep, p_str = item.rpartition(":")
                if not sep or not p_str.lstrip("-").isdigit():
                    valid_pairs = False
                    break
                terms.append((w, int(p_str)))

            # third column is free text rather than term:pos pairs -> process it as a document body
            if not valid_pairs:
                doc = Document(doc_id=doc_id, trec_id=trec_id, url="", title="", body=rest)
                yield doc_id, trec_id, process_document(doc, cfg)
                continue

            if terms:
                # shift 0-based files to pos_base=1 (or the reverse), decided once from the first document
                if pos_shift is None:
                    min_p = min(p for _, p in terms)
                    pos_shift = (cfg.pos_base - min_p) if min_p in (0, 1) and min_p != cfg.pos_base else 0
                if pos_shift:
                    terms = [(w, p + pos_shift) for w, p in terms]

                if is_raw_tokens is None:
                    # raw tokenizer output has capitals or a high stopword share (~40% of English text);
                    # processed output may still hold a few stems equal to a stopword ("others" -> "other"),
                    # so a single stopword is not proof of raw input
                    n_sw = sum(w.lower() in STOPWORDS for w, _ in terms)
                    is_raw_tokens = any(w != w.lower() for w, _ in terms) or n_sw >= 0.2 * len(terms)
                if is_raw_tokens:
                    tokens = [Token(w, p) for w, p in terms]
                    terms = linguistic(tokens, cfg)

            yield doc_id, trec_id, terms


# ==============================================================================
# Data structures: postings, vocabulary trie, skip pointers, postings line format
# ==============================================================================
@dataclass
class PostingEntry:
    """One document's entry in a term's postings list: doc_id, term frequency (tf), positions, skip pointer."""
    doc_id: int
    tf: int
    positions: List[int]
    # skip pointer (index of the target entry in the postings list, its doc_id); None = no skip here
    skip: Optional[Tuple[int, int]] = None


class VocabularyTrieNode:
    """Trie node; a node that ends a term stores the term's df and `meta` (the byte offset of its postings)."""
    __slots__ = ("children", "is_end", "df", "meta")

    def __init__(self):
        self.children: Dict[str, "VocabularyTrieNode"] = {}
        self.is_end: bool = False
        self.df: int = 0
        # int byte offset of the term's line in postings.txt
        self.meta: Any = None


class VocabularyTrie:
    """Vocabulary stored as a trie: exact lookup and prefix lookup (Slide 7 efficiency feature)."""
    def __init__(self):
        self.root = VocabularyTrieNode()
        self.size = 0

    def insert(self, term: str, df: int, meta: Any = None) -> None:
        """Add `term` (or overwrite its df / meta if it already exists)."""
        node = self.root
        for ch in term:
            if ch not in node.children:
                node.children[ch] = VocabularyTrieNode()
            node = node.children[ch]
        if not node.is_end:
            self.size += 1
        node.is_end = True
        node.df = df
        node.meta = meta

    def search(self, term: str) -> Optional[Tuple[int, Any]]:
        """Return (df, meta) if `term` is in the vocabulary, else None (a prefix of a term is not a term)."""
        node = self.root
        for ch in term:
            if ch not in node.children:
                return None
            node = node.children[ch]
        return (node.df, node.meta) if node.is_end else None

    def starts_with(self, prefix: str) -> List[Tuple[str, int, Any]]:
        """Return [(term, df, meta), ...] for every term starting with `prefix`, in lexicographical order."""
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return []
            node = node.children[ch]
        results: List[Tuple[str, int, Any]] = []
        # iterative pre-order DFS with an explicit stack (no RecursionError on very long terms);
        # children are pushed in reverse order so they are popped in lexicographical order
        stack: List[Tuple[VocabularyTrieNode, str]] = [(node, prefix)]
        while stack:
            cur, word = stack.pop()
            if cur.is_end:
                results.append((word, cur.df, cur.meta))
            for ch in sorted(cur.children, reverse=True):
                stack.append((cur.children[ch], word + ch))
        return results


def compute_skip_pointers(postings: List[PostingEntry]) -> List[PostingEntry]:
    """Attach skip pointers every floor(sqrt(L)) entries, L = len(postings); none when L < 4 (Slide 7).
    Entry i (i a multiple of the step) points to entry i + step. Existing pointers are reset."""
    L = len(postings)
    if L < 4:
        for p in postings:
            p.skip = None
        return postings
    step = math.isqrt(L)
    for i in range(L):
        if i % step == 0 and i + step < L:
            postings[i].skip = (i + step, postings[i + step].doc_id)
        else:
            postings[i].skip = None
    return postings


def format_posting_standard(term: str, postings: List[PostingEntry], multiline: bool = False) -> str:
    """Format one postings record as in Slide 5 (textbook Fig. 2.11, Sec. 2.4.2):
        term, df: <doc1, tf1: <pos1, pos2, ...>; doc2, tf2: <pos1, pos2, ...>>
    multiline=False (default) writes the whole record on one line, so a file can be read line by line and
    a byte offset identifies a term; multiline=True breaks lines after "df:" and after each ";" as on the slide.
    """
    df = len(postings)
    doc_parts = []
    for p in postings:
        pos_str = ", ".join(str(pos) for pos in p.positions)
        doc_parts.append(f"{p.doc_id}, {p.tf}: <{pos_str}>")
    if multiline:
        sep = ";\n"
        return f"{term}, {df}:\n<{sep.join(doc_parts)}>"
    return f"{term}, {df}: <{'; '.join(doc_parts)}>"


def parse_posting_line(line: str, attach_skips: bool = True) -> Tuple[str, int, List[PostingEntry]]:
    """Parse one postings.txt record (single-line or multi-line) back into (term, df, [PostingEntry, ...]).
    Skip pointers are not stored in the file; they are recomputed here when attach_skips=True."""
    raw = (
        line.strip()
        .replace(":\r\n<", ": <")
        .replace(":\n<", ": <")
        .replace(";\r\n", "; ")
        .replace(";\n", "; ")
    )
    head_part, body_part = raw.split(": <", 1)
    term, df_str = head_part.rsplit(", ", 1)
    df = int(df_str)
    # drop the record's closing '>'
    body = body_part[:-1]
    postings: List[PostingEntry] = []
    for chunk in body.split("; "):
        doc_tf_part, pos_part = chunk.split(": <", 1)
        doc_id_str, tf_str = doc_tf_part.split(", ", 1)
        positions = [int(x) for x in pos_part[:-1].split(", ")]
        postings.append(PostingEntry(doc_id=int(doc_id_str), tf=int(tf_str), positions=positions))
    if attach_skips:
        compute_skip_pointers(postings)
    return term, df, postings


# ==============================================================================
# Positional inverted index: streaming SPIMI construction and queries
# ==============================================================================
class PositionalIndex:
    """Positional inverted index, built with streaming SPIMI straight to disk.

    Postings live only in postings.txt; the optional trie holds each term's byte offset into it. Supports
    skip pointers, AND queries, phrase queries and proximity search."""

    def __init__(self, cfg: Config = Config()):
        self.cfg = cfg
        self.trie: Optional[VocabularyTrie] = VocabularyTrie() if cfg.use_trie else None
        self.num_docs: int = 0
        self.total_tokens: int = 0
        self._postings_path: Optional[str] = None

    @staticmethod
    def _read_posting_record_from_handle(fp) -> Optional[str]:
        """Read one postings record (single-line or multi-line, ends with '>') from a binary file handle."""
        lines: List[str] = []
        for raw_line in fp:
            s = raw_line.decode("utf-8").rstrip("\r\n")
            if not lines and not s.strip():
                continue
            lines.append(s)
            if s.endswith(">"):
                return "\n".join(lines)
        return "\n".join(lines) if lines else None

    def get_postings(self, term: str) -> List[PostingEntry]:
        """Return the postings list of `term` ([] if absent).

        With a trie: seek to the stored byte offset and parse one record. Without a trie: scan the sorted
        postings.txt and stop once the current term passes `term` (no vocabulary kept in memory)."""
        if self.trie is not None:
            res = self.trie.search(term)
            if res is None or not self._postings_path:
                return []
            with open(self._postings_path, "rb") as fp:
                fp.seek(res[1])
                rec_str = self._read_posting_record_from_handle(fp)
            if not rec_str:
                return []
            _, _, plist = parse_posting_line(rec_str, attach_skips=self.cfg.use_skip_pointers)
            return plist
        if self._postings_path and os.path.exists(self._postings_path):
            with open(self._postings_path, "rb") as fp:
                while True:
                    rec_str = self._read_posting_record_from_handle(fp)
                    if rec_str is None:
                        break
                    head_line = rec_str.split("\n", 1)[0]
                    head_part = head_line.split(": <", 1)[0] if ": <" in head_line else head_line.rstrip(":")
                    rec_term, _ = head_part.rsplit(", ", 1)
                    if rec_term == term:
                        _, _, plist = parse_posting_line(rec_str, attach_skips=self.cfg.use_skip_pointers)
                        return plist
                    if rec_term > term:
                        break
        return []

    @classmethod
    def _flush_spimi_blocks(
        cls,
        doc_iter: Iterator[Tuple[int, str, List[Tuple[str, int]]]],
        cfg: Config,
        spimi_temp_dir: str,
        idx: "PositionalIndex",
    ) -> List[str]:
        """Read the document stream and write sorted SPIMI block files to spimi_temp_dir; return their paths.

        A block is flushed only after a whole document, once it holds >= cfg.spimi_block_limit (term, pos)
        pairs, so a document never spans two blocks. Block line format: term<TAB>doc:p1,p2 doc:p1,... ."""
        block_files: List[str] = []
        block_index: Dict[str, List[PostingEntry]] = defaultdict(list)
        block_pairs = 0
        last_doc_id: Optional[int] = None

        def flush_block(block_id: int) -> str:
            bpath = os.path.join(spimi_temp_dir, f"block_{block_id:04d}.txt")
            with open(bpath, "w", encoding="utf-8", newline="\n") as bf:
                for term in sorted(block_index.keys()):
                    entries_str = " ".join(
                        f"{p.doc_id}:{','.join(map(str, p.positions))}"
                        for p in block_index[term]
                    )
                    bf.write(f"{term}\t{entries_str}\n")
            block_index.clear()
            return bpath

        for doc_id, _, terms in doc_iter:
            # postings are concatenated in arrival order, so they are sorted only if doc_ids increase
            if last_doc_id is not None and doc_id <= last_doc_id:
                raise ValueError(f"doc_ids must be strictly increasing: got {doc_id} after {last_doc_id}")
            last_doc_id = doc_id
            idx.num_docs += 1
            idx.total_tokens += len(terms)
            if terms:
                # group this document's positions by term -> one PostingEntry per term
                doc_term_pos: Dict[str, List[int]] = defaultdict(list)
                for term, pos in terms:
                    doc_term_pos[term].append(pos)
                for term, pos_list in doc_term_pos.items():
                    block_index[term].append(
                        PostingEntry(doc_id=doc_id, tf=len(pos_list), positions=pos_list)
                    )
                block_pairs += len(terms)

            # flush only between documents, so a document never spans two blocks
            if block_pairs >= cfg.spimi_block_limit:
                block_files.append(flush_block(len(block_files)))
                block_pairs = 0

        if block_index:
            block_files.append(flush_block(len(block_files)))
        return block_files

    @staticmethod
    def _iter_merged_blocks(block_files: List[str]) -> Iterator[Tuple[str, List[PostingEntry]]]:
        """K-way merge of sorted SPIMI block files; yields (term, complete postings list) in term order.

        Only one line per block file is in memory at a time. The heap orders by (term, block index), so a
        term's entries are concatenated in block order; blocks were written in doc_id order, so the
        merged postings list is already sorted by doc_id."""
        handles = [open(bp, "r", encoding="utf-8") for bp in block_files]
        heap: List[Tuple[str, int, List[PostingEntry]]] = []

        def parse_block_line(line: str) -> Tuple[str, List[PostingEntry]]:
            term, rest = line.rstrip("\n").split("\t", 1)
            entries = []
            for item in rest.split(" "):
                if not item:
                    continue
                d_str, p_str = item.split(":", 1)
                pos_list = [int(x) for x in p_str.split(",")]
                entries.append(PostingEntry(doc_id=int(d_str), tf=len(pos_list), positions=pos_list))
            return term, entries

        try:
            for b_idx, fh in enumerate(handles):
                line = fh.readline()
                if line:
                    t, ents = parse_block_line(line)
                    heapq.heappush(heap, (t, b_idx, ents))

            cur_term: Optional[str] = None
            cur_postings: List[PostingEntry] = []

            while heap:
                term, b_idx, ents = heapq.heappop(heap)
                if cur_term is None:
                    cur_term = term
                    cur_postings = list(ents)
                elif term == cur_term:
                    cur_postings.extend(ents)
                else:
                    yield cur_term, cur_postings
                    cur_term = term
                    cur_postings = list(ents)

                next_line = handles[b_idx].readline()
                if next_line:
                    nt, nents = parse_block_line(next_line)
                    heapq.heappush(heap, (nt, b_idx, nents))

            if cur_term is not None:
                yield cur_term, cur_postings
        finally:
            for fh in handles:
                fh.close()

    @classmethod
    def build_and_save_streaming(
        cls,
        doc_iter: Iterator[Tuple[int, str, List[Tuple[str, int]]]],
        dict_path: str,
        postings_path: str,
        cfg: Config = Config(),
        dict_include_df: bool = True,
        multiline_postings: bool = False,
        spimi_temp_dir: Optional[str] = None,
        show_progress: bool = False,
    ) -> Tuple["PositionalIndex", Dict[str, Any]]:
        """Streaming SPIMI: write blocks, then K-way merge them directly into dictionary.txt and postings.txt,
        one term at a time; returns (index, stats).
        show_progress=True shows a progress bar (merged terms) for the merge phase if tqdm is installed.

        The full postings never sit in memory. With cfg.use_trie the trie stores each term's byte offset in
        postings.txt, so get_postings() can seek straight to it. Temporary block files are deleted at the end."""
        idx = cls(cfg)
        cleanup_dir = False
        if spimi_temp_dir is None:
            spimi_temp_dir = tempfile.mkdtemp(prefix="spimi_stream_")
            cleanup_dir = True
        else:
            os.makedirs(spimi_temp_dir, exist_ok=True)

        os.makedirs(os.path.dirname(os.path.abspath(dict_path)) or ".", exist_ok=True)
        os.makedirs(os.path.dirname(os.path.abspath(postings_path)) or ".", exist_ok=True)

        try:
            block_files = cls._flush_spimi_blocks(doc_iter, cfg, spimi_temp_dir, idx)
            vocab_count = 0
            with open(dict_path, "w", encoding="utf-8", newline="\n") as f_dict, \
                 open(postings_path, "wb") as f_post:
                merged = cls._iter_merged_blocks(block_files)
                if show_progress and tqdm is not None:
                    # the vocabulary size is unknown until the merge ends, so the bar counts terms
                    merged = tqdm(merged, desc="2/2 Merge SPIMI blocks", unit=" terms", mininterval=0.5)
                for term, plist in merged:
                    vocab_count += 1
                    df = len(plist)
                    if dict_include_df:
                        f_dict.write(f"{term}, {df}\n")
                    else:
                        f_dict.write(f"{term}\n")

                    # postings.txt is opened in binary mode so tell() is an exact byte offset for the trie
                    byte_offset = f_post.tell()
                    line_bytes = (format_posting_standard(term, plist, multiline=multiline_postings) + "\n").encode("utf-8")
                    f_post.write(line_bytes)

                    if idx.trie is not None:
                        idx.trie.insert(term, df, byte_offset)

            idx._postings_path = postings_path
            stats = {
                "num_docs": idx.num_docs,
                "total_tokens": idx.total_tokens,
                "vocab_size": vocab_count,
                "dict_path": dict_path,
                "postings_path": postings_path,
                "num_blocks": len(block_files),
            }
            return idx, stats
        finally:
            if cleanup_dir and os.path.exists(spimi_temp_dir):
                shutil.rmtree(spimi_temp_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # Query processing: AND with skip pointers, phrase query, proximity search
    # --------------------------------------------------------------------------

    def _resolve_query_postings(self, word: str) -> List[PostingEntry]:
        """Postings of one query word. The word goes through this index's tokenizer and linguistic module, like
        document text and like phrase_query ("Information" -> "inform"). The raw word is never looked up as is:
        it may also exist as a rare term of its own ("Informationes" is stemmed to "information", df 1), which
        would hide the real postings of "inform". A word that is removed (stopword) or splits into several
        tokens has no postings."""
        processed = linguistic(tokenize(word, self.cfg), self.cfg)
        if len(processed) != 1:
            return []
        return self.get_postings(processed[0][0])

    def intersect_with_skips(self, term1: str, term2: str) -> Tuple[List[int], Dict[str, int]]:
        """term1 AND term2 by a linear merge that follows skip pointers when they do not overshoot (Slide 7).
        term1 / term2 are query words, normalized like document text. Returns (doc_ids, {"comparisons": ...,
        "skips_used": ...})."""
        p1 = self._resolve_query_postings(term1)
        p2 = self._resolve_query_postings(term2)
        answer: List[int] = []
        i, j = 0, 0
        comparisons = 0
        skips_used = 0

        while i < len(p1) and j < len(p2):
            comparisons += 1
            # advance the list with the smaller doc_id; follow skips only while they do not pass the other doc_id
            d1, d2 = p1[i].doc_id, p2[j].doc_id
            if d1 == d2:
                answer.append(d1)
                i += 1
                j += 1
            elif d1 < d2:
                if self.cfg.use_skip_pointers and p1[i].skip is not None and p1[i].skip[1] <= d2:
                    while p1[i].skip is not None and p1[i].skip[1] <= d2:
                        i = p1[i].skip[0]
                        skips_used += 1
                        comparisons += 1
                else:
                    i += 1
            else:
                if self.cfg.use_skip_pointers and p2[j].skip is not None and p2[j].skip[1] <= d1:
                    while p2[j].skip is not None and p2[j].skip[1] <= d1:
                        j = p2[j].skip[0]
                        skips_used += 1
                        comparisons += 1
                else:
                    j += 1

        return answer, {"comparisons": comparisons, "skips_used": skips_used}

    def phrase_query(self, query_str: str) -> List[Tuple[int, List[int]]]:
        """Exact phrase query using positions (Slide 6); returns [(doc_id, [start positions]), ...].

        The query goes through the same tokenizer and linguistic module, so its terms keep their relative
        positions, including gaps left by removed stopwords."""
        q_terms = linguistic(tokenize(query_str, self.cfg), self.cfg)
        if not q_terms:
            return []

        first_term, first_pos = q_terms[0]
        # q_terms are already processed, so they are looked up directly
        first_postings = self.get_postings(first_term)
        if not first_postings:
            return []

        # doc_id -> candidate phrase start positions (aligned to query position 0)
        candidates: Dict[int, List[int]] = {
            p.doc_id: [pos - first_pos for pos in p.positions]
            for p in first_postings
        }

        # keep a start s only if every later query term occurs at s + (its query position)
        for term, q_pos in q_terms[1:]:
            t_postings = {
                p.doc_id: set(p.positions)
                for p in self.get_postings(term)
                if p.doc_id in candidates
            }
            next_candidates: Dict[int, List[int]] = {}
            for doc_id, start_positions in candidates.items():
                if doc_id not in t_postings:
                    continue
                doc_pos_set = t_postings[doc_id]
                matched_starts = [s for s in start_positions if (s + q_pos) in doc_pos_set]
                if matched_starts:
                    next_candidates[doc_id] = matched_starts
            candidates = next_candidates
            if not candidates:
                break

        return [
            (doc_id, [s + first_pos for s in sorted(starts)])
            for doc_id, starts in sorted(candidates.items())
        ]

    def proximity_query(self, term1: str, term2: str, max_distance: int) -> List[Tuple[int, List[Tuple[int, int]]]]:
        """Proximity search: [(doc_id, [(pos1, pos2), ...]), ...] for every pair with |pos1 - pos2| <= max_distance.
        term1 / term2 are query words, normalized like document text. Skip pointers are used across documents
        and a two-pointer sliding window inside a document."""
        p1 = self._resolve_query_postings(term1)
        p2 = self._resolve_query_postings(term2)
        results: List[Tuple[int, List[Tuple[int, int]]]] = []
        i, j = 0, 0
        while i < len(p1) and j < len(p2):
            d1, d2 = p1[i].doc_id, p2[j].doc_id
            if d1 == d2:
                matches: List[Tuple[int, int]] = []
                pos_a, pos_b = p1[i].positions, p2[j].positions
                # sliding window over pos_b: lo = first position >= pos1 - max_distance (only moves forward)
                lo, n_b = 0, len(pos_b)
                for pos1 in pos_a:
                    while lo < n_b and pos_b[lo] < pos1 - max_distance:
                        lo += 1
                    k_idx = lo
                    while k_idx < n_b and pos_b[k_idx] <= pos1 + max_distance:
                        matches.append((pos1, pos_b[k_idx]))
                        k_idx += 1
                if matches:
                    results.append((d1, matches))
                i += 1
                j += 1
            elif d1 < d2:
                if self.cfg.use_skip_pointers and p1[i].skip is not None and p1[i].skip[1] <= d2:
                    while p1[i].skip is not None and p1[i].skip[1] <= d2:
                        i = p1[i].skip[0]
                else:
                    i += 1
            else:
                if self.cfg.use_skip_pointers and p2[j].skip is not None and p2[j].skip[1] <= d1:
                    while p2[j].skip is not None and p2[j].skip[1] <= d1:
                        j = p2[j].skip[0]
                else:
                    j += 1
        return results


# ==============================================================================
# Command-line switches (Slide 7: features turned on/off by a parameter)
# ==============================================================================
def build_cli_parser() -> argparse.ArgumentParser:
    """Argument parser for run_cli(); every Config switch has a flag."""
    parser = argparse.ArgumentParser(
        description="IR HW1: Positional Inverted Indexer for WARC / ClueWeb09 collections."
    )
    parser.add_argument("--input", "-i", type=str, default="07.warc.gz",
                        help="Path to input .warc, .warc.gz, pre-tokenized .txt, or plain-text .txt file.")
    parser.add_argument("--output-dir", "-o", type=str, default="output",
                        help="Directory where dictionary.txt and postings.txt will be written.")
    parser.add_argument("--baseline", action="store_true",
                        help="Turn OFF case folding, stopword removal and stemming (--lemmatize / --pos-tag can still be added).")
    parser.add_argument("--no-title", action="store_true", help="Exclude <title> text from document body.")
    parser.add_argument("--no-case-folding", action="store_true", help="Disable case folding.")
    parser.add_argument("--no-stopwords", action="store_true", help="Disable stopword removal.")
    parser.add_argument("--no-stem", action="store_true", help="Disable Porter stemming.")
    parser.add_argument("--lemmatize", action="store_true", help="Enable WordNet lemmatization (before stemming).")
    parser.add_argument("--pos-tag", action="store_true",
                        help="Lemmatize with POS tags from NLTK's perceptron tagger instead of noun-then-verb (slower); "
                             "implies --lemmatize.")
    parser.add_argument("--no-trie", action="store_true", help="Disable VocabularyTrie construction.")
    parser.add_argument("--no-skip-pointers", action="store_true", help="Disable skip pointers on postings lists.")
    parser.add_argument("--hyphen-mode", choices=["split", "join", "keep"], default="split",
                        help="How to handle hyphenated words (default: split).")
    parser.add_argument("--pos-base", type=int, choices=[0, 1], default=0,
                        help="Starting token position offset: 0 (default) or 1 (Slide 5 example).")
    parser.add_argument("--dict-terms-only", action="store_true",
                        help="Write only 'term' per line in dictionary.txt instead of 'term, df'.")
    parser.add_argument("--multiline-postings", action="store_true",
                        help="Write postings in Slide 5 multi-line wrapped format.")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Optional limit on number of documents to index.")
    parser.add_argument("--block-limit", type=int, default=1_000_000,
                        help="Max (term, pos) pairs per SPIMI block before disk flush.")
    parser.add_argument("--no-progress", action="store_true", help="Do not show progress bars.")
    parser.add_argument("--txt-input", choices=["auto", "tokens", "terms"], default="auto",
                        help=".txt input holds raw tokens (run the linguistic module), processed terms (index as "
                             "they are), or auto: decide from the first document.")
    return parser


def print_settings(cfg: Config, args: argparse.Namespace) -> None:
    """Print the input / output paths and every Config value; '*' marks values changed by a flag."""
    default = Config()
    notes = {
        "include_title": "index <title> + body; title tokens take the first positions",
        "hyphen_mode": '"state-of-the-art" -> state / of / the / art',
        "pos_base": "position of the first token in a document",
        "case_folding": "lower-case every token",
        "stopword_removal": "NLTK English list, before lemmatization / stemming; gaps kept",
        "lemmatization": 'WordNet lemma (noun, then verb): "mice" -> "mouse"',
        "pos_tagging": "lemmatize with NLTK POS tags instead of noun-then-verb (slower)",
        "stemming": 'Porter stem (after lemmatization when both are on): "running" -> "run"',
        "use_trie": "vocabulary trie with byte offsets into postings.txt",
        "use_skip_pointers": "skip pointers every floor(sqrt(df)) postings",
        "spimi_block_limit": "(term, pos) pairs per SPIMI block",
    }
    dict_format = "term only" if args.dict_terms_only else '"term, df" per line'
    postings_format = "multi-line (Slide 5 layout)" if args.multiline_postings else "one term per line"
    print("IR HW1 Positional Inverted Indexer")
    print(f"  input      : {args.input}")
    print(f"  output dir : {args.output_dir}")
    print("  settings ('*' = changed by a flag; the defaults produced the submitted output/):")
    for name, note in notes.items():
        value = getattr(cfg, name)
        mark = "*" if value != getattr(default, name) else " "
        print(f"   {mark} {name:<18}{str(value):<9}{note}")
    print(f"   {'*' if args.dict_terms_only else ' '} {'dictionary.txt':<18}{dict_format}")
    print(f"   {'*' if args.multiline_postings else ' '} {'postings.txt':<18}{postings_format}")
    if args.max_docs is not None:
        print(f"   * {'max_docs':<18}{args.max_docs}")
    print(flush=True)


def _warc_stream_with_progress(path: str, cfg: Config, max_docs: Optional[int]):
    """iter_processed_docs(path, cfg, max_docs) with a progress bar over the input file's bytes.
    The bar is updated once per WARC record (one tell() call), so it costs almost nothing."""
    bar = tqdm(total=os.path.getsize(path), desc="1/2 Index documents", unit="B", unit_scale=True,
               unit_divisor=1024, mininterval=0.5)
    n_docs = 0

    def on_progress(offset: int) -> None:
        bar.update(offset - bar.n)

    try:
        for item in iter_processed_docs(path, cfg, max_docs=max_docs, on_progress=on_progress):
            n_docs += 1
            if n_docs % 500 == 0:
                bar.set_postfix_str(f"{n_docs:,} docs", refresh=False)
            yield item
        if max_docs is None:
            # the last record usually ends a few bytes before the end of the gzip stream
            bar.update(bar.total - bar.n)
        bar.set_postfix_str(f"{n_docs:,} docs")
    finally:
        bar.close()


def run_cli(argv: Optional[List[str]] = None) -> Dict[str, Any]:
    """Build an index from the command-line style arguments `argv` and return the build statistics.
    Inputs ending in .txt are read with iter_processed_file, anything else as a WARC file."""
    parser = build_cli_parser()
    args = parser.parse_args(argv)

    if args.baseline:
        cfg = Config.baseline(
            include_title=not args.no_title,
            hyphen_mode=args.hyphen_mode,
            pos_base=args.pos_base,
            lemmatization=args.lemmatize or args.pos_tag,
            pos_tagging=args.pos_tag,
            use_trie=not args.no_trie,
            use_skip_pointers=not args.no_skip_pointers,
            spimi_block_limit=args.block_limit,
        )
    else:
        cfg = Config(
            include_title=not args.no_title,
            hyphen_mode=args.hyphen_mode,
            pos_base=args.pos_base,
            case_folding=not args.no_case_folding,
            stopword_removal=not args.no_stopwords,
            lemmatization=args.lemmatize or args.pos_tag,
            pos_tagging=args.pos_tag,
            stemming=not args.no_stem,
            use_trie=not args.no_trie,
            use_skip_pointers=not args.no_skip_pointers,
            spimi_block_limit=args.block_limit,
        )

    print_settings(cfg, args)
    show_progress = not args.no_progress and tqdm is not None

    if args.input.endswith(".txt"):
        raw_tokens = {"auto": None, "tokens": True, "terms": False}[args.txt_input]
        stream = iter_processed_file(args.input, cfg=cfg, max_docs=args.max_docs, raw_tokens=raw_tokens)
        if show_progress:
            # text input: the total is unknown, so the bar only counts documents
            stream = tqdm(stream, desc="1/2 Index documents", unit=" docs", mininterval=0.5)
    elif show_progress:
        stream = _warc_stream_with_progress(args.input, cfg, args.max_docs)
    else:
        stream = iter_processed_docs(args.input, cfg, max_docs=args.max_docs)

    _, stats = PositionalIndex.build_and_save_streaming(
        stream,
        dict_path=os.path.join(args.output_dir, "dictionary.txt"),
        postings_path=os.path.join(args.output_dir, "postings.txt"),
        cfg=cfg,
        dict_include_df=not args.dict_terms_only,
        multiline_postings=args.multiline_postings,
        show_progress=show_progress,
    )
    return stats


if __name__ == "__main__":
    t0 = time.time()
    stats = run_cli()
    print("Index built:", {k: v for k, v in stats.items() if not k.endswith("_path")})
    print(f"Elapsed: {time.time() - t0:.1f}s")
