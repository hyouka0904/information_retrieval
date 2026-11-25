import bisect
import math
import os
import re
import sys
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

# make hw1_indexer.py (same folder as this file) importable from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# NLTK data shipped with the project (stopwords), so the program also runs on a machine without internet
import nltk
# searched before the user's own NLTK folders
nltk.data.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data"))
# HW1 code: tokenize, linguistic and the stopword list
import hw1_indexer as hw1

# a word the spelling corrector works on: ASCII letters, apostrophes allowed inside (same rule as HW1's stemmer)
ALPHA_WORD_RE = re.compile(r"^[a-z][a-z']*$")
# a run of letters in a raw query string, with the characters around it checked separately
QUERY_WORD_RE = re.compile(r"[A-Za-z][A-Za-z']*")
# one edit must be paid for by a candidate this many times more frequent (noisy-channel weighting)
EDIT_COST = 50
# Soundex digit of every letter (IIR Section 3.4); vowels and h, w, y map to 0
SOUNDEX_DIGIT = {c: d for d, letters in {"0": "aeiouhwy", "1": "bfpv", "2": "cgjkqsxz", "3": "dt",
                                           "4": "l", "5": "mn", "6": "r"}.items() for c in letters}


# translate a wildcard pattern into a compiled regex: * = any run of characters, ? = exactly one
def wildcard_regex(pattern: str) -> "re.Pattern":
    # every other character is literal
    return re.compile("".join(".*" if c == "*" else "." if c == "?" else re.escape(c) for c in pattern))


# Damerau-Levenshtein distance in its optimal-string-alignment form (insert, delete, substitute, swap neighbours)
def osa_distance(a: str, b: str, limit: int = 1 << 30) -> int:
    # identical strings need no edit
    if a == b:
        # zero edits
        return 0
    # lengths of both strings
    la, lb = len(a), len(b)
    # the length difference alone already exceeds the limit
    if abs(la - lb) > limit:
        # any value above the limit tells the caller "too far"
        return limit + 1
    # row i-2 of the dynamic-programming table (needed for transpositions)
    prev2: List[int] = []
    # row i-1, initialised as row 0: turning "" into b[:j] costs j insertions
    prev = list(range(lb + 1))
    # one row per character of a
    for i in range(1, la + 1):
        # row i, column 0: deleting i characters of a
        cur = [i] + [0] * lb
        # smallest value in this row, used for early exit
        row_min = i
        # one column per character of b
        for j in range(1, lb + 1):
            # 0 when the characters match, 1 for a substitution
            cost = 0 if a[i - 1] == b[j - 1] else 1
            # cheapest of deletion, insertion and substitution / match
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            # two swapped neighbouring characters count as one edit
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                # compare with a transposition from two rows back
                v = min(v, prev2[j - 2] + 1)
            # store the cell
            cur[j] = v
            # track the row minimum
            if v < row_min:
                # new minimum
                row_min = v
        # every later cell is at least the row minimum, so stop once it exceeds the limit
        if row_min > limit:
            # report "too far"
            return limit + 1
        # shift the rows
        prev2, prev = prev, cur
    # bottom-right cell
    return prev[lb]


# Soundex code of a word as defined in IIR Section 3.4 (first letter kept, then three digits)
def soundex(word: str) -> str:
    # only letters take part
    letters = [c for c in word.lower() if c in SOUNDEX_DIGIT]
    # a word without letters has no code
    if not letters:
        # empty code never matches anything
        return ""
    # step 1: keep the first letter; step 2: every later letter becomes its digit
    digits = [SOUNDEX_DIGIT[c] for c in letters[1:]]
    # step 3: one digit out of every run of identical consecutive digits
    collapsed = [d for k, d in enumerate(digits) if k == 0 or d != digits[k - 1]]
    # step 4: drop the zeros
    kept = [d for d in collapsed if d != "0"]
    # step 5: pad with trailing zeros and keep the first letter plus three digits
    return (letters[0].upper() + "".join(kept) + "000")[:4]


# padded bigrams of a word ("$" marks both ends, IIR Section 3.3.4)
def bigrams(word: str) -> List[str]:
    # boundary markers make the first and last letter count as much as inner ones
    padded = f"${word}$"
    # every overlapping pair of characters
    return [padded[k:k + 2] for k in range(len(padded) - 1)]


# surface forms of the collection (output/surface_vocab.tsv): wildcard lookup and the spelling dictionary
class SurfaceVocab:

    # read surface_vocab.tsv from the index folder
    def __init__(self, index_dir: str):
        # file written by prep.py / the prep notebook
        path = os.path.join(index_dir, "surface_vocab.tsv")
        # surface forms in code-point order (the file order)
        self.surfaces: List[str] = []
        # index term of every surface form (same position)
        self.terms: List[str] = []
        # collection frequency of every surface form (same position)
        self.cf: List[int] = []
        # LF-only line splitting, matching how the file was written
        with open(path, encoding="utf-8", newline="\n") as f:
            # one "surface<TAB>term<TAB>cf<TAB>df" per line
            for line in f:
                # split into the four fields
                surface, term, cf_s, _ = line.rstrip("\n").split("\t")
                # store the surface form
                self.surfaces.append(surface)
                # store its index term
                self.terms.append(term)
                # store its collection frequency
                self.cf.append(int(cf_s))
        # surface form -> position, for exact lookups
        self.position: Dict[str, int] = {s: k for k, s in enumerate(self.surfaces)}
        # reversed surface forms in sorted order, built on the first suffix pattern
        self._reversed_keys: Optional[List[str]] = None
        # surface position of every entry of _reversed_keys (same order)
        self._reversed_pos: List[int] = []

    # load the vocabulary if the file exists, otherwise return None
    @classmethod
    def load(cls, index_dir: str) -> Optional["SurfaceVocab"]:
        # missing file: wildcard and spelling features fall back or stay off
        if not os.path.exists(os.path.join(index_dir, "surface_vocab.tsv")):
            # nothing to load
            return None
        # read the file
        return cls(index_dir)

    # positions of every surface form whose text starts with `prefix`, from a sorted list
    @staticmethod
    def _prefix_run(sorted_keys: List[str], prefix: str) -> range:
        # first key >= prefix
        lo = bisect.bisect_left(sorted_keys, prefix)
        # first key past every key with this prefix ("\U0010ffff" sorts after any real character)
        hi = bisect.bisect_left(sorted_keys, prefix + "\U0010ffff")
        # contiguous run
        return range(lo, hi)

    # positions of every surface form matching a wildcard pattern (* = any run, ? = one character)
    def match_pattern(self, pattern: str) -> List[int]:
        # literal text before the first wildcard character
        lead = re.split(r"[*?]", pattern, maxsplit=1)[0]
        # literal text after the last wildcard character
        tail = re.split(r"[*?]", pattern)[-1]
        # regex for the final check
        regex = wildcard_regex(pattern)
        # a longer trailing literal narrows the search more, so use the reversed list (IIR's reverse B-tree)
        if len(tail) > len(lead):
            # build the reversed list once
            if self._reversed_keys is None:
                # every surface form spelled backwards, sorted, with its position
                pairs = sorted((s[::-1], k) for k, s in enumerate(self.surfaces))
                # the reversed strings, for bisect
                self._reversed_keys = [r for r, _ in pairs]
                # their positions in the surface list
                self._reversed_pos = [k for _, k in pairs]
            # forms ending with the trailing literal
            pool = [self._reversed_pos[k] for k in self._prefix_run(self._reversed_keys, tail[::-1])]
        # otherwise the leading literal narrows the sorted list (the whole list when it is empty)
        else:
            # forms starting with the leading literal
            pool = list(self._prefix_run(self.surfaces, lead))
        # keep the forms the full pattern matches, in surface order
        return sorted(k for k in pool if regex.fullmatch(self.surfaces[k]))

    # index terms reached through surface forms matching the pattern (sorted, duplicates removed)
    def terms_for_pattern(self, pattern: str) -> List[str]:
        # one term per matching surface form
        return sorted({self.terms[k] for k in self.match_pattern(pattern)})


# spelling correction on query words: edit distance with a bigram index, Soundex as the phonetic option
class SpellCorrector:

    # store the settings; the dictionary itself is built on the first word that needs correcting
    def __init__(self, idx, vocab_loader: Callable[[], Optional[SurfaceVocab]], phonetic: bool = False,
                 min_cf: int = 2, rare_cf: int = 50, rare_ratio: int = 50):
        # index access (df of processed terms, linguistic settings)
        self.idx = idx
        # returns the surface vocabulary (or None when surface_vocab.tsv is missing)
        self._vocab_loader = vocab_loader
        # whether Soundex candidates are added
        self.phonetic = phonetic
        # dictionary words must occur at least this often in the collection
        self.min_cf = min_cf
        # a correctly-indexed spelling seen at most rare_cf times is checked for a much more frequent neighbour
        self.rare_cf = rare_cf
        # ... and replaced only if that neighbour is at least this many times more frequent
        self.rare_ratio = rare_ratio
        # correction dictionary words, None until built
        self.words: Optional[List[str]] = None
        # collection frequency of every dictionary word (same position)
        self.freq: List[int] = []
        # word -> position in the dictionary
        self.word_pos: Dict[str, int] = {}
        # bigram -> word length -> positions of words containing that bigram
        self.gram_index: Dict[str, Dict[int, List[int]]] = {}
        # number of distinct bigrams of every dictionary word
        self.gram_count: List[int] = []
        # Soundex code -> positions, built on first phonetic use
        self._soundex_index: Optional[Dict[str, List[int]]] = None
        # True once the surface vocabulary turned out to be missing
        self.unavailable = False

    # the surface vocabulary, or None (and unavailable set) when surface_vocab.tsv is missing
    def _vocab(self) -> Optional[SurfaceVocab]:
        # the loader caches its result, so calling it again is cheap
        vocab = self._vocab_loader()
        # remember a missing file for the caller's message
        self.unavailable = vocab is None
        # vocabulary or None
        return vocab

    # build the correction dictionary and its bigram index; returns False when there is no vocabulary
    def _ensure_dictionary(self) -> bool:
        # already built
        if self.words is not None:
            # ready
            return True
        # surface forms of the collection
        vocab = self._vocab()
        # surface_vocab.tsv was never built
        if vocab is None:
            # spelling correction is off
            return False
        # dictionary words
        words: List[str] = []
        # their frequencies
        freq: List[int] = []
        # every surface form
        for s, c in zip(vocab.surfaces, vocab.cf):
            # numbers, CJK and mixed tokens are never spelling targets; one-off strings are mostly typos themselves
            if c >= self.min_cf and ALPHA_WORD_RE.match(s):
                # add the word
                words.append(s)
                # and its frequency
                freq.append(c)
        # stopwords are not in the index, but "teh" should still become "the"; give them the top frequency
        top = max(freq) + 1 if freq else 1
        # every stopword not already present
        for w in sorted(hw1.STOPWORDS - set(words)):
            # only plain alphabetic stopwords
            if ALPHA_WORD_RE.match(w):
                # add with the top frequency
                words.append(w)
                # stopwords are the most frequent words of any English text
                freq.append(top)
        # number of distinct bigrams of every dictionary word (same position)
        self.gram_count = []
        # every dictionary word
        for k, w in enumerate(words):
            # its distinct bigrams
            grams = set(bigrams(w))
            # remember how many there are
            self.gram_count.append(len(grams))
            # each distinct bigram once per word
            for g in grams:
                # add the word under its length
                self.gram_index.setdefault(g, {}).setdefault(len(w), []).append(k)
        # word -> position
        self.word_pos = {w: k for k, w in enumerate(words)}
        # frequencies
        self.freq = freq
        # publish the word list last, since it marks the dictionary as built
        self.words = words
        # ready
        return True

    # largest edit distance allowed for a word: 1 for short words, 2 otherwise
    @staticmethod
    def max_distance(word: str) -> int:
        # two edits on a word of four letters or less would turn it into almost anything
        return 1 if len(word) <= 4 else 2

    # dictionary words within the allowed edit distance: [(distance, word position), ...]
    def edit_candidates(self, word: str) -> List[Tuple[int, int]]:
        # allowed distance for this word
        k = self.max_distance(word)
        # length of the word
        la = len(word)
        # distinct bigrams of the word
        grams = set(bigrams(word))
        # word position -> number of shared distinct bigrams
        shared: Dict[int, int] = {}
        # every distinct bigram of the query word
        for g in grams:
            # words containing it, grouped by length
            by_len = self.gram_index.get(g)
            # bigram absent from the dictionary
            if by_len is None:
                # nothing to count
                continue
            # only lengths within k of the word can be within distance k
            for length in range(max(1, la - k), la + k + 1):
                # every word of that length containing the bigram
                for pos in by_len.get(length, ()):
                    # count the shared bigram
                    shared[pos] = shared.get(pos, 0) + 1
        # surviving candidates
        out: List[Tuple[int, int]] = []
        # every word sharing at least one bigram
        for pos, n in shared.items():
            # candidate word
            w = self.words[pos]
            # one edit destroys at most 3 bigram occurrences (a swap), so k edits leave at least
            # (distinct bigrams) - 3k of either word's distinct bigrams in the other one
            if n < max(len(grams), self.gram_count[pos]) - 3 * k:
                # cannot be within distance k
                continue
            # exact distance, stopping early beyond k
            d = osa_distance(word, w, k)
            # close enough
            if d <= k:
                # keep it
                out.append((d, pos))
        # candidates with their distances
        return out

    # dictionary words that sound alike (same Soundex digits), within one more edit than edit_candidates allows
    def phonetic_candidates(self, word: str) -> List[Tuple[int, int]]:
        # build the Soundex index once
        if self._soundex_index is None:
            # digit part of the code -> positions
            self._soundex_index = {}
            # every dictionary word
            for pos, w in enumerate(self.words):
                # file under the digits only, so "kristmas" (K623) meets "christmas" (C623)
                self._soundex_index.setdefault(soundex(w)[1:], []).append(pos)
        # loosest distance accepted for a sound-alike
        k = self.max_distance(word) + 1
        # surviving candidates
        out: List[Tuple[int, int]] = []
        # every word with the same digits
        for pos in self._soundex_index.get(soundex(word)[1:], ()):
            # exact distance, stopping early beyond k
            d = osa_distance(word, self.words[pos], k)
            # close enough
            if d <= k:
                # keep it
                out.append((d, pos))
        # sound-alike candidates
        return out

    # best replacement for a lowercase word, or None when it should be left alone
    def correct_word(self, word: str, context: Sequence[str] = ()) -> Optional[str]:
        # only alphabetic words of three or more letters are corrected
        if len(word) < 3 or not ALPHA_WORD_RE.match(word):
            # leave it
            return None
        # the index terms the word turns into
        terms = [t for t, _ in hw1.linguistic(hw1.tokenize(word, self.idx.cfg), self.idx.cfg)]
        # a stopword is never corrected
        if not terms:
            # leave it
            return None
        # surface vocabulary (needed for the word's own frequency)
        vocab = self._vocab()
        # surface_vocab.tsv was never built
        if vocab is None:
            # spelling correction is off
            return None
        # how often this exact spelling occurs in the collection
        own = vocab.cf[vocab.position[word]] if word in vocab.position else 0
        # every term present in the index
        known = all(self.idx.has_term(t) for t in terms)
        # a spelling that occurs often enough is trusted (web pages also contain typos such as "goverment")
        if known and own > self.rare_cf:
            # leave it
            return None
        # bigram index and Soundex need the dictionary
        if not self._ensure_dictionary():
            # leave it
            return None
        # candidates by edit distance
        cands = self.edit_candidates(word)
        # positions of words that sound alike (empty unless phonetic correction is on)
        sound: Set[int] = set()
        # sound-alikes on request
        if self.phonetic:
            # sound-alike candidates, which may be one edit further away than plain ones
            ph = self.phonetic_candidates(word)
            # remember which words sound alike
            sound = {pos for _, pos in ph}
            # positions already found by edit distance
            seen = {p for _, p in cands}
            # add the new ones
            cands = cands + [(d, pos) for d, pos in ph if pos not in seen]
        # the word itself is not a correction
        cands = [(d, pos) for d, pos in cands if self.words[pos] != word]
        # nothing close enough
        if not cands:
            # leave it
            return None
        # documents shared with the neighbouring query words (IIR 3.3.5), for the most promising candidates
        together: Dict[int, int] = {}
        # neighbours known: "hong kng" should become "hong kong", not the more frequent "hong king"
        if context:
            # documents of every neighbouring term
            neighbour_docs = [{d for d, _ in self.idx.get_doc_tfs(t)} for t in context]
            # the twenty candidates with the best score without context
            top = sorted(cands, key=lambda c: self._score(c[0], c[1], sound, 0))[:20]
            # score each of them
            for _, pos in top:
                # the candidate's index terms
                cterms = [t for t, _ in hw1.linguistic(hw1.tokenize(self.words[pos], self.idx.cfg), self.idx.cfg)]
                # a stopword candidate has no documents of its own
                if not cterms:
                    # no co-occurrence evidence
                    continue
                # documents containing the candidate
                cdocs = {d for d, _ in self.idx.get_doc_tfs(cterms[0])}
                # documents shared with each neighbour, summed
                together[pos] = sum(len(cdocs & nd) for nd in neighbour_docs)
        # lowest score wins; ties broken alphabetically so the result is deterministic
        d, pos = min(cands, key=lambda c: (self._score(c[0], c[1], sound, together.get(c[1], 0)), self.words[c[1]]))
        # an unknown word takes the best candidate
        if not known:
            # replacement
            return self.words[pos]
        # a rare known spelling is replaced only by a much more frequent neighbour
        if self.freq[pos] >= self.rare_ratio * max(own, 1):
            # replacement
            return self.words[pos]
        # keep the rare word
        return None

    # noisy-channel style score: each edit costs a factor EDIT_COST, frequency and shared documents help
    def _score(self, d: int, pos: int, sound: Set[int], together: int) -> float:
        # -log of (EDIT_COST^-d * frequency * (1 + shared documents) * 2 if it sounds alike)
        return (d * math.log(EDIT_COST) - math.log(self.freq[pos]) - math.log(1 + together)
                - (math.log(2) if pos in sound else 0.0))

    # correct every eligible word of a raw query string: (new text, [(old word, new word), ...])
    def correct_query(self, text: str) -> Tuple[str, List[Tuple[str, str]]]:
        # (match, index term usable as context or None) of every run of letters that may be corrected
        runs: List[Tuple["re.Match", Optional[str]]] = []
        # every run of letters
        for m in QUERY_WORD_RE.finditer(text):
            # the run itself
            w = m.group()
            # character before the run ("" at the start)
            before = text[m.start() - 1] if m.start() > 0 else ""
            # character after the run ("" at the end)
            after = text[m.end()] if m.end() < len(text) else ""
            # operators, wildcard pieces, letters glued to digits ("mp3") and the "/" of "/3" are left alone
            if (w in ("AND", "OR", "NOT") or (before and before in "*?/") or (after and after in "*?")
                    or before.isdigit() or after.isdigit()):
                # not a correction target
                continue
            # index terms of the word
            terms = [t for t, _ in hw1.linguistic(hw1.tokenize(w, self.idx.cfg), self.idx.cfg)]
            # a single term that is in the index can serve as context for its neighbours
            ctx = terms[0] if len(terms) == 1 and self.idx.has_term(terms[0]) else None
            # remember the run
            runs.append((m, ctx))
        # pieces of the output string
        parts: List[str] = []
        # corrections made
        changes: List[Tuple[str, str]] = []
        # end of the previous run that was replaced
        last = 0
        # every candidate run, with its neighbours available as context
        for k, (m, _) in enumerate(runs):
            # index terms of the words right before and after it
            context = [runs[j][1] for j in (k - 1, k + 1) if 0 <= j < len(runs) and runs[j][1] is not None]
            # best replacement of the lowercase word
            new = self.correct_word(m.group().lower(), context)
            # nothing to change
            if new is None:
                # keep as typed
                continue
            # text before this word
            parts.append(text[last:m.start()])
            # the replacement
            parts.append(new)
            # record the change
            changes.append((m.group(), new))
            # continue after the word
            last = m.end()
        # rest of the string
        parts.append(text[last:])
        # corrected query and the list of changes
        return "".join(parts), changes
