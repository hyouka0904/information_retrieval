import os
import re
import sys
from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Tuple, Union

# make hw1_indexer.py (same folder as this file) importable from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# NLTK data shipped with the project (stopwords), so the program also runs on a machine without internet
import nltk
# searched before the user's own NLTK folders
nltk.data.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data"))
# HW1 code: tokenize, linguistic and Config
import hw1_indexer as hw1


# raised for malformed queries (unbalanced parentheses, dangling operators, bad proximity operands, ...)
class QuerySyntaxError(ValueError):
    # no behaviour beyond ValueError; the class only gives the error its own type
    pass


# ======================================================================
# AST nodes
# ======================================================================

# one processed index term, e.g. Term('busi')
@dataclass(frozen=True)
class Term:
    # the term exactly as stored in the index
    term: str

    # text form used when the parsed query is printed
    def __str__(self) -> str:
        # print the bare term
        return self.term


# terms at fixed relative positions: ((term, offset), ...), the first offset is 0
@dataclass(frozen=True)
class Phrase:
    # (term, offset from the first term); gaps come from removed stopwords
    terms: Tuple[Tuple[str, int], ...]

    # text form used when the parsed query is printed
    def __str__(self) -> str:
        # words to print, with "_" for every gap position
        parts: List[str] = []
        # offset expected for the next term if there is no gap
        expected = 0
        # walk the terms in order
        for term, off in self.terms:
            # one "_" per skipped position (a removed stopword)
            parts.extend("_" * (off - expected))
            # the term itself
            parts.append(term)
            # next term is expected right after this one
            expected = off + 1
        # quoted, space separated
        return '"' + " ".join(parts) + '"'


# two terms within k positions of each other, in either order
@dataclass(frozen=True)
class Proximity:
    # first operand (a processed term)
    left: str
    # second operand (a processed term)
    right: str
    # maximum allowed |pos_left - pos_right|
    k: int

    # text form used when the parsed query is printed
    def __str__(self) -> str:
        # textbook notation "a /k b"
        return f"({self.left} /{self.k} {self.right})"


# pattern matched against index terms and surface forms; only case folding is applied to it
@dataclass(frozen=True)
class Wildcard:
    # e.g. "comput*" or "wom?n"
    pattern: str

    # text form used when the parsed query is printed
    def __str__(self) -> str:
        # print the pattern as typed (after case folding)
        return self.pattern


# every child must match
@dataclass(frozen=True)
class And:
    # two or more operands
    children: Tuple["Node", ...]

    # text form used when the parsed query is printed
    def __str__(self) -> str:
        # infix with explicit parentheses
        return "(" + " AND ".join(map(str, self.children)) + ")"


# at least one child must match
@dataclass(frozen=True)
class Or:
    # two or more operands
    children: Tuple["Node", ...]

    # text form used when the parsed query is printed
    def __str__(self) -> str:
        # infix with explicit parentheses
        return "(" + " OR ".join(map(str, self.children)) + ")"


# documents that do not match the child
@dataclass(frozen=True)
class Not:
    # the negated operand
    child: "Node"

    # text form used when the parsed query is printed
    def __str__(self) -> str:
        # prefix notation
        return f"NOT {self.child}"


# any AST node
Node = Union[Term, Phrase, Proximity, Wildcard, And, Or, Not]
# nodes that hold index terms directly (no children)
LEAF_TYPES = (Term, Phrase, Proximity, Wildcard)


# result of parse_query(): mode, filter tree, ranking nodes and warnings
@dataclass
class Query:
    # the original query string
    raw: str
    # "free" or "boolean" (see README, Query syntax)
    mode: str
    # Boolean constraint on the result set; None = no constraint (free text without phrase / proximity)
    filter: Optional[Node]
    # nodes whose terms are used for cosine ranking (terms under NOT are excluded later)
    scoring: List[Node]
    # human-readable notes, e.g. operands dropped because they were stopwords
    warnings: List[str] = field(default_factory=list)

    # multi-line summary for debugging and for -v on the CLI
    def __str__(self) -> str:
        # one line per part, for debugging and for echoing the parsed query on the CLI
        lines = [f"mode    : {self.mode}",
                 f"filter  : {self.filter if self.filter is not None else '(none)'}",
                 f"scoring : {', '.join(map(str, self.scoring)) or '(none)'}"]
        # append each warning on its own line
        lines += [f"warning : {w}" for w in self.warnings]
        # join into one block
        return "\n".join(lines)


# ======================================================================
# helpers on ASTs
# ======================================================================

# leaf nodes not inside a NOT (their terms are the ranking terms)
def positive_leaves(node: Optional[Node]) -> Iterator[Node]:
    # nothing to yield for an empty tree
    if node is None:
        # end the generator
        return
    # a leaf is yielded as is
    if isinstance(node, LEAF_TYPES):
        # one leaf
        yield node
    # AND / OR: recurse into every child
    elif isinstance(node, (And, Or)):
        # visit children left to right
        for child in node.children:
            # re-yield whatever the child yields
            yield from positive_leaves(child)
    # NOT: negated terms describe unwanted documents, so they never contribute to the score


# join operands with And / Or, dropping empty ones and flattening nested nodes of the same kind
def _combine(cls, items: List[Optional[Node]]) -> Optional[Node]:
    # drop operands that vanished (stopwords) during normalization
    kept = [x for x in items if x is not None]
    # flattened children: (a AND (b AND c)) -> (a AND b AND c)
    flat: List[Node] = []
    # visit every remaining operand
    for x in kept:
        # same operator one level down: lift its children
        if isinstance(x, cls):
            # merge the nested operands
            flat.extend(x.children)
        # different node type: keep as one operand
        else:
            # one operand
            flat.append(x)
    # every operand vanished
    if not flat:
        # nothing left to match
        return None
    # a single operand needs no operator node
    if len(flat) == 1:
        # the operand itself
        return flat[0]
    # build the n-ary operator node
    return cls(tuple(flat))


# ======================================================================
# lexer
# ======================================================================

# opening / closing quote characters accepted for phrases (ASCII and typographic)
_QUOTES = '"\u201c\u201d'
# characters that end a bare word
_WORD_END = set("()") | set(_QUOTES)
# a proximity operator such as "/3"
_PROX_RE = re.compile(r"/(\d+)")


# split a query string into (kind, value) tokens: LPAREN RPAREN AND OR NOT PROX PHRASE WORD
def _lex(text: str) -> List[Tuple[str, object]]:
    # collected tokens
    tokens: List[Tuple[str, object]] = []
    # current character index
    i = 0
    # length of the query
    n = len(text)
    # scan the whole string
    while i < n:
        # current character
        c = text[i]
        # whitespace only separates tokens
        if c.isspace():
            # skip it
            i += 1
        # opening parenthesis
        elif c == "(":
            # emit LPAREN
            tokens.append(("LPAREN", c))
            # consume it
            i += 1
        # closing parenthesis
        elif c == ")":
            # emit RPAREN
            tokens.append(("RPAREN", c))
            # consume it
            i += 1
        # start of a quoted phrase
        elif c in _QUOTES:
            # index of the closing quote (any accepted quote character)
            j = i + 1
            # advance until a quote or the end of the string
            while j < n and text[j] not in _QUOTES:
                # next character
                j += 1
            # an unclosed quote runs to the end of the query
            tokens.append(("PHRASE", text[i + 1:j]))
            # continue after the closing quote (or past the end)
            i = j + 1
        # anything else starts a bare word
        else:
            # end index of the word
            j = i
            # a word ends at whitespace, a parenthesis or a quote
            while j < n and not text[j].isspace() and text[j] not in _WORD_END:
                # next character
                j += 1
            # the word text
            w = text[i:j]
            # continue after the word
            i = j
            # Boolean operators are recognized only in UPPERCASE
            if w in ("AND", "OR", "NOT"):
                # the operator itself is the token kind
                tokens.append((w, w))
            # proximity operator "/k"
            elif _PROX_RE.fullmatch(w):
                # store k as an int
                tokens.append(("PROX", int(w[1:])))
            # ordinary word (may contain wildcard characters)
            else:
                # emit WORD
                tokens.append(("WORD", w))
    # complete token list
    return tokens


# ======================================================================
# parser
# ======================================================================

# recursive-descent parser over the token list produced by _lex()
class _Parser:

    # token list, cursor, linguistic settings and collected warnings
    def __init__(self, tokens: List[Tuple[str, object]], cfg: "hw1.Config"):
        # tokens to consume
        self.toks = tokens
        # index of the next token
        self.i = 0
        # linguistic settings of the index
        self.cfg = cfg
        # notes collected while parsing
        self.warnings: List[str] = []

    # ---------------- token stream ----------------

    # kind of the next token without consuming it
    def peek(self) -> Optional[str]:
        # kind of the next token, or None at the end
        return self.toks[self.i][0] if self.i < len(self.toks) else None

    # consume and return the next token
    def take(self) -> Tuple[str, object]:
        # current token
        tok = self.toks[self.i]
        # advance past it
        self.i += 1
        # hand it to the caller
        return tok

    # ---------------- normalization ----------------

    # (term, position) pairs of a text, processed exactly like the documents
    def _terms_of(self, text: str) -> List[Tuple[str, int]]:
        # same tokenizer + linguistic module as the documents, so query terms match index terms
        return hw1.linguistic(hw1.tokenize(text, self.cfg), self.cfg)

    # node for a text: None (only stopwords), Term (one term) or Phrase (several terms)
    def _text_node(self, text: str, what: str) -> Optional[Node]:
        # processed (term, position) pairs of the text
        pairs = self._terms_of(text)
        # every token was a stopword (or nothing indexable)
        if not pairs:
            # tell the user why the operand disappeared
            self.warnings.append(f"{what} {text!r} has no indexable term (stopword?) and was ignored")
            # the operand vanishes
            return None
        # exactly one term: a plain term lookup
        if len(pairs) == 1:
            # single-term node
            return Term(pairs[0][0])
        # several terms ("3.5mm", "state-of-the-art", a quoted phrase): positions relative to the first term
        base = pairs[0][1]
        # phrase node keeps the gaps of removed stopwords
        return Phrase(tuple((t, p - base) for t, p in pairs))

    # node for a bare word: Wildcard if it contains * or ?, otherwise like _text_node
    def _word_node(self, word: str) -> Optional[Node]:
        # wildcard characters turn the word into a pattern on index terms
        if "*" in word or "?" in word:
            # the pattern without wildcard characters
            literal = word.replace("*", "").replace("?", "")
            # "*" or "??" alone would match the whole vocabulary
            if not literal:
                # reject instead of scanning every term
                raise QuerySyntaxError(f"wildcard {word!r} needs at least one literal character")
            # only case folding applies: stemming "comput*" would change the prefix itself
            return Wildcard(word.lower() if self.cfg.case_folding else word)
        # ordinary word: tokenize + linguistic
        return self._text_node(word, "word")

    # the single index term of a proximity operand, or a syntax error
    def _single_term(self, tok: Tuple[str, object]) -> str:
        # proximity operands must be plain words
        if tok[0] != "WORD":
            # phrases or operators cannot be proximity operands
            raise QuerySyntaxError(f"proximity operand must be a word, got {tok[1]!r}")
        # normalized node of the word
        node = self._word_node(str(tok[1]))
        # must reduce to exactly one index term
        if not isinstance(node, Term):
            # stopword, wildcard or multi-term word
            raise QuerySyntaxError(f"proximity operand {tok[1]!r} must be a single non-stopword term")
        # the processed term string
        return node.term

    # ---------------- grammar ----------------

    # lowest precedence: OR
    def parse_or(self) -> Optional[Node]:
        # or_expr := and_expr (OR and_expr)*
        items = [self.parse_and()]
        # keep reading "OR and_expr"
        while self.peek() == "OR":
            # consume OR
            self.take()
            # an OR must be followed by an operand
            if self.peek() in (None, "RPAREN", "OR", "AND"):
                # e.g. "linux OR" or "a OR AND b"
                raise QuerySyntaxError("OR is missing its right operand")
            # right operand
            items.append(self.parse_and())
        # build the OR node (flattened, empty operands dropped)
        return _combine(Or, items)

    # AND, explicit or implied by adjacency
    def parse_and(self) -> Optional[Node]:
        # and_expr := not_expr ((AND)? not_expr)*   -- adjacency means AND
        items = [self.parse_not()]
        # continue while another operand (or an explicit AND) follows
        while self.peek() in ("AND", "NOT", "WORD", "PHRASE", "LPAREN"):
            # explicit AND
            if self.peek() == "AND":
                # consume AND
                self.take()
                # an AND must be followed by an operand
                if self.peek() in (None, "RPAREN", "OR", "AND"):
                    # e.g. "linux AND" or "a AND OR b"
                    raise QuerySyntaxError("AND is missing its right operand")
            # right operand
            items.append(self.parse_not())
        # build the AND node (flattened, empty operands dropped)
        return _combine(And, items)

    # NOT
    def parse_not(self) -> Optional[Node]:
        # not_expr := NOT not_expr | prox_expr
        if self.peek() == "NOT":
            # consume NOT
            self.take()
            # NOT needs an operand
            if self.peek() in (None, "RPAREN", "OR", "AND"):
                # e.g. "linux NOT"
                raise QuerySyntaxError("NOT is missing its operand")
            # negated operand
            child = self.parse_not()
            # NOT of a vanished operand vanishes too
            if child is None:
                # nothing to negate
                return None
            # double negation cancels: NOT NOT x = x (so x still counts as a ranking term)
            if isinstance(child, Not):
                # the doubly negated operand
                return child.child
            # ordinary negation
            return Not(child)
        # no NOT: proximity level
        return self.parse_prox()

    # proximity chains: a /k b /m c
    def parse_prox(self) -> Optional[Node]:
        # a proximity chain starts with a plain word
        if self.peek() == "WORD" and self.i + 1 < len(self.toks) and self.toks[self.i + 1][0] == "PROX":
            # first operand as a single term
            left = self._single_term(self.take())
            # pairwise constraints of the chain
            pairs: List[Node] = []
            # "a /3 b /2 c" = (a /3 b) AND (b /2 c)
            while self.peek() == "PROX":
                # distance k
                k = int(self.take()[1])
                # k = 0 would demand the same position
                if k < 1:
                    # reject meaningless distance
                    raise QuerySyntaxError("proximity distance must be at least 1")
                # the operator needs a right operand
                if self.peek() is None:
                    # e.g. "hong /3"
                    raise QuerySyntaxError("proximity operator is missing its right operand")
                # right operand as a single term
                right = self._single_term(self.take())
                # one pairwise constraint
                pairs.append(Proximity(left, right, k))
                # the right operand starts the next pair
                left = right
            # one constraint, or an AND of the chained ones
            return _combine(And, pairs)
        # not a proximity chain: plain atom
        return self.parse_atom()

    # highest precedence: word, phrase or parenthesized expression
    def parse_atom(self) -> Optional[Node]:
        # atom := WORD | PHRASE | ( or_expr )
        kind = self.peek()
        # ran out of tokens where an operand was expected
        if kind is None:
            # e.g. "(" at the very end
            raise QuerySyntaxError("unexpected end of query")
        # parenthesized sub-expression
        if kind == "LPAREN":
            # consume "("
            self.take()
            # "()" is not a valid operand
            if self.peek() == "RPAREN":
                # reject empty group
                raise QuerySyntaxError("empty parentheses")
            # inner expression
            node = self.parse_or()
            # every "(" needs a ")"
            if self.peek() != "RPAREN":
                # e.g. "(a OR b"
                raise QuerySyntaxError("missing ')'")
            # consume ")"
            self.take()
            # the inner expression is the atom
            return node
        # quoted phrase
        if kind == "PHRASE":
            # tokenize + linguistic over the whole phrase, keeping relative positions
            return self._text_node(str(self.take()[1]), "phrase")
        # bare word
        if kind == "WORD":
            # term, multi-term phrase or wildcard
            return self._word_node(str(self.take()[1]))
        # a proximity operator where an operand should be, e.g. "/3 kong"
        if kind == "PROX":
            # reject
            raise QuerySyntaxError("proximity operator is missing its left operand")
        # stray operator or ")" where an operand should be
        raise QuerySyntaxError(f"unexpected {self.toks[self.i][1]!r}")


# ======================================================================
# entry point
# ======================================================================

# parse a query string with the index's linguistic settings (IndexReader.cfg); HW1 defaults if cfg is None
def parse_query(text: str, cfg: Optional["hw1.Config"] = None) -> Query:
    # fall back to HW1's default settings (the configuration of the submitted index)
    cfg = cfg if cfg is not None else hw1.Config()
    # token list
    toks = _lex(text)
    # parser state
    p = _Parser(toks, cfg)
    # any operator or parenthesis switches to Boolean mode
    boolean = any(k in ("AND", "OR", "NOT", "LPAREN", "RPAREN") for k, _ in toks)

    # Boolean mode: the whole query is one expression
    if boolean:
        # parse with full precedence
        tree = p.parse_or()
        # leftover tokens mean something like "a ) b"
        if p.peek() is not None:
            # report the first unexpected token
            raise QuerySyntaxError(f"unexpected {p.toks[p.i][1]!r}")
        # a pure negation has nothing to rank by
        if tree is not None and not any(True for _ in positive_leaves(tree)):
            # still valid: every document not matching is returned with score 0
            p.warnings.append("query has no positive term; results cannot be ranked")
        # the expression is both the filter and the source of ranking terms
        return Query(text, "boolean", tree, [tree] if tree is not None else [], p.warnings)

    # free text: a flat sequence of atoms and proximity chains
    items: List[Node] = []
    # items that every result must match
    required: List[Node] = []
    # read until the token list is exhausted
    while p.peek() is not None:
        # kind of the token that starts this item (decides whether the item is required)
        kind = p.peek()
        # next item (None if it was only stopwords)
        node = p.parse_prox()
        # an item made only of stopwords is skipped
        if node is None:
            # nothing to add
            continue
        # every item contributes ranking terms
        items.append(node)
        # quoted phrases and proximity chains are required; a word that splits into several terms
        # ("non-profit" -> non, profit) is only ranked, so free text stays as loose as the user typed it
        if kind == "PHRASE" or isinstance(node, (Proximity, And)):
            # add to the filter
            required.append(node)
    # AND of the required items, or None when there is none
    flt = _combine(And, required)
    # free-text query object
    return Query(text, "free", flt, items, p.warnings)


# demo: python query_parser.py [query ...]
if __name__ == "__main__":
    # sample queries covering every construct
    samples = [
        "Microsoft",
        "United States government",
        "non-profit organization",
        "open source AND Linux",
        "software engineer OR project manager",
        '"Hong Kong" business',
        '"United States of America"',
        "hong /3 kong",
        "comput* AND NOT (windows OR mac)",
        "the AND linux",
    ]
    # queries given on the command line replace the samples
    queries = sys.argv[1:] or samples
    # parse and print each one
    for q in queries:
        # echo the input
        print(f"> {q}")
        # parse errors are printed instead of crashing the demo
        try:
            # indented parse result
            print("  " + str(parse_query(q)).replace("\n", "\n  "))
        # malformed query
        except QuerySyntaxError as e:
            # show the error message
            print(f"  error   : {e}")
