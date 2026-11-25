import json
import os

import pytest

import hw1_indexer as hw1
from query_parser import parse_query
from rag.doc_store import window_of
from rag.lexical import LexicalRetriever, sanitize_prompt
from tests.synth import MAGIC, synthetic_index  # noqa: F401  (fixture)

CFG = hw1.Config()


def terms_of(word):
    return frozenset(t for t, _ in hw1.linguistic(hw1.tokenize(word, CFG), CFG))


def _terms(text):
    return frozenset(t for t, _ in hw1.linguistic(hw1.tokenize(text, CFG), CFG))


# ---------------- sanitize_prompt ----------------

def test_sanitize_prompt_question_mark():
    assert sanitize_prompt("When was the premiere of The Magic Flute?") == "When was the premiere of The Magic Flute"


def test_sanitize_prompt_operators():
    r = sanitize_prompt('Is "Linux" AND (GNU) /3 free* OR NOT')
    assert not any(c in r for c in '"()/*?')
    assert not {"AND", "OR", "NOT"} & set(r.split())
    assert parse_query(r, CFG).mode == "free"


def test_sanitize_keeps_words_containing_operators():
    assert sanitize_prompt("ANDROID ORACLE NOTE") == "ANDROID ORACLE NOTE"
    assert sanitize_prompt("“quoted”") == "quoted"


# ---------------- windows ----------------

def test_window_picks_term_cluster():
    body = ("lorem ipsum dolor sit amet " * 6) + "Magic opera tonight. The Magic Flute, a famous opera. " + \
           ("consectetur adipiscing elit sed " * 6)
    text, hl = window_of(body, _terms("magic flute"), 10, terms_of)
    assert "Magic" in text and "Flute" in text
    assert hl == ("Magic", "Flute")
    assert text.startswith("… ") and text.endswith(" …")


def test_window_moves_back_to_sentence_start():
    body = "Filler words here. Team27 registry: student 115598063 maintains the generator. More filler " * 3
    text, hl = window_of(body, _terms("115598063"), 12, terms_of)
    assert "Team27 registry: student 115598063" in text
    assert hl == ("115598063",)


def test_window_no_terms_returns_prefix():
    body = " ".join(f"w{i}" for i in range(50))
    text, hl = window_of(body, frozenset(), 10, terms_of)
    assert text == " ".join(f"w{i}" for i in range(10)) + " …"
    assert hl == ()
    assert window_of("", frozenset({"x"}), 10, terms_of) == ("", ())


# ---------------- retriever on the synthetic rebuilt index ----------------

@pytest.fixture(scope="module")
def synth_retriever(synthetic_index):
    r = LexicalRetriever({"index_dir": synthetic_index["dir"],
                          "lexical": {"scheme": "bm25", "k1": 1.2, "b": 0.75, "passage_words": 150}})
    yield r
    r.close()


def test_synth_stopword_only_prompt_returns_empty(synth_retriever):
    assert synth_retriever.search("what is the", 5) == []
    assert synth_retriever.search("???", 5) == []
    assert synth_retriever.search("qqqxyzzy", 5) == []


def test_synth_special_token_rank1(synth_retriever, synthetic_index):
    for e in synthetic_index["lock"]["entries"]:
        hits = synth_retriever.search(e["token"], 5)
        assert hits[0].doc_id in e["doc_ids"] and hits[1].doc_id in e["doc_ids"]
        assert e["token"].split()[0].lower() in hits[0].text.lower()
        assert hits[0].source == "lexical" and hits[0].trec_id.startswith("clueweb09-")


def test_synth_question_finds_fact(synth_retriever, synthetic_index):
    hits = synth_retriever.search("What is the project codename for 115598063?", 3)
    e = next(x for x in synthetic_index["lock"]["entries"] if x["token"] == "115598063")
    assert hits[0].doc_id in e["doc_ids"]
    assert "AMBER-LYNX-3" in hits[0].text
    assert "115598063" in hits[0].highlights


def test_synth_collection_prompt(synth_retriever):
    hits = synth_retriever.search("When did The Magic Flute premiere?", 3)
    assert hits[0].doc_id == 8          # page index 7 -> doc 8
    assert "Vienna in 1791" in hits[0].text
    assert hits[0].title == "Opera history"
    # "flute" occurs in every synthetic page (idf 0), so HW2 drops it from the ranking terms
    assert set(hits[0].highlights) >= {"Magic", "premiered"}


def test_synth_scores_descending_and_k(synth_retriever):
    hits = synth_retriever.search("Vienna Mozart Keelung harbor registry codename", 4)
    assert len(hits) == 4
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_registry_builds_lexical(synthetic_index):
    from rag.registry import available_retrievers, make_retriever
    r = make_retriever("lexical", {"index_dir": synthetic_index["dir"], "lexical": {}})
    assert r.name == "lexical"
    assert "lexical" in available_retrievers({})


# ---------------- real index (marker index) ----------------

def _lock():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "special_tokens.lock.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def real_retriever(cfg):
    r = LexicalRetriever(cfg)
    yield r
    r.close()


@pytest.mark.index
def test_stopword_only_prompt_returns_empty(real_retriever):
    assert real_retriever.search("what is the", 5) == []


@pytest.mark.index
def test_special_token_rank1(real_retriever):
    for e in _lock()["entries"]:
        hits = real_retriever.search(e["token"], 5)
        assert hits[0].doc_id in e["doc_ids"]
        assert e["token"].lower() in hits[0].text.lower()
