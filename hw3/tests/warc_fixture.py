"""Small hand-written WARC/0.18 files for Track A's tests (no real collection needed)."""
import gzip
import random

WORDS = ("river harbor market lamp paper music opera flute garden train station museum festival bridge "
         "valley mountain coffee bakery library theater painter village castle ocean island forest").split()


def page(title: str, body_words: int, seed: int, extra: str = "", charset: str = "utf-8",
         body_tag: bool = True) -> bytes:
    rng = random.Random(seed)
    text = " ".join(rng.choice(WORDS) for _ in range(body_words))
    body_open = '<body class="main">' if body_tag else ""
    html = (f"<html><head><title>{title}</title>"
            f'<meta http-equiv="Content-Type" content="text/html; charset={charset}"></head>'
            f"{body_open}<h1>{title}</h1><p>{extra} {text}</p>{'</body>' if body_tag else ''}</html>")
    return html.encode("utf-8")


def record(rtype: str, headers: dict, content: bytes, eol: bytes = b"\r\n") -> bytes:
    lines = [b"WARC/0.18", f"WARC-Type: {rtype}".encode()]
    lines += [f"{k}: {v}".encode() for k, v in headers.items()]
    lines.append(f"Content-Length: {len(content)}".encode())
    return eol.join(lines) + eol + eol + content + b"\r\n\r\n"


def response(i: int, html: bytes) -> bytes:
    http = b"HTTP/1.1 200 OK\nContent-Type: text/html\n\n" + html
    return record("response", {"WARC-Target-URI": f"http://example{i}.test/page",
                               "WARC-TREC-ID": f"clueweb09-en0000-07-{i:05d}",
                               "Content-Type": "application/http;msgtype=response"}, http)


def write_warc(path, pages) -> None:
    """pages: list of html bytes; doc i+1 gets TREC-ID ...-{i:05d}."""
    data = record("warcinfo", {"WARC-Date": "2009-03-65T08:43:19-0800"}, b"software: test\n")
    data += b"".join(response(i, h) for i, h in enumerate(pages))
    with gzip.open(path, "wb") as f:
        f.write(data)


def default_pages():
    return [page(f"Page {i}", 400 if i != 3 else 50, seed=i) for i in range(5)]
