import io

from pypdf import PdfWriter

from tests.conftest import last_user  # noqa: F401  (kept for symmetry)

DOC = (
    "Our company offers three pricing plans. The Starter plan costs 10 dollars per month.\n\n"
    "The Pro plan costs 50 dollars per month and includes priority support.\n\n"
    "Interviews are held on weekdays between nine and five."
)


def test_upload_txt_then_duplicate(make_env) -> None:  # type: ignore[no-untyped-def]
    env = make_env(lambda s, m: "")
    files = {"file": ("plans.txt", DOC.encode(), "text/plain")}
    r = env.client.post("/api/v1/documents", files=files, data={"strategy": "fixed"})
    assert r.status_code == 201
    body = r.json()
    assert body["chunking_strategy"] == "fixed" and body["chunk_count"] >= 1
    assert body["duplicate"] is False

    again = env.client.post("/api/v1/documents", files=files, data={"strategy": "fixed"})
    assert again.status_code == 200
    assert again.json()["id"] == body["id"] and again.json()["duplicate"] is True

    other = env.client.post("/api/v1/documents", files=files, data={"strategy": "recursive"})
    assert other.status_code == 201  # same file, different strategy => new document

    assert len(env.client.get("/api/v1/documents").json()) == 2


def test_rejects_bad_extension(make_env) -> None:  # type: ignore[no-untyped-def]
    env = make_env(lambda s, m: "")
    r = env.client.post("/api/v1/documents", files={"file": ("a.exe", b"hi", "text/plain")})
    assert r.status_code == 415


def test_rejects_fake_pdf(make_env) -> None:  # type: ignore[no-untyped-def]
    env = make_env(lambda s, m: "")
    r = env.client.post(
        "/api/v1/documents", files={"file": ("a.pdf", b"not a pdf", "application/pdf")}
    )
    assert r.status_code == 415


def test_blank_pdf_is_422(make_env) -> None:  # type: ignore[no-untyped-def]
    env = make_env(lambda s, m: "")
    writer = PdfWriter()
    writer.add_blank_page(200, 200)
    buf = io.BytesIO()
    writer.write(buf)
    r = env.client.post(
        "/api/v1/documents", files={"file": ("scan.pdf", buf.getvalue(), "application/pdf")}
    )
    assert r.status_code == 422


def test_empty_txt_is_422(make_env) -> None:  # type: ignore[no-untyped-def]
    env = make_env(lambda s, m: "")
    r = env.client.post("/api/v1/documents", files={"file": ("e.txt", b"   \n ", "text/plain")})
    assert r.status_code == 422


def test_file_too_large(make_env) -> None:  # type: ignore[no-untyped-def]
    env = make_env(lambda s, m: "")
    big = b"a" * (10 * 1024 * 1024 + 1)
    r = env.client.post("/api/v1/documents", files={"file": ("big.txt", big, "text/plain")})
    assert r.status_code == 413
