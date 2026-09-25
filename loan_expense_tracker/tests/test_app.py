import io
import os
from datetime import date

import pytest

import app as app_module
import receipts


PG_URL = os.environ.get("TEST_DATABASE_URL", "")


@pytest.fixture(params=["sqlite", "postgres"])
def client(request, tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    if request.param == "postgres":
        if not PG_URL:
            pytest.skip("set TEST_DATABASE_URL to run against Postgres")
        import psycopg
        with psycopg.connect(PG_URL, autocommit=True) as con:
            con.execute("DROP TABLE IF EXISTS transactions, loans, settings CASCADE")
        database = PG_URL
    else:
        database = str(tmp_path / "t.db")
    a = app_module.create_app({
        "DATABASE": database,
        "UPLOAD_DIR": str(tmp_path / "up"),
        "TESTING": True,
        "APP_PASSWORD": "",
    })
    return a.test_client()


def test_pages_render(client):
    for url in ["/", "/transactions", "/loans", "/loans/new", "/add", "/add?kind=income", "/settings"]:
        assert client.get(url).status_code == 200, url


def test_add_expense_and_income_show_in_totals(client):
    today = date.today().isoformat()
    client.post("/add", data={"kind": "expense", "amount": "12,50", "category": "Dining", "day": today})
    client.post("/add", data={"kind": "income", "amount": "100", "category": "Revenue", "day": today})
    html = client.get("/").get_data(as_text=True)
    assert "+$87.50" in html  # today's net
    assert "Dining" in html


def test_loan_projection_and_mark_paid(client):
    r = client.post("/loans/new", data={
        "name": "Car", "lender": "Halyk", "principal": "12000", "rate": "0",
        "term_months": "12", "first_due": date.today().isoformat()})
    assert r.status_code == 302
    html = client.get("/").get_data(as_text=True)
    assert "Car" in html and "$1,000.00" in html
    client.post("/loans/1/pay/1")
    detail = client.get("/loans/1").get_data(as_text=True)
    assert "1 of 12 payments" in detail
    assert "$11,000.00" in detail  # balance
    # payment became an expense
    assert "Loan payment" in client.get("/transactions").get_data(as_text=True)
    # toggling again un-pays it
    client.post("/loans/1/pay/1")
    assert "0 of 12 payments" in client.get("/loans/1").get_data(as_text=True)


def test_scan_receipt_auto_applies_and_matches_loan(client, monkeypatch):
    client.post("/loans/new", data={
        "name": "Car", "lender": "Halyk Bank", "principal": "12000", "rate": "0",
        "term_months": "12", "first_due": date.today().isoformat()})
    monkeypatch.setattr(receipts, "extract", lambda *a, **k: {
        "merchant": "Halyk Bank", "date": date.today().isoformat(), "total": 1000.0,
        "currency": "", "kind": "expense", "category": "Fees", "summary": "Loan installment",
        "loan_name": "", "source": "claude"})
    r = client.post("/scan", data={"receipt": (io.BytesIO(b"fake"), "r.jpg", "image/jpeg")},
                    content_type="multipart/form-data", follow_redirects=True)
    html = r.get_data(as_text=True)
    assert "marked as payment for" in html
    assert "1 of 12 payments" in client.get("/loans/1").get_data(as_text=True)


def test_scan_receipt_fills_expense(client, monkeypatch):
    monkeypatch.setattr(receipts, "extract", lambda *a, **k: {
        "merchant": "Starbucks", "date": "2026-09-20", "total": 7.35, "currency": "USD",
        "kind": "expense", "category": "Dining", "summary": "Latte", "loan_name": "", "source": "claude"})
    client.post("/scan", data={"receipt": (io.BytesIO(b"fake"), "r.png", "image/png")},
                content_type="multipart/form-data")
    html = client.get("/transactions?month=2026-09").get_data(as_text=True)
    assert "Starbucks" in html and "−$7.35" in html


def test_unreadable_receipt_goes_to_edit(client):
    r = client.post("/scan", data={"receipt": (io.BytesIO(b"x"), "r.png", "image/png")},
                    content_type="multipart/form-data")
    assert r.status_code == 302 and "/tx/" in r.headers["Location"]


def test_rejects_non_receipt_upload(client):
    r = client.post("/scan", data={"receipt": (io.BytesIO(b"x"), "evil.html", "text/html")},
                    content_type="multipart/form-data", follow_redirects=True)
    assert "must be a photo" in r.get_data(as_text=True)


def test_password_protection(tmp_path):
    a = app_module.create_app({"DATABASE": str(tmp_path / "p.db"), "UPLOAD_DIR": str(tmp_path / "u"),
                               "APP_PASSWORD": "s3cret", "TESTING": True})
    c = a.test_client()
    assert c.get("/").status_code == 302
    c.post("/login", data={"password": "s3cret"})
    assert c.get("/").status_code == 200


def test_healthz_is_public(tmp_path):
    a = app_module.create_app({"DATABASE": str(tmp_path / "h.db"), "UPLOAD_DIR": str(tmp_path / "u"),
                               "APP_PASSWORD": "s3cret", "TESTING": True})
    r = a.test_client().get("/healthz")
    assert r.status_code == 200 and r.get_data(as_text=True) == "ok"


def test_receipt_is_served_back_and_deleted(client):
    r = client.post("/scan", data={"receipt": (io.BytesIO(b"%PDF-fake"), "r.pdf", "application/pdf")},
                    content_type="multipart/form-data")
    tx_url = r.headers["Location"]
    html = client.get(tx_url).get_data(as_text=True)
    name = html.split("/receipts/")[1].split('"')[0]
    got = client.get("/receipts/" + name)
    assert got.status_code == 200 and got.data == b"%PDF-fake" and got.mimetype == "application/pdf"
    client.post(tx_url + "/delete")
    assert client.get("/receipts/" + name).status_code == 404


def test_export_csv(client):
    client.post("/add", data={"kind": "expense", "amount": "5", "category": "Dining",
                              "day": "2026-09-01", "merchant": "Cafe"})
    csv = client.get("/export.csv").get_data(as_text=True)
    assert csv.splitlines()[1].startswith("2026-09-01,expense,Dining,5.0,Cafe")


def test_hosted_config_requires_database_and_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    with pytest.raises(RuntimeError, match="DATABASE_URL.*SUPABASE_URL"):
        app_module.create_app({"DATABASE": str(tmp_path / "x.db"), "UPLOAD_DIR": str(tmp_path / "u"),
                               "APP_PASSWORD": "pw"})
