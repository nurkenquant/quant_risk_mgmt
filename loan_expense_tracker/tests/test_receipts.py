import receipts


def test_parse_text_finds_total_date_and_category():
    text = """SUPER MARKET MAGNUM
    Milk 2 x 450.00       900.00
    Bread                 250.00
    SUBTOTAL            1,150.00
    VAT 12%               138.00
    TOTAL               1,288.00
    25.09.2026 14:32
    """
    out = receipts.parse_text(text)
    assert out["total"] == 1288.00
    assert out["date"] == "2026-09-25"
    assert out["category"] == "Groceries"
    assert out["merchant"].startswith("SUPER MARKET")


def test_parse_text_russian_total_with_space_thousands():
    out = receipts.parse_text("ТОО Кофейня\nЛатте 1 500\nИТОГО: 12 500\n2026-09-01")
    assert out["total"] == 12500
    assert out["date"] == "2026-09-01"


def test_extract_without_backends_returns_none(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    f = tmp_path / "r.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n")
    assert receipts.extract(str(f), "image/png") is None


def test_claude_backend_builds_request_and_parses(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import anthropic

    sent = {}

    class FakeClient:
        def __init__(self):
            self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

        def create(self, **kw):
            sent.update(kw)
            body = {"merchant": "Halyk Bank", "date": "2026-09-10", "total": 192639.95, "currency": "KZT",
                    "kind": "expense", "category": "Loan payment", "summary": "Installment",
                    "loan_name": "Car loan"}
            return SimpleNamespace(stop_reason="end_turn",
                                   content=[SimpleNamespace(type="text", text=json.dumps(body))])

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    f = tmp_path / "r.jpg"
    f.write_bytes(b"\xff\xd8\xff")
    out = receipts.extract(str(f), "image/jpeg", ["Car loan"])
    assert out["loan_name"] == "Car loan" and out["source"] == "claude"
    assert sent["output_config"]["format"]["type"] == "json_schema"
    assert sent["messages"][0]["content"][0]["type"] == "image"
    assert "Car loan" in sent["messages"][0]["content"][1]["text"]
