"""SupabaseStorage against a small stand-in for the Storage REST API."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import storage


class FakeSupabase(BaseHTTPRequestHandler):
    objects: dict = {}
    buckets: set = set()
    log: list = []

    def _auth_ok(self):
        return self.headers.get("Authorization") == "Bearer service-key"

    def _reply(self, code, body=b"{}", ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length", 0)))

    def do_POST(self):
        self.log.append(("POST", self.path))
        if not self._auth_ok():
            return self._reply(401)
        body = self._body()
        if self.path == "/storage/v1/bucket":
            name = json.loads(body)["name"]
            if name in self.buckets:
                return self._reply(400, b'{"error":"Duplicate"}')
            self.buckets.add(name)
            return self._reply(200)
        bucket, key = self.path.removeprefix("/storage/v1/object/").split("/", 1)
        assert bucket in self.buckets
        self.objects[key] = (body, self.headers["Content-Type"])
        return self._reply(200)

    def do_GET(self):
        if not self._auth_ok():
            return self._reply(401)
        key = self.path.removeprefix("/storage/v1/object/receipts/")
        if key not in self.objects:
            return self._reply(400, b'{"error":"not_found"}')
        data, ctype = self.objects[key]
        return self._reply(200, data, ctype)

    def do_DELETE(self):
        if not self._auth_ok():
            return self._reply(401)
        for key in json.loads(self._body())["prefixes"]:
            self.objects.pop(key, None)
        return self._reply(200, b"[]")

    def log_message(self, *a):
        pass


@pytest.fixture()
def fake_url():
    FakeSupabase.objects, FakeSupabase.buckets, FakeSupabase.log = {}, set(), []
    srv = HTTPServer(("127.0.0.1", 0), FakeSupabase)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_save_load_delete_roundtrip(fake_url):
    s = storage.SupabaseStorage(fake_url, "service-key")
    s.save("20260925-abc.jpg", b"jpegbytes", "image/jpeg")
    s.save("20260925-def.pdf", b"pdfbytes", "application/pdf")
    assert FakeSupabase.buckets == {"receipts"}
    assert FakeSupabase.log.count(("POST", "/storage/v1/bucket")) == 1  # created once
    assert s.load("20260925-abc.jpg") == b"jpegbytes"
    assert FakeSupabase.objects["20260925-def.pdf"][1] == "application/pdf"
    s.delete("20260925-abc.jpg")
    with pytest.raises(FileNotFoundError):
        s.load("20260925-abc.jpg")


def test_existing_bucket_is_fine(fake_url):
    FakeSupabase.buckets.add("receipts")
    storage.SupabaseStorage(fake_url, "service-key").save("a.png", b"x", "image/png")
    assert "a.png" in FakeSupabase.objects


def test_from_config_picks_backend(tmp_path):
    local = storage.from_config({"UPLOAD_DIR": str(tmp_path)})
    assert isinstance(local, storage.LocalStorage)
    sb = storage.from_config({"UPLOAD_DIR": str(tmp_path), "SUPABASE_URL": "https://x.supabase.co",
                              "SUPABASE_SERVICE_KEY": "k"})
    assert isinstance(sb, storage.SupabaseStorage)


def test_local_storage_blocks_path_traversal(tmp_path):
    s = storage.LocalStorage(str(tmp_path / "r"))
    with pytest.raises(FileNotFoundError):
        s.load("../secret")
