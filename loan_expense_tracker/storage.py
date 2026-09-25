"""Receipt file storage: Supabase Storage when configured, else a local folder."""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from urllib.parse import quote


class LocalStorage:
    def __init__(self, folder: str):
        self.folder = folder
        os.makedirs(folder, exist_ok=True)

    def _path(self, name: str) -> str:
        path = os.path.realpath(os.path.join(self.folder, name))
        if not path.startswith(os.path.realpath(self.folder) + os.sep):
            raise FileNotFoundError(name)
        return path

    def save(self, name: str, data: bytes, mime: str):
        with open(self._path(name), "wb") as f:
            f.write(data)

    def load(self, name: str) -> bytes:
        with open(self._path(name), "rb") as f:
            return f.read()

    def delete(self, name: str):
        try:
            os.remove(self._path(name))
        except OSError:
            pass


class SupabaseStorage:
    """Private Supabase Storage bucket, accessed with the service-role key.

    Files are never public: the app streams them to signed-in users.
    """

    def __init__(self, url: str, key: str, bucket: str = "receipts"):
        self.base = url.rstrip("/") + "/storage/v1"
        self.key = key
        self.bucket = bucket
        self._bucket_ready = False

    def _request(self, method: str, path: str, data: bytes | None = None,
                 content_type: str = "application/json", extra: dict | None = None):
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.key}")
        req.add_header("apikey", self.key)
        if data is not None:
            req.add_header("Content-Type", content_type)
        for k, v in (extra or {}).items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.read()

    def _object(self, name: str) -> str:
        return f"/object/{self.bucket}/{quote(name)}"

    def _ensure_bucket(self):
        if self._bucket_ready:
            return
        body = json.dumps({"id": self.bucket, "name": self.bucket, "public": False}).encode()
        try:
            self._request("POST", "/bucket", body)
        except urllib.error.HTTPError as e:
            if e.code not in (400, 409):  # 400/409: bucket already exists
                raise
        self._bucket_ready = True

    def save(self, name: str, data: bytes, mime: str):
        self._ensure_bucket()
        self._request("POST", self._object(name), data, mime, {"x-upsert": "true"})

    def load(self, name: str) -> bytes:
        try:
            return self._request("GET", self._object(name))
        except urllib.error.HTTPError as e:
            if e.code in (400, 404):
                raise FileNotFoundError(name) from e
            raise

    def delete(self, name: str):
        body = json.dumps({"prefixes": [name]}).encode()
        try:
            self._request("DELETE", f"/object/{self.bucket}", body)
        except urllib.error.HTTPError:
            pass


def from_config(config) -> LocalStorage | SupabaseStorage:
    if config.get("SUPABASE_URL") and config.get("SUPABASE_SERVICE_KEY"):
        return SupabaseStorage(config["SUPABASE_URL"], config["SUPABASE_SERVICE_KEY"],
                               config.get("SUPABASE_BUCKET") or "receipts")
    return LocalStorage(config["UPLOAD_DIR"])
