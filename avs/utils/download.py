from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_url(url: str, dest: Path, *, sha256: str | None = None, overwrite: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not overwrite:
        if sha256 is None or sha256_file(dest) == sha256:
            return dest

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()

    # Some environments inject a localhost proxy env that is not reachable from this process.
    # If the proxy points to localhost, disable proxies for this download.
    proxy_env = " ".join(
        [
            os.environ.get("http_proxy", ""),
            os.environ.get("https_proxy", ""),
            os.environ.get("HTTP_PROXY", ""),
            os.environ.get("HTTPS_PROXY", ""),
            os.environ.get("ALL_PROXY", ""),
        ]
    ).lower()
    if "127.0.0.1" in proxy_env or "localhost" in proxy_env:
        urllib.request.install_opener(urllib.request.build_opener(urllib.request.ProxyHandler({})))

    urllib.request.urlretrieve(url, tmp)  # noqa: S310 - controlled URL in our code
    os.replace(tmp, dest)

    if sha256 is not None:
        got = sha256_file(dest)
        if got != sha256:
            raise ValueError(f"sha256 mismatch for {dest}: expected {sha256}, got {got}")
    return dest
