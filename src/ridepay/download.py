"""Download TLC files with a size check, so a cut-off transfer never looks like a complete month."""
from __future__ import annotations

import shutil
import time
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq


def download(url: str, dest: Path, chunk: int = 1 << 20, attempts: int = 4) -> Path:
    """Download to ``dest.part``, resuming with an HTTP Range request if the transfer is cut off;
    rename to ``dest`` only once the size matches Content-Length and the file reads as Parquet."""
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.unlink(missing_ok=True)
    expected = 0
    for attempt in range(1, attempts + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "ab" if have else "wb") as out:
                if have and resp.status != 206:      # server ignored the range: start again
                    out.truncate(0)
                    have = 0
                length = int(resp.headers.get("Content-Length") or 0)
                expected = expected or (have + length)
                shutil.copyfileobj(resp, out, chunk)
        except OSError as err:                       # includes timeouts and connection resets
            print(f"download attempt {attempt} interrupted: {err}", flush=True)
        if expected and tmp.exists() and tmp.stat().st_size == expected:
            break
        time.sleep(5 * attempt)
    got = tmp.stat().st_size if tmp.exists() else 0
    if not expected or got != expected:
        tmp.unlink(missing_ok=True)
        raise IOError(f"{url}: got {got} bytes, expected {expected}")
    if dest.suffix == ".parquet":
        pq.ParquetFile(tmp)  # raises if the file is not a readable Parquet file
    tmp.rename(dest)
    return dest


def parquet_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows
