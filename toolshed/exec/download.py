"""Fetching large files without losing them.

The download is the longest and most fragile leg of the whole install: tens of
gigabytes, over domestic broadband, from a CDN, possibly overnight. Every design
choice here comes from a way that goes wrong.

The central invariant: **a file under its final name is complete and verified.**
Everything is written to a ``.part`` beside the destination and only moved into
place once its hash matches. Nothing else in the system then has to wonder
whether a model on disk is trustworthy, and a resumed, interrupted or crashed
download can never leave a plausible-looking corrupt file behind.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

import httpx

CHUNK = 1024 * 1024
DEFAULT_ATTEMPTS = 5
# Re-check free space this often; a 40 GB download can fill a disk mid-flight.
SPACE_CHECK_EVERY = 256 * 1024 * 1024
# Ask for this much more room than the download needs.
SPACE_HEADROOM = 1.15


class DownloadError(RuntimeError):
    """A download failed in a way the user may be able to do something about."""

    def __init__(self, message: str, *, reason_key: str = "download_failed") -> None:
        super().__init__(message)
        self.reason_key = reason_key


class Cancelled(RuntimeError):
    """The user asked to stop. Not an error; the .part file survives."""


class _StalePart(Exception):
    """The .part on disk is longer than the file the server has.

    That happens when a model is re-published smaller under the same name, or
    when a .part from some other attempt is lying about. The server answers a
    Range past its end with 416, and taking that as "already have it all" --
    which is what this code did -- would install the stale bytes as the model.
    """


@dataclass
class Progress:
    filename: str
    downloaded: int
    total: int
    bytes_per_second: float = 0.0

    @property
    def fraction(self) -> float:
        return 0.0 if self.total <= 0 else min(1.0, self.downloaded / self.total)


ProgressFn = Callable[[Progress], None]
CancelFn = Callable[[], bool]


def _free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    return shutil.disk_usage(probe).free


def check_space(dest_dir: Path, needed: int) -> None:
    """Refuse before starting rather than dying at 90 percent."""
    if needed <= 0:
        return
    free = _free_bytes(dest_dir)
    want = int(needed * SPACE_HEADROOM)
    if free < want:
        raise DownloadError(
            f"Not enough space in {dest_dir}. Need about "
            f"{want / 1e9:.1f} GB, {free / 1e9:.1f} GB free.",
            reason_key="disk_full",
        )


def _total_from_416(response: httpx.Response) -> int:
    """The full length a 416 reports, as ``Content-Range: bytes */N``, or 0."""
    header = response.headers.get("content-range", "")
    _, _, tail = header.rpartition("/")
    return int(tail) if tail.strip().isdigit() else 0


def _headers(existing: int, token: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def remote_size(url: str, *, token: str | None = None, client: httpx.Client | None = None) -> int:
    """Content length, or 0 when the server will not say."""
    owned = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=30.0)
    try:
        r = client.head(url, headers=_headers(0, token))
        if r.status_code >= 400:
            return 0
        # HF reports the true object size here even when the body is a redirect.
        for header in ("x-linked-size", "content-length"):
            if header in r.headers:
                try:
                    return int(r.headers[header])
                except ValueError:
                    continue
        return 0
    except httpx.HTTPError:
        return 0
    finally:
        if owned:
            client.close()


def _stream_to_part(
    client: httpx.Client,
    url: str,
    part: Path,
    *,
    token: str | None,
    expected_total: int,
    on_progress: ProgressFn | None,
    should_cancel: CancelFn | None,
    dest_dir: Path,
) -> tuple[int, str]:
    """One attempt. Returns (bytes on disk, hex digest of the whole file)."""
    existing = part.stat().st_size if part.exists() else 0
    digest = hashlib.sha256()

    # Rehash what we already have. Cheaper than re-downloading, and it is the
    # only way a resumed transfer can end with a hash covering the whole file.
    if existing:
        with part.open("rb") as fh:
            for block in iter(lambda: fh.read(CHUNK), b""):
                digest.update(block)

    try:
        return _stream_response(client, url, part, existing, digest, token=token,
                                expected_total=expected_total, on_progress=on_progress,
                                should_cancel=should_cancel, dest_dir=dest_dir)
    except _StalePart:
        part.unlink(missing_ok=True)
        return _stream_to_part(client, url, part, token=token, expected_total=expected_total,
                               on_progress=on_progress, should_cancel=should_cancel,
                               dest_dir=dest_dir)


def _stream_response(
    client: httpx.Client,
    url: str,
    part: Path,
    existing: int,
    digest,
    *,
    token: str | None,
    expected_total: int,
    on_progress: ProgressFn | None,
    should_cancel: CancelFn | None,
    dest_dir: Path,
) -> tuple[int, str]:
    with client.stream("GET", url, headers=_headers(existing, token)) as response:
        if existing and response.status_code == 200:
            # The server ignored our Range and is sending the whole file again.
            existing, digest = 0, hashlib.sha256()
            part.unlink(missing_ok=True)
        elif existing and response.status_code == 416:
            # Either we already have everything the server has, or we have
            # *more* than it has -- and only the size tells those apart.
            total = expected_total or _total_from_416(response)
            if total and existing != total:
                raise _StalePart()
            return existing, digest.hexdigest()
        elif response.status_code not in (200, 206):
            raise DownloadError(
                _http_message(response.status_code, url),
                reason_key=_http_reason(response.status_code),
            )

        total = expected_total
        if not total:
            length = response.headers.get("content-length")
            if length and length.isdigit():
                total = int(length) + existing

        started, done, since_space_check = time.monotonic(), existing, 0
        mode = "ab" if existing else "wb"
        with part.open(mode) as fh:
            for chunk in response.iter_bytes(CHUNK):
                if should_cancel and should_cancel():
                    fh.flush()
                    os.fsync(fh.fileno())
                    raise Cancelled()
                fh.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                since_space_check += len(chunk)
                if since_space_check >= SPACE_CHECK_EVERY:
                    since_space_check = 0
                    if _free_bytes(dest_dir) < CHUNK * 64:
                        fh.flush()
                        raise DownloadError(
                            f"Ran out of space in {dest_dir} partway through.",
                            reason_key="disk_full",
                        )
                if on_progress:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    on_progress(Progress(part.stem, done, total,
                                         (done - existing) / elapsed))
            fh.flush()
            os.fsync(fh.fileno())
    return done, digest.hexdigest()


def _http_message(status: int, url: str) -> str:
    if status == 401:
        return "This model needs a free Hugging Face account."
    if status == 403:
        return "This model's author requires you to accept their terms first."
    if status == 404:
        return f"The file is no longer where we expected it: {url}"
    return f"The download server returned an error ({status})."


def _http_reason(status: int) -> str:
    return {401: "auth_required", 403: "terms_required", 404: "not_found"}.get(
        status, "http_error")


def download_file(
    url: str,
    dest: Path,
    *,
    sha256: str | None = None,
    size_bytes: int = 0,
    token: str | None = None,
    attempts: int = DEFAULT_ATTEMPTS,
    on_progress: ProgressFn | None = None,
    should_cancel: CancelFn | None = None,
    client: httpx.Client | None = None,
    part_dir: Path | None = None,
) -> str:
    """Fetch ``url`` to ``dest``, resuming and verifying. Returns the sha256.

    ``url`` must be the canonical address, never a redirect target: CDN links
    are presigned and expire, and the headline resume case is someone closing
    the app overnight. The redirect is followed afresh on every attempt.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # The .part must share a filesystem with the destination, or the final
    # os.replace stops being atomic and turns into a slow copy.
    part = (part_dir or dest.parent) / (dest.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and sha256 and file_digest(dest) == sha256:
        return sha256  # already here and verified; nothing to do

    # Room for what is still to come, not for the whole file: a resume with
    # 30 of 40 GB already on disk needs 10, and demanding 40 refused exactly
    # the machines that had made the most progress.
    already = part.stat().st_size if part.exists() else 0
    total = size_bytes or remote_size(url, token=token, client=client)
    check_space(dest.parent, max(0, total - already))

    owned = client is None
    client = client or httpx.Client(follow_redirects=True, timeout=httpx.Timeout(30.0, read=120.0))
    try:
        last: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                written, digest = _stream_to_part(
                    client, url, part, token=token, expected_total=size_bytes,
                    on_progress=on_progress, should_cancel=should_cancel,
                    dest_dir=dest.parent,
                )
            except Cancelled:
                raise
            except DownloadError as exc:
                # Authentication and missing files will not fix themselves.
                if exc.reason_key in {"auth_required", "terms_required", "not_found"}:
                    raise
                last = exc
            except (httpx.HTTPError, OSError) as exc:
                last = exc
            else:
                if sha256 and digest != sha256:
                    # Corrupt. Start clean rather than resuming onto bad bytes.
                    part.unlink(missing_ok=True)
                    last = DownloadError(
                        f"{dest.name} downloaded incorrectly.",
                        reason_key="checksum_mismatch")
                    continue
                if size_bytes and written != size_bytes:
                    # Discard it: resuming onto the wrong length can only
                    # produce the wrong length again, five times over.
                    part.unlink(missing_ok=True)
                    last = DownloadError(
                        f"{dest.name} is {written} bytes, expected {size_bytes}.",
                        reason_key="short_file")
                    continue
                os.replace(part, dest)   # atomic, and overwrites on Windows
                return digest

            if attempt < attempts:
                time.sleep(min(2 ** attempt, 30))

        raise DownloadError(
            f"Could not download {dest.name} after {attempts} attempts: {last}",
            reason_key=getattr(last, "reason_key", "download_failed"),
        )
    finally:
        if owned:
            client.close()


def file_digest(path: Path) -> str:
    """sha256 of a file on disk, read in chunks so a 20 GB model does not have
    to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_files(downloads, **kwargs) -> Iterator[tuple[str, str]]:
    """Fetch several files in order, yielding (filename, sha256) as each lands."""
    for item in downloads:
        digest = download_file(item.url, item.dest / item.filename,
                               sha256=item.sha256 if item.hash_is_frozen else None,
                               size_bytes=item.size_bytes, **kwargs)
        yield item.filename, digest
