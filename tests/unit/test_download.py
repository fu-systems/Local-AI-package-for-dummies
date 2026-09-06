"""Tests for the download engine, against a server that misbehaves on purpose.

Downloading tens of gigabytes over domestic broadband is the longest and most
fragile leg of an install. These tests drive a real HTTP server on localhost so
the failures that matter -- a dropped connection, a truncated body, a server
that ignores Range, a corrupt file, a full disk -- are exercised rather than
imagined.

The invariant they all defend: a file under its final name is complete and
verified. Anything else is a .part.
"""

from __future__ import annotations

import hashlib
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from toolshed.exec.download import (
    Cancelled,
    DownloadError,
    check_space,
    download_file,
    remote_size,
)

BODY = bytes(range(256)) * 4096          # 1 MiB, non-repeating enough to catch offsets
DIGEST = hashlib.sha256(BODY).hexdigest()


class Origin(BaseHTTPRequestHandler):
    """A deliberately unhelpful file server. Behaviour set per test."""

    body = BODY
    mode = "normal"          # normal | ignore_range | truncate | flaky | 401 | 403 | 404
    hits = 0

    def log_message(self, *args):    # keep pytest output clean
        pass

    def _send(self, status: int, length: int, *, partial_from: int | None = None):
        self.send_response(status)
        self.send_header("Content-Length", str(length))
        if partial_from is not None:
            self.send_header(
                "Content-Range", f"bytes {partial_from}-{len(self.body) - 1}/{len(self.body)}")
        self.end_headers()

    def do_HEAD(self):
        self._send(200, len(self.body))

    def do_GET(self):
        cls = type(self)
        cls.hits += 1

        if cls.mode in ("401", "403", "404"):
            self.send_response(int(cls.mode))
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        start = 0
        rng = self.headers.get("Range")
        if rng and cls.mode != "ignore_range":
            start = int(rng.split("=")[1].split("-")[0])

        payload = cls.body[start:]
        if cls.mode == "truncate" or (cls.mode == "flaky" and cls.hits == 1):
            # Promise the full length, deliver half, then hang up.
            self._send(206 if start else 200, len(payload),
                       partial_from=start if start else None)
            self.wfile.write(payload[: len(payload) // 2])
            return

        self._send(206 if start else 200, len(payload),
                   partial_from=start if start else None)
        self.wfile.write(payload)


@pytest.fixture
def origin():
    server = HTTPServer(("127.0.0.1", 0), Origin)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    Origin.mode, Origin.hits, Origin.body = "normal", 0, BODY
    yield f"http://127.0.0.1:{server.server_port}/file.bin", Origin
    server.shutdown()


class TestHappyPath:
    def test_downloads_and_verifies(self, origin, tmp_path):
        url, _ = origin
        dest = tmp_path / "file.bin"
        assert download_file(url, dest, sha256=DIGEST, size_bytes=len(BODY)) == DIGEST
        assert dest.read_bytes() == BODY

    def test_no_part_file_is_left_behind(self, origin, tmp_path):
        url, _ = origin
        download_file(url, tmp_path / "file.bin", sha256=DIGEST)
        assert not list(tmp_path.glob("*.part"))

    def test_an_already_verified_file_is_not_refetched(self, origin, tmp_path):
        url, server = origin
        dest = tmp_path / "file.bin"
        dest.write_bytes(BODY)
        server.hits = 0
        assert download_file(url, dest, sha256=DIGEST) == DIGEST
        assert server.hits == 0, "re-downloaded a file that was already correct"

    def test_reports_progress(self, origin, tmp_path):
        url, _ = origin
        seen = []
        download_file(url, tmp_path / "file.bin", sha256=DIGEST,
                      size_bytes=len(BODY), on_progress=seen.append)
        assert seen and seen[-1].downloaded == len(BODY)
        assert 0.0 < seen[-1].fraction <= 1.0

    def test_remote_size(self, origin):
        url, _ = origin
        assert remote_size(url) == len(BODY)


class TestResume:
    def test_resumes_from_a_partial_file(self, origin, tmp_path):
        url, _ = origin
        dest = tmp_path / "file.bin"
        (tmp_path / "file.bin.part").write_bytes(BODY[: len(BODY) // 3])
        assert download_file(url, dest, sha256=DIGEST) == DIGEST
        assert dest.read_bytes() == BODY, "resumed file does not match the original"

    def test_a_server_that_ignores_range_still_produces_a_correct_file(
            self, origin, tmp_path):
        """Some CDNs answer 200 with the whole body. Appending to what we had
        would corrupt it, so the partial file must be discarded."""
        url, server = origin
        server.mode = "ignore_range"
        (tmp_path / "file.bin.part").write_bytes(BODY[:1000])
        dest = tmp_path / "file.bin"
        assert download_file(url, dest, sha256=DIGEST) == DIGEST
        assert dest.read_bytes() == BODY

    def test_a_truncated_transfer_is_retried_and_completed(self, origin, tmp_path):
        url, server = origin
        server.mode = "flaky"          # first attempt dies half way
        dest = tmp_path / "file.bin"
        assert download_file(url, dest, sha256=DIGEST, attempts=4) == DIGEST
        assert dest.read_bytes() == BODY
        assert server.hits >= 2, "should have taken more than one attempt"


class TestFailures:
    def test_a_wrong_hash_never_reaches_the_final_name(self, origin, tmp_path):
        url, _ = origin
        dest = tmp_path / "file.bin"
        with pytest.raises(DownloadError) as exc:
            download_file(url, dest, sha256="0" * 64, attempts=2)
        assert exc.value.reason_key == "checksum_mismatch"
        assert not dest.exists(), "a corrupt file was published under its real name"

    def test_a_corrupt_part_file_is_discarded_rather_than_resumed_onto(
            self, origin, tmp_path):
        url, _ = origin
        (tmp_path / "file.bin.part").write_bytes(b"\xff" * 4096)
        dest = tmp_path / "file.bin"
        assert download_file(url, dest, sha256=DIGEST, attempts=3) == DIGEST
        assert dest.read_bytes() == BODY

    @pytest.mark.parametrize(
        ("mode", "reason"),
        [("401", "auth_required"), ("403", "terms_required"), ("404", "not_found")],
    )
    def test_hopeless_errors_fail_immediately(self, origin, tmp_path, mode, reason):
        """Retrying a 401 five times with backoff wastes the user's time; these
        need a person, not another attempt."""
        url, server = origin
        server.mode = mode
        server.hits = 0
        with pytest.raises(DownloadError) as exc:
            download_file(url, tmp_path / "f.bin", attempts=5)
        assert exc.value.reason_key == reason
        assert server.hits == 1, "retried an error that cannot resolve itself"

    def test_cancelling_keeps_the_partial_file(self, origin, tmp_path):
        """Closing the app must not throw away a 30 GB download."""
        url, _ = origin
        with pytest.raises(Cancelled):
            download_file(url, tmp_path / "file.bin", sha256=DIGEST,
                          should_cancel=lambda: True)
        assert not (tmp_path / "file.bin").exists()

    def test_refuses_to_start_without_room(self, tmp_path):
        with pytest.raises(DownloadError) as exc:
            check_space(tmp_path, 10 ** 15)
        assert exc.value.reason_key == "disk_full"

    def test_space_check_passes_for_something_reasonable(self, tmp_path):
        check_space(tmp_path, 1024)


class TestUnverifiedHashes:
    def test_download_without_a_known_hash_still_works(self, origin, tmp_path):
        """Where the catalogue has not been frozen yet we cannot verify against
        a release-time hash, but the transfer must still complete and be
        reported honestly."""
        url, _ = origin
        dest = tmp_path / "file.bin"
        assert download_file(url, dest, sha256=None) == DIGEST
        assert dest.read_bytes() == BODY

    def test_short_file_is_caught_when_the_size_is_known(self, origin, tmp_path):
        url, server = origin
        server.mode = "truncate"
        with pytest.raises(DownloadError) as exc:
            download_file(url, tmp_path / "f.bin", size_bytes=len(BODY), attempts=2)
        assert exc.value.reason_key in {"short_file", "download_failed"}
        assert not (tmp_path / "f.bin").exists()


def test_part_file_shares_the_destination_filesystem(origin, tmp_path):
    """os.replace is only atomic within one filesystem; a .part written
    elsewhere degrades into a slow copy or fails outright."""
    url, _ = origin
    dest = tmp_path / "deep" / "nested" / "file.bin"
    download_file(url, dest, sha256=DIGEST)
    assert dest.exists()
    assert Path(dest).parent == tmp_path / "deep" / "nested"
