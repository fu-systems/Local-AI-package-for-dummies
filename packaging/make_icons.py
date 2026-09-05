#!/usr/bin/env python3
"""Generate the application icons.

Pure standard library: no Pillow, which is on the PyInstaller exclude list and
has no business being a build dependency for two small images. Outputs are
committed, so this only needs rerunning when the design changes.

    python3 packaging/make_icons.py

Design: a shed. A dark slate ground, a warm amber roof and body, a door. It has
to survive being drawn at 16x16 in a Windows taskbar, so it is three bold shapes
and nothing else.
"""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent

BG = (26, 29, 35, 255)        # slate
ROOF = (232, 152, 58, 255)    # amber
BODY = (242, 197, 134, 255)   # light amber
DOOR = (26, 29, 35, 255)      # slate, punched back out of the body

ICO_SIZES = (16, 24, 32, 48, 64, 128, 256)


def render(size: int) -> bytes:
    """Return raw RGBA pixel rows for one square icon of the given size."""
    s = size
    px = bytearray()

    # Geometry in fractions of the canvas so every size is the same picture.
    margin = s * 0.10
    roof_apex_y = s * 0.20
    eaves_y = s * 0.46
    body_bottom = s - margin
    left, right = margin, s - margin
    door_w = s * 0.16
    door_top = s * 0.60

    corner = s * 0.18          # rounded corner radius of the slate ground

    for y in range(s):
        for x in range(s):
            cx, cy = x + 0.5, y + 0.5
            colour = (0, 0, 0, 0)

            # Rounded-square background.
            inside = True
            for ox, oy in ((corner, corner), (s - corner, corner),
                           (corner, s - corner), (s - corner, s - corner)):
                in_x = cx < corner if ox == corner else cx > s - corner
                in_y = cy < corner if oy == corner else cy > s - corner
                if in_x and in_y and (cx - ox) ** 2 + (cy - oy) ** 2 > corner ** 2:
                    inside = False
                    break
            if inside:
                colour = BG

                # Roof: an isosceles triangle from the apex down to the eaves.
                if roof_apex_y <= cy <= eaves_y:
                    t = (cy - roof_apex_y) / (eaves_y - roof_apex_y)
                    half = (right - left) / 2 * t
                    mid = s / 2
                    if mid - half <= cx <= mid + half:
                        colour = ROOF

                # Body.
                elif eaves_y < cy <= body_bottom:
                    inset = (right - left) * 0.09
                    if left + inset <= cx <= right - inset:
                        colour = BODY
                        if door_top <= cy <= body_bottom and abs(cx - s / 2) <= door_w / 2:
                            colour = DOOR

            px.extend(colour)

    # Prepend the mandatory PNG filter byte to each scanline.
    rows = bytearray()
    stride = s * 4
    for y in range(s):
        rows.append(0)
        rows.extend(px[y * stride:(y + 1) * stride])
    return bytes(rows)


def png(size: int) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)  # 8-bit RGBA
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", ihdr)
            + chunk(b"IDAT", zlib.compress(render(size), 9))
            + chunk(b"IEND", b""))


def ico(sizes: tuple[int, ...]) -> bytes:
    """Build a multi-resolution .ico. PNG-compressed entries are valid on Vista+."""
    images = [png(s) for s in sizes]
    header = struct.pack("<HHH", 0, 1, len(images))
    offset = len(header) + 16 * len(images)
    entries, blob = bytearray(), bytearray()
    for size, data in zip(sizes, images, strict=True):
        entries.extend(struct.pack(
            "<BBBBHHII",
            0 if size >= 256 else size,   # 0 means 256
            0 if size >= 256 else size,
            0, 0, 1, 32, len(data), offset,
        ))
        blob.extend(data)
        offset += len(data)
    return header + bytes(entries) + bytes(blob)


def main() -> int:
    win = HERE / "windows" / "toolshed.ico"
    lin = HERE / "linux" / "toolshed.png"
    win.parent.mkdir(parents=True, exist_ok=True)
    lin.parent.mkdir(parents=True, exist_ok=True)
    win.write_bytes(ico(ICO_SIZES))
    lin.write_bytes(png(256))
    print(f"wrote {win} ({win.stat().st_size} bytes)")
    print(f"wrote {lin} ({lin.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
