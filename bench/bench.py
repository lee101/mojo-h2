"""mojo-h2 against hpack and hyperframe on identical wire data."""

from __future__ import annotations

import math
import os
import platform
import struct
import time

import numpy as np
from hpack import Encoder as ReferenceEncoder
from hpack.huffman import HuffmanEncoder
from hpack.huffman_constants import REQUEST_CODES, REQUEST_CODES_LENGTH
from hpack.huffman_table import decode_huffman
from hyperframe.frame import DataFrame as ReferenceDataFrame
from hyperframe.frame import Frame as ReferenceFrame

from mojo_h2 import Encoder, decode_frame_headers, encode_frame_headers
from mojo_h2._lib import huffman_decode, huffman_encode
from mojo_h2.frame import DataFrame


def timeit(function, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def machine() -> str:
    model = "unknown CPU"
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as source:
            for line in source:
                if line.startswith("model name"):
                    model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        model = platform.processor() or model
    return f"{model}; {platform.system()} {platform.release()}; Python {platform.python_version()}"


def main() -> None:
    reference_huffman = HuffmanEncoder(REQUEST_CODES, REQUEST_CODES_LENGTH)
    phrase = b"www.example.com: gzip, deflate; cache-control=no-cache\r\n"
    payload_size = 128 * 1024
    payload = (phrase * (payload_size // len(phrase) + 1))[:payload_size]
    encoded = huffman_encode(payload)
    assert encoded == reference_huffman.encode(payload)
    assert huffman_decode(encoded) == decode_huffman(encoded) == payload

    rng = np.random.default_rng(4)
    count = 250_000
    lengths = rng.integers(0, 2**24, count, dtype=np.int64)
    types = np.full(count, 0xF0, dtype=np.uint8)
    flags = rng.integers(0, 256, count, dtype=np.uint8)
    streams = rng.integers(0, 2**31, count, dtype=np.int64)

    def reference_frame_encode():
        return b"".join(
            struct.pack(
                "!HBBBL", (int(length) >> 8) & 0xFFFF, int(length) & 0xFF,
                int(frame_type), int(flag), int(stream),
            )
            for length, frame_type, flag, stream
            in zip(lengths, types, flags, streams)
        )

    packed = encode_frame_headers(lengths, types, flags, streams)
    assert packed == reference_frame_encode()

    def reference_frame_decode():
        for offset in range(0, len(packed), 9):
            ReferenceFrame.parse_frame_header(memoryview(packed[offset:offset + 9]))

    large_header = [(b"x-cookie", payload)]

    def mojo_large_hpack():
        return Encoder().encode(large_header)

    def reference_large_hpack():
        return ReferenceEncoder().encode(large_header)

    assert mojo_large_hpack() == reference_large_hpack()
    common = [
        (b":method", b"GET"), (b":scheme", b"https"), (b":path", b"/"),
        (b":authority", b"example.com"), (b"accept", b"*/*"),
    ]

    def mojo_indexed_blocks():
        encoder = Encoder()
        for _ in range(10_000):
            encoder.encode(common)

    def reference_indexed_blocks():
        encoder = ReferenceEncoder()
        for _ in range(10_000):
            encoder.encode(common)

    body = b"x" * 32

    def mojo_scalar_frames():
        for stream_id in range(1, 10_001):
            DataFrame(stream_id, body).serialize()

    def reference_scalar_frames():
        for stream_id in range(1, 10_001):
            ReferenceDataFrame(stream_id, body).serialize()

    cases = [
        (
            "Huffman encode, 128 KiB",
            lambda: huffman_encode(payload),
            lambda: reference_huffman.encode(payload),
            3,
        ),
        (
            f"Huffman decode, {len(encoded) / 1024:.0f} KiB",
            lambda: huffman_decode(encoded),
            lambda: decode_huffman(encoded),
            3,
        ),
        (
            "HPACK encode, one 128 KiB value",
            mojo_large_hpack,
            reference_large_hpack,
            3,
        ),
        (
            "HPACK encode, 10k indexed blocks",
            mojo_indexed_blocks,
            reference_indexed_blocks,
            3,
        ),
        (
            "Frame-header encode, 250k",
            lambda: encode_frame_headers(lengths, types, flags, streams),
            reference_frame_encode,
            3,
        ),
        (
            "Frame-header decode, 250k",
            lambda: decode_frame_headers(packed),
            reference_frame_decode,
            3,
        ),
        (
            "DATA serialize, 10k scalar",
            mojo_scalar_frames,
            reference_scalar_frames,
            3,
        ),
    ]

    print(f"Machine: {machine()}")
    print()
    print("| case | mojo-h2 | upstream | upstream / Mojo |")
    print("| --- | ---: | ---: | ---: |")
    for name, ours, upstream, repeat in cases:
        ours()
        upstream()
        mojo_seconds = timeit(ours, repeat)
        upstream_seconds = timeit(upstream, repeat)
        ratio = upstream_seconds / mojo_seconds
        label = "faster" if ratio >= 1 else "slower"
        print(
            f"| {name} | {mojo_seconds * 1000:.2f} ms | "
            f"{upstream_seconds * 1000:.2f} ms | {ratio:.2f}x {label} |"
        )


if __name__ == "__main__":
    main()
