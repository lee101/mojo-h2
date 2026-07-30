from __future__ import annotations

import os
import struct

import numpy as np
import pytest
from hpack.huffman import HuffmanEncoder
from hpack.huffman_constants import REQUEST_CODES, REQUEST_CODES_LENGTH
from hpack.huffman_table import decode_huffman

from mojo_h2._lib import huffman_decode, huffman_encode
from mojo_h2 import decode_frame_headers, encode_frame_headers


@pytest.mark.parametrize(
    "data",
    [b"", b"www.example.com", b"custom-key", bytes(range(256)), os.urandom(4096)],
)
def test_huffman_kernels_match_upstream(data):
    reference_encoder = HuffmanEncoder(REQUEST_CODES, REQUEST_CODES_LENGTH)
    encoded = huffman_encode(data)
    assert encoded == reference_encoder.encode(data)
    assert huffman_decode(encoded) == decode_huffman(encoded) == data


@pytest.mark.parametrize("bad", [b"\xff", b"\xff\xff\xff\xff", b"\x00"])
def test_huffman_decoder_rejects_invalid_streams_like_upstream(bad):
    with pytest.raises(Exception):
        decode_huffman(bad)
    with pytest.raises(ValueError):
        huffman_decode(bad)


def test_batch_frame_header_codec_matches_struct():
    rng = np.random.default_rng(42)
    n = 10_000
    lengths = rng.integers(0, 2**24, n, dtype=np.int64)
    types = rng.integers(0, 256, n, dtype=np.uint8)
    flags = rng.integers(0, 256, n, dtype=np.uint8)
    streams = rng.integers(0, 2**32, n, dtype=np.int64)
    encoded = encode_frame_headers(lengths, types, flags, streams)
    reference = b"".join(
        struct.pack(
            "!HBBBL", (int(length) >> 8) & 0xFFFF, int(length) & 0xFF,
            int(frame_type), int(flag), int(stream) & 0x7FFFFFFF,
        )
        for length, frame_type, flag, stream
        in zip(lengths, types, flags, streams)
    )
    assert encoded == reference
    got = decode_frame_headers(encoded)
    assert np.array_equal(got[0], lengths)
    assert np.array_equal(got[1], types)
    assert np.array_equal(got[2], flags)
    assert np.array_equal(got[3], streams & 0x7FFFFFFF)


def test_batch_frame_header_validation():
    with pytest.raises(ValueError):
        encode_frame_headers([2**24], [1], [0], [1])
    with pytest.raises(ValueError):
        encode_frame_headers([1], [1, 2], [0], [1])
    with pytest.raises(ValueError):
        decode_frame_headers(b"short")
    with pytest.raises(ValueError):
        encode_frame_headers([1], [256], [0], [1])
    with pytest.raises(ValueError):
        encode_frame_headers([1], [1], [-1], [1])
    with pytest.raises(ValueError):
        encode_frame_headers([1], [1], [0], [-1])
    with pytest.raises(TypeError):
        encode_frame_headers([1.5], [1], [0], [1])
    with pytest.raises(ValueError):
        encode_frame_headers([[1]], [1], [0], [1])
    assert encode_frame_headers([], [], [], []) == b""


def test_batch_decoder_uses_buffer_nbytes_and_rejects_strides():
    headers = np.frombuffer(
        encode_frame_headers([1, 2], [3, 4], [5, 6], [7, 8]),
        dtype=np.uint8,
    ).reshape(2, 9)
    decoded = decode_frame_headers(memoryview(headers))
    assert decoded[0].tolist() == [1, 2]
    with pytest.raises(BufferError):
        decode_frame_headers(memoryview(headers[:, ::2]))
