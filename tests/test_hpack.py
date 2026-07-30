from __future__ import annotations

import random

import hpack
import pytest
from hpack.hpack import decode_integer as reference_decode_integer
from hpack.hpack import encode_integer as reference_encode_integer

from mojo_h2.exceptions import (
    HPACKDecodingError,
    InvalidTableSizeError,
    OversizedHeaderListError,
)
from mojo_h2.hpack import (
    Decoder,
    Encoder,
    HeaderTuple,
    NeverIndexedHeaderTuple,
    decode_integer,
    encode_integer,
)


@pytest.mark.parametrize("prefix_bits", range(1, 9))
def test_integer_codec_matches_hpack(prefix_bits):
    boundary = (1 << prefix_bits) - 1
    values = [0, 1, max(0, boundary - 1), boundary, boundary + 1, 127, 128,
              255, 4096, 2**31 - 1]
    for value in values:
        encoded = encode_integer(value, prefix_bits)
        reference = reference_encode_integer(value, prefix_bits)
        assert encoded == reference
        assert decode_integer(encoded, prefix_bits) == reference_decode_integer(
            reference, prefix_bits
        )


def test_integer_codec_validation_matches_upstream():
    with pytest.raises(ValueError):
        encode_integer(-1, 5)
    with pytest.raises(ValueError):
        encode_integer(1, 0)
    with pytest.raises(ValueError):
        decode_integer(b"\0", 9)
    with pytest.raises(HPACKDecodingError):
        decode_integer(b"\x1f\x80", 5)


def test_indexed_field_one_and_multi_byte_boundaries_match_upstream():
    ours = Encoder()
    upstream = hpack.Encoder()
    for index in (126, 127):
        entries = index - ours.header_table.STATIC_TABLE_LENGTH
        for entry in range(entries):
            name = f"x-{entry}".encode()
            value = b"value"
            ours.header_table.dynamic_entries.append((name, value))
            upstream.header_table.dynamic_entries.append((name, value))
        target = ours.header_table.get_by_index(index)
        assert ours.add(target, False) == upstream.add(target, False)
        ours.header_table.dynamic_entries.clear()
        upstream.header_table.dynamic_entries.clear()


@pytest.mark.parametrize(
    ("headers", "wire"),
    [
        (
            [(b":method", b"GET"), (b":scheme", b"http"), (b":path", b"/"),
             (b":authority", b"www.example.com")],
            "828684410f7777772e6578616d706c652e636f6d",
        ),
        (
            [(b"custom-key", b"custom-header")],
            "400a637573746f6d2d6b65790d637573746f6d2d686561646572",
        ),
    ],
)
def test_rfc_7541_non_huffman_examples(headers, wire):
    encoded = Encoder().encode(headers, huffman=False)
    assert encoded.hex() == wire
    assert Decoder().decode(encoded, raw=True) == headers


def test_rfc_7541_huffman_request_sequence():
    blocks = [
        (
            [(b":method", b"GET"), (b":scheme", b"http"), (b":path", b"/"),
             (b":authority", b"www.example.com")],
            "828684418cf1e3c2e5f23a6ba0ab90f4ff",
        ),
        (
            [(b":method", b"GET"), (b":scheme", b"http"), (b":path", b"/"),
             (b":authority", b"www.example.com"),
             (b"cache-control", b"no-cache")],
            "828684be5886a8eb10649cbf",
        ),
        (
            [(b":method", b"GET"), (b":scheme", b"https"),
             (b":path", b"/index.html"), (b":authority", b"www.example.com"),
             (b"custom-key", b"custom-value")],
            "828785bf408825a849e95ba97d7f8925a849e95bb8e8b4bf",
        ),
    ]
    encoder = Encoder()
    decoder = Decoder()
    for headers, expected in blocks:
        encoded = encoder.encode(headers)
        assert encoded.hex() == expected
        assert decoder.decode(encoded, raw=True) == headers


@pytest.mark.parametrize("huffman_enabled", [False, True])
def test_encoder_is_byte_identical_to_upstream_across_dynamic_blocks(
    huffman_enabled,
):
    ours = Encoder()
    theirs = hpack.Encoder()
    blocks = [
        [(":method", "GET"), (":scheme", "https"), (":path", "/"),
         (":authority", "example.com")],
        [(":method", "GET"), (":scheme", "https"), (":path", "/news"),
         (":authority", "example.com"), ("cache-control", "no-cache")],
        [(":status", "200"), ("content-type", "text/plain"),
         ("etag", '"abc"'), ("x-request-id", "123456")],
    ]
    for headers in blocks * 3:
        assert ours.encode(headers, huffman_enabled) == theirs.encode(
            headers, huffman_enabled
        )


def test_random_header_blocks_match_upstream_both_directions():
    rng = random.Random(7)
    names = ["x-id", "x-token", "cache-control", "content-type", "user-agent"]
    ours_encoder, upstream_encoder = Encoder(), hpack.Encoder()
    ours_decoder, upstream_decoder = Decoder(), hpack.Decoder()
    for block_index in range(80):
        headers = [
            (rng.choice(names), f"value-{rng.randrange(20)}-{block_index % 7}")
            for _ in range(rng.randrange(1, 10))
        ]
        ours = ours_encoder.encode(headers)
        upstream = upstream_encoder.encode(headers)
        assert ours == upstream
        assert ours_decoder.decode(upstream, raw=True) == upstream_decoder.decode(
            ours, raw=True
        )


def test_never_indexed_tuple_matches_upstream():
    ours = Encoder().encode(
        [NeverIndexedHeaderTuple(b"authorization", b"secret")]
    )
    upstream = hpack.Encoder().encode(
        [hpack.NeverIndexedHeaderTuple(b"authorization", b"secret")]
    )
    assert ours == upstream
    decoded = Decoder().decode(ours, raw=True)
    assert isinstance(decoded[0], NeverIndexedHeaderTuple)
    assert decoded == [(b"authorization", b"secret")]


def test_dict_pseudo_headers_are_moved_first_like_upstream():
    headers = {"accept": "*/*", ":path": "/", "user-agent": "test", ":method": "GET"}
    assert Encoder().encode(headers) == hpack.Encoder().encode(headers)


def test_table_size_update_and_eviction_match_upstream():
    ours, upstream = Encoder(), hpack.Encoder()
    ours.header_table_size = upstream.header_table_size = 128
    headers = [(b"x-long-name", b"a" * 80), (b"x-second", b"b" * 30)]
    assert ours.encode(headers) == upstream.encode(headers)
    assert list(ours.header_table.dynamic_entries) == list(
        upstream.header_table.dynamic_entries
    )


def test_decoder_rejects_invalid_huffman_padding():
    with pytest.raises(HPACKDecodingError):
        Decoder().decode(bytes.fromhex("4001618100"), raw=True)


def test_decoder_enforces_header_list_limit():
    block = Encoder().encode([(b"x", b"a" * 100)])
    with pytest.raises(OversizedHeaderListError):
        Decoder(max_header_list_size=32).decode(block)


def test_decoder_enforces_allowed_table_size():
    encoder = Encoder()
    encoder.header_table_size = 1024
    block = encoder.encode([])
    decoder = Decoder()
    decoder.max_allowed_table_size = 512
    with pytest.raises(InvalidTableSizeError):
        decoder.decode(block)


def test_unicode_and_raw_result_types():
    block = Encoder().encode([("x-name", "välue")])
    text = Decoder().decode(block)
    raw = Decoder().decode(block, raw=True)
    assert text == [("x-name", "välue")]
    assert isinstance(text[0], HeaderTuple)
    assert raw == [(b"x-name", "välue".encode())]
